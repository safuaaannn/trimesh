"""
Body Measurement Evaluation App  (v3)
======================================
Flat-table comparison: Ground Truth  vs  SMPL-X (normalised)  vs  MHR (raw).
No charts — just clean, readable tables grouped by body part.

Run:  streamlit run app.py
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────── constants ────────────────────────────────────

CSV_PATH = Path(__file__).parent / "measurement_database.csv"

# MHR key → SMPL-X key
MEASUREMENT_MAP: Dict[str, str] = {
    "arm_length":         "arm right length",
    "waist_girth":        "waist circumference",
    "bust_girth":         "chest circumference",
    "hip_girth":          "hip circumference",
    "inside_leg_height":  "inside leg height",
    "shoulder_to_crotch": "shoulder to crotch height",
    "shoulder_width":     "shoulder breadth",
    "thigh_girth":        "thigh left circumference",
}

PRETTY_LABELS: Dict[str, str] = {
    "arm_length":         "Arm Length",
    "waist_girth":        "Waist Girth",
    "bust_girth":         "Bust / Chest Girth",
    "hip_girth":          "Hip Girth",
    "inside_leg_height":  "Inside Leg Height",
    "shoulder_to_crotch": "Shoulder → Crotch",
    "shoulder_width":     "Shoulder Width",
    "thigh_girth":        "Thigh Girth",
}

BODY_PARTS: List[str] = list(MEASUREMENT_MAP.keys())

MASTER_COLUMNS = [
    "Name", "Height", "Gender", "Body Part",
    "Actual", "SMPLX Normalized", "MHR Raw",
    "SMPLX Error", "MHR Error",
    "SMPLX Error (%)", "MHR Error (%)",
]

DISPLAY_COLUMNS = [
    "Name", "Height", "Actual", "SMPLX Normalized",
    "MHR Raw", "SMPLX Error", "MHR Error",
    "SMPLX Error (%)", "MHR Error (%)",
]

CURRENT_DISPLAY = [
    "Body Part", "Actual", "SMPLX Normalized",
    "MHR Raw", "SMPLX Error", "MHR Error",
    "SMPLX Error (%)", "MHR Error (%)",
]


# ─────────────────────────────── API helpers ──────────────────────────────────

def call_smplx_api(
    url: str, image_bytes: bytes, filename: str,
    height_cm: float, gender: str,
) -> Optional[dict]:
    """POST to SMPL-X (SHAPY). Returns raw JSON or None."""
    try:
        resp = requests.post(
            url,
            files={"front_image": (filename, image_bytes, "image/jpeg")},
            data={"height_cm": str(height_cm), "gender": gender},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"❌ SMPL-X API error: {exc}")
        return None


def call_mhr_api(
    url: str, image_bytes: bytes, filename: str,
    height_cm: float,
) -> Optional[dict]:
    """POST to MHR (SAM-3D). Returns raw JSON or None."""
    try:
        resp = requests.post(
            url,
            files={"image": (filename, image_bytes, "image/jpeg")},
            data={"height_cm": str(height_cm)},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"❌ MHR API error: {exc}")
        return None


# ──────────────────────── response parsing ────────────────────────────────────

def parse_smplx(raw: dict) -> Tuple[Optional[float], Dict[str, float]]:
    """Returns (estimated_height_cm, {smplx_key: value_cm})."""
    try:
        front = raw["views"]["front"]["measurements"]
    except (KeyError, TypeError):
        st.error("❌ Could not parse SMPL-X response.")
        return None, {}
    height = front.get("height")
    if height is None:
        st.error("❌ SMPL-X response missing estimated height.")
        return None, {}
    meas: Dict[str, float] = {}
    for smplx_key in MEASUREMENT_MAP.values():
        v = front.get(smplx_key)
        if v is not None:
            meas[smplx_key] = float(v)
    return float(height), meas


def parse_mhr(raw: dict) -> Dict[str, float]:
    """Returns {mhr_key: value_cm}."""
    try:
        block = raw["measurements"]
    except (KeyError, TypeError):
        st.error("❌ Could not parse MHR response.")
        return {}
    out: Dict[str, float] = {}
    for mhr_key in MEASUREMENT_MAP:
        entry = block.get(mhr_key)
        if entry is None:
            continue
        val = entry.get("value_cm") if isinstance(entry, dict) else entry
        if val is not None:
            out[mhr_key] = float(val)
    return out


# ──────────────────────── comparison logic ────────────────────────────────────

def build_comparison(
    person_name: str,
    height_cm: float,
    gender: str,
    actual_tape: Dict[str, float],
    smplx_height: Optional[float],
    smplx_raw: Dict[str, float],
    mhr_raw: Dict[str, float],
) -> pd.DataFrame:
    """Normalise SMPL-X then compute errors against ground truth."""
    S = (height_cm / smplx_height) if (smplx_height and smplx_height > 0) else 1.0

    rows = []
    for mhr_key, smplx_key in MEASUREMENT_MAP.items():
        actual    = actual_tape.get(mhr_key, 0.0)
        smplx_val = smplx_raw.get(smplx_key)
        mhr_val   = mhr_raw.get(mhr_key)

        smplx_norm = round(smplx_val * S, 2) if smplx_val is not None else None

        # Errors against ground truth (None when no tape measurement)
        if actual and actual > 0:
            smplx_err     = round(abs(smplx_norm - actual), 2) if smplx_norm is not None else None
            mhr_err       = round(abs(mhr_val - actual), 2)    if mhr_val is not None else None
            smplx_err_pct = round(((smplx_norm - actual) / actual) * 100, 2) if smplx_norm is not None else None
            mhr_err_pct   = round(((mhr_val - actual) / actual) * 100, 2)    if mhr_val is not None else None
        else:
            smplx_err = None
            mhr_err   = None
            smplx_err_pct = None
            mhr_err_pct   = None

        rows.append({
            "Name":             person_name,
            "Height":           height_cm,
            "Gender":           gender,
            "Body Part":        PRETTY_LABELS[mhr_key],
            "Actual":           round(actual, 2) if actual else None,
            "SMPLX Normalized": smplx_norm,
            "MHR Raw":          round(mhr_val, 2) if mhr_val is not None else None,
            "SMPLX Error":      smplx_err,
            "MHR Error":        mhr_err,
            "SMPLX Error (%)": smplx_err_pct,
            "MHR Error (%)": mhr_err_pct,
        })

    return pd.DataFrame(rows, columns=MASTER_COLUMNS)


# ───────────────────── formatting / styling ───────────────────────────────────

def _fmt_pct(val) -> str:
    """Format percentage with explicit sign: +5.23% / -2.10% / —."""
    if pd.isna(val):
        return "—"
    return f"{val:+.2f}%"


def _highlight_winner(row):
    """Green background on the smaller error (cm + %) cells."""
    GREEN = "background-color: #c6efce; color: #006100"
    styles = [""] * len(row)
    se_idx = row.index.get_loc("SMPLX Error")
    me_idx = row.index.get_loc("MHR Error")
    se, me = row["SMPLX Error"], row["MHR Error"]
    if pd.notna(se) and pd.notna(me):
        # Compare absolute cm errors to decide winner
        if abs(se) < abs(me):
            styles[se_idx] = GREEN
            if "SMPLX Error (%)" in row.index:
                styles[row.index.get_loc("SMPLX Error (%)")] = GREEN
        elif abs(me) < abs(se):
            styles[me_idx] = GREEN
            if "MHR Error (%)" in row.index:
                styles[row.index.get_loc("MHR Error (%)")] = GREEN
    return styles


# ──────────────────────── CSV persistence ─────────────────────────────────────

# Canonical body-part order (matches UI iteration)
_PART_ORDER = {label: i for i, label in enumerate(PRETTY_LABELS.values())}


def load_master() -> pd.DataFrame:
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH)
            for col in MASTER_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            df = df[MASTER_COLUMNS]
            # Remove rows with empty/null names
            df = df[df["Name"].notna() & (df["Name"].astype(str).str.strip() != "")]
            # Deduplicate: keep last entry per (Name, Body Part)
            df = df.drop_duplicates(subset=["Name", "Body Part"], keep="last")

            # Recompute missing % errors from existing data (back-fills old CSVs)
            mask = df["SMPLX Error (%)"].isna() & df["Actual"].notna() & (df["Actual"] > 0)
            if mask.any():
                df.loc[mask & df["SMPLX Normalized"].notna(), "SMPLX Error (%)"] = (
                    ((df["SMPLX Normalized"] - df["Actual"]) / df["Actual"]) * 100
                ).round(2)
                df.loc[mask & df["MHR Raw"].notna(), "MHR Error (%)"] = (
                    ((df["MHR Raw"] - df["Actual"]) / df["Actual"]) * 100
                ).round(2)

            df = df.reset_index(drop=True)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=MASTER_COLUMNS)


def save_master(df: pd.DataFrame):
    # Sort by body-part order then name for a clean CSV
    df = df.copy()
    df["_sort"] = df["Body Part"].map(_PART_ORDER).fillna(999)
    df = df.sort_values(["_sort", "Name"]).drop(columns=["_sort"]).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)


# ──────────────────────────── Streamlit UI ────────────────────────────────────

def main():
    st.set_page_config(page_title="Body Measurement Evaluator", layout="wide")
    st.title("🧍 Body Measurement Evaluator")
    st.caption("Compare Ground Truth vs SMPL-X (SHAPY) vs MHR (SAM-3D)")

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ API Configuration")
        smplx_url = st.text_input("SMPL-X API URL", "http://192.168.86.6:8000/measure")
        mhr_url   = st.text_input("MHR API URL",   "http://localhost:8000/measure")

        st.divider()
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.master_df = pd.DataFrame(columns=MASTER_COLUMNS)
            if CSV_PATH.exists():
                CSV_PATH.unlink()
            st.success("All data cleared.")
            st.rerun()

        # ── Delete individual person ──────────────────────────────────────
        st.subheader("Manage Data")
        if "master_df" not in st.session_state:
            st.session_state.master_df = load_master()

        _master = st.session_state.master_df
        if _master.empty:
            st.caption("No data available to delete.")
        else:
            names = sorted(_master["Name"].dropna().unique().tolist())
            del_name = st.selectbox("Select person to delete", names, key="del_person")
            if st.button("🗑️ Delete Person Data", type="primary"):
                st.session_state.master_df = _master[_master["Name"] != del_name].reset_index(drop=True)
                save_master(st.session_state.master_df)
                st.success(f"Deleted all data for **{del_name}**.")
                st.rerun()

    # ── Load state (no-op if already loaded in sidebar block above) ───────
    if "master_df" not in st.session_state:
        st.session_state.master_df = load_master()

    # ── Input: basic info ─────────────────────────────────────────────────
    st.subheader("📝 Person Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        person_name = st.text_input("Person Name")
    with c2:
        height_cm = st.number_input("Actual Height (cm)", 100.0, 250.0, 170.0, 0.5)
    with c3:
        gender = st.selectbox("Gender", ["Male", "Female", "Neutral"])

    uploaded = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.image(uploaded, width=200)

    # ── Input: ground truth tape measurements ─────────────────────────────
    st.subheader("📏 Actual Tape Measurements (optional)")
    actual_tape: Dict[str, float] = {}
    row1 = st.columns(4)
    row2 = st.columns(4)
    tape_cols = row1 + row2
    for i, mhr_key in enumerate(BODY_PARTS):
        with tape_cols[i]:
            actual_tape[mhr_key] = st.number_input(
                PRETTY_LABELS[mhr_key] + " (cm)",
                min_value=0.0, max_value=300.0, value=0.0, step=0.5,
                key=f"tape_{mhr_key}",
            )

    run_btn = st.button("🚀 Run Analysis", type="primary",
                        disabled=not (person_name and uploaded))

    # ── Run analysis ──────────────────────────────────────────────────────
    if run_btn and uploaded and person_name:
        image_bytes = uploaded.getvalue()
        filename    = uploaded.name

        with st.spinner("📡 Calling both APIs…"):
            smplx_resp = call_smplx_api(smplx_url, image_bytes, filename,
                                        height_cm, gender.lower())
            mhr_resp   = call_mhr_api(mhr_url, image_bytes, filename, height_cm)

        if smplx_resp is None and mhr_resp is None:
            st.error("Both APIs failed — check sidebar URLs and server logs.")
            return

        # Parse
        smplx_height, smplx_meas = parse_smplx(smplx_resp) if smplx_resp else (None, {})
        mhr_meas = parse_mhr(mhr_resp) if mhr_resp else {}

        if smplx_resp and smplx_height is None:
            st.warning("⚠️ SMPL-X did not return height — showing un-normalised values.")

        # Build comparison table
        comparison = build_comparison(
            person_name, height_cm, gender,
            actual_tape, smplx_height, smplx_meas, mhr_meas,
        )

        if comparison.empty:
            st.warning("No measurements returned from either API.")
            return

        # ── Current run table ─────────────────────────────────────────────
        st.subheader(f"📊 Results for {person_name}")
        display = comparison[CURRENT_DISPLAY].copy()
        styled = display.style.apply(_highlight_winner, axis=1).format(
            _fmt_pct, subset=["SMPLX Error (%)", "MHR Error (%)"],
        ).format(precision=2, na_rep="—", subset=[
            c for c in display.columns
            if c not in ("SMPLX Error (%)", "MHR Error (%)")
        ])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Persist — upsert: overwrite existing rows for this person
        merged = pd.concat(
            [st.session_state.master_df, comparison], ignore_index=True
        )
        merged = merged.drop_duplicates(
            subset=["Name", "Body Part"], keep="last"
        ).reset_index(drop=True)
        st.session_state.master_df = merged
        save_master(merged)
        st.success(f"✅ Saved {len(comparison)} rows for **{person_name}**.")

    # ── Historical data grouped by body part ──────────────────────────────
    st.divider()
    st.header("📋 Historical Data — Grouped by Body Part")

    master = st.session_state.master_df
    if master.empty:
        st.info("No data yet — run an analysis above to populate this section.")
    else:
        for mhr_key in BODY_PARTS:
            label    = PRETTY_LABELS[mhr_key]
            filtered = master[master["Body Part"] == label]
            if filtered.empty:
                continue

            st.subheader(f"{label} — All People")
            cols = [c for c in DISPLAY_COLUMNS if c in filtered.columns]
            pct_cols = [c for c in cols if c.endswith("(%)")]
            num_cols = [c for c in cols if c not in pct_cols]
            styled = filtered[cols].style.apply(
                _highlight_winner, axis=1
            ).format(_fmt_pct, subset=pct_cols
            ).format(precision=2, na_rep="—", subset=num_cols)
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # Download
        st.divider()
        csv_bytes = master.to_csv(index=False).encode()
        st.download_button("⬇️ Download Master CSV", csv_bytes,
                           "measurement_database.csv", "text/csv")


if __name__ == "__main__":
    main()
