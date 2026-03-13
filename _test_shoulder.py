import sqlite3, json, numpy as np
from scipy.sparse.csgraph import dijkstra

conn = sqlite3.connect('data/session_store.db')
row = conn.execute('SELECT rig_data FROM sessions WHERE session_id=?',
                   ('800652d6-9dd9-46b5-97a9-04ae4247577d',)).fetchone()
rig_data = json.loads(row[0])
verts = np.array(rig_data[0]['mesh']['vertices'], dtype=float)
faces = np.array(rig_data[0]['mesh']['faces'], dtype=np.int32)

from sam_3d_body.measurements.body_metrics import _build_mesh_adjacency
graph = _build_mesh_adjacency(verts, faces)

L, R = 7956, 6802
d, pred = dijkstra(graph, directed=False, indices=R, return_predecessors=True)
dist = d[L]
chord = np.linalg.norm(verts[L] - verts[R])
print(f'Direct geodesic R->L: {dist:.4f}m (chord={chord:.4f}m)')

path = []
cur = L
while cur != R and cur >= 0:
    path.append(cur)
    cur = int(pred[cur])
path.append(R)
path.reverse()
path_pts = verts[path]
print(f'Path has {len(path)} vertices')
print(f'Y range: {path_pts[:,1].min():.4f} to {path_pts[:,1].max():.4f}')
print(f'Z range: {path_pts[:,2].min():.4f} to {path_pts[:,2].max():.4f}')
print(f'X range: {path_pts[:,0].min():.4f} to {path_pts[:,0].max():.4f}')

# Check if path stays at shoulder height (Y~1.46)
y_min_threshold = verts[L][1] - 0.15
below = path_pts[:,1] < y_min_threshold
print(f'Points below shoulder-0.15m: {below.sum()} / {len(path)}')

conn.close()
