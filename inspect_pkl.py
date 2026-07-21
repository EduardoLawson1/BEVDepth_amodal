# inspect_pkl.py
import pickle
import os
import numpy as np
from pyquaternion import Quaternion

PKL       = "./data/custom/custom_infos_train.pkl"
DATA_ROOT = "Town01"   # mesmo valor de data_root no custom_exp.py

with open(PKL, "rb") as f:
    infos = pickle.load(f)

print(f"Total de frames: {len(infos)}")
print("=" * 60)

info = infos[0]

# ── [1] Chaves do info-dict ───────────────────────────────────
print("\n[1] Chaves do info-dict:")
for k, v in info.items():
    if k not in ("cam_infos", "ann_infos"):
        print(f"  {k}: {v}")

# ── [2] Câmeras ───────────────────────────────────────────────
print("\n[2] Câmeras:")
for cam_name, cam in info["cam_infos"].items():
    img_path = os.path.join(DATA_ROOT, cam["filename"])
    exists   = os.path.exists(img_path)
    cs       = cam["calibrated_sensor"]
    ep       = cam["ego_pose"]
    print(f"  {cam_name}")
    print(f"    filename  : {img_path}  [{'OK' if exists else 'MISSING'}]")
    print(f"    intrinsic : fx={cs['camera_intrinsic'][0][0]:.1f}  cx={cs['camera_intrinsic'][0][2]:.1f}")
    print(f"    s2e quat  : {cs['rotation']}  |norm|={np.linalg.norm(cs['rotation']):.4f}")
    print(f"    ego quat  : {ep['rotation']}  |norm|={np.linalg.norm(ep['rotation']):.4f}")

# ── [3] Anotações ─────────────────────────────────────────────
print("\n[3] Anotações (ann_infos):")
print(f"  Total de atores no frame: {len(info['ann_infos'])}")
for ann in info["ann_infos"][:5]:
    print(f"  [{ann['category_name']}]  translation={[round(v,2) for v in ann['translation']]}  size={[round(v,2) for v in ann['size']]}")

# ── [4] Simulação de get_gt() — posições em ego-frame ─────────
print("\n[4] Posições em ego-frame (simulação de get_gt):")
cam0    = list(info["cam_infos"].values())[0]
ego_q   = Quaternion(cam0["ego_pose"]["rotation"])
ego_t   = np.array(cam0["ego_pose"]["translation"])
rot_inv = ego_q.inverse.rotation_matrix
for ann in info["ann_infos"][:5]:
    c = rot_inv @ (np.array(ann["translation"]) - ego_t)
    print(f"  [{ann['category_name']}]  ego=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})  dist_xy={np.hypot(c[0],c[1]):.1f}m")

# ── [5] Sanidade global ───────────────────────────────────────
print("\n[5] Sanidade em todos os frames:")
missing, empty, class_counter, dists = 0, 0, {}, []

for info in infos:
    for cam in info["cam_infos"].values():
        if not os.path.exists(os.path.join(DATA_ROOT, cam["filename"])):
            missing += 1
    if not info["ann_infos"]:
        empty += 1
    cam0    = list(info["cam_infos"].values())[0]
    ego_q   = Quaternion(cam0["ego_pose"]["rotation"])
    ego_t   = np.array(cam0["ego_pose"]["translation"])
    rot_inv = ego_q.inverse.rotation_matrix
    for ann in info["ann_infos"]:
        class_counter[ann["category_name"]] = class_counter.get(ann["category_name"], 0) + 1
        c = rot_inv @ (np.array(ann["translation"]) - ego_t)
        dists.append(np.hypot(c[0], c[1]))

print(f"  Imagens faltando      : {missing}")
print(f"  Frames sem anotações  : {empty} / {len(infos)}")
print(f"  Distribuição de classes: {class_counter}")
if dists:
    print(f"  Dist XY min/mean/max  : {min(dists):.1f} / {np.mean(dists):.1f} / {max(dists):.1f} m")