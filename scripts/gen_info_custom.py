"""
gen_info_custom.py
==================
Gera os arquivos .pkl usados pelo BEVDepth/CarlaDetDataset a partir do
formato Town0n (dataset CARLA customizado).

Uso:
    # Rode da raiz do repo BEVDepth (importante para os paths relativos)
    python gen_info_custom.py \
        --data-root .          \
        --out-dir   ./data/custom \
        --train-split 0.8

    # Se quiser limitar o range de detecção (padrão nuScenes: 51.2m)
    python gen_info_custom.py --data-root . --max-range 51.2

Saída:
    <out-dir>/custom_infos_train.pkl
    <out-dir>/custom_infos_val.pkl

Formato do pkl (compatível com CarlaDetDataset):
{
    "sample_token":  str,            # "Town01/seq_0001/0000002493"
    "scene_token":   str,            # "Town01/seq_0001"
    "frame_id":      str,            # "0000002493"
    "timestamp":     float,          # sim_time_s
    "ego2global_translation": [...], # nuScenes coords
    "ego2global_rotation":    [...], # quaternion [w,x,y,z]
    "cam_infos": {
        "CAM_FRONT": {
            "filename":          str,   # relativo à raiz do repo
            "depth_filename":    str | None,
            "calibrated_sensor": {
                "camera_intrinsic": [[...]], # 3x3
                "rotation":    [w,x,y,z],   # sensor→ego
                "translation": [x,y,z],     # sensor→ego, nuScenes coords
            },
            "ego_pose": {
                "rotation":    [w,x,y,z],   # ego→world
                "translation": [x,y,z],     # ego→world, nuScenes coords
            },
            "timestamp": float,
        },
        ... # CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
    },
    "ann_infos": [
        {
            "category_name": str,       # "car", "pedestrian", etc.
            "translation":   [x,y,z],   # world coords, nuScenes
            "size":          [w,l,h],   # metros
            "rotation":      [w,x,y,z], # world coords, nuScenes
            "velocity":      [vx,vy,vz],
            "track_id":      int,
        },
        ...
    ],
    "lidar_path": str | None,
}

Convenções de coordenadas
--------------------------
CARLA usa Left-Handed (X-forward, Y-right, Z-up) com rotações RPY em graus.
nuScenes usa Right-Handed (X-forward, Y-left, Z-up).

Conversão aplicada:
  world_nuscenes = (X, -Y, Z)    ← inverte apenas Y
  yaw_nuscenes   = -yaw_carla    ← inverte sinal do yaw (e pitch)

CarlaDetDataset.get_gt() recebe translation/rotation em world coords e faz
a transformação world→ego internamente via Quaternion, por isso salvamos
os atores em world coords (não ego).
"""

import argparse
import json
import math
import os
import pickle
from glob import glob
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# ─── Mapeamento câmera local → chave nuScenes ────────────────────────────────
CAM_MAP = {
    "front":       "CAM_FRONT",
    "front_left":  "CAM_FRONT_LEFT",
    "front_right": "CAM_FRONT_RIGHT",
    "rear":        "CAM_BACK",
    "rear_left":   "CAM_BACK_LEFT",
    "rear_right":  "CAM_BACK_RIGHT",
}

# ─── Classes do dataset ───────────────────────────────────────────────────────
CLASSES = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]

# semantic_class em frames.jsonl → classe final
# "vehicle" é refinado via blueprint_id do actors.json
SEMANTIC_CLASS_MAP = {
    # Valores reais de semantic_class em frames.jsonl (confirmado via diagnóstico)
    "pedestrian": "pedestrian",
    "walker":     "pedestrian",   # actor_type alternativo, por segurança
    "car":        "car",
    "truck":      "truck",
    "bus":        "bus",
    "motorcycle": "motorcycle",
    "bicycle":    "bicycle",
    "vehicle":    "car",          # fallback caso alguma sequência use o genérico
}
KNOWN_CLASSES = set(SEMANTIC_CLASS_MAP.keys())

# blueprint_id (prefixo) → refinamento de classe.
# Agora só é necessário para quando semantic_class == "vehicle" (fallback acima).
# Para "car"/"truck"/"bus" etc. já vêm corretos do frames.jsonl.
BLUEPRINT_PREFIX_MAP = {
    "vehicle.carlamotors.firetruck":  "truck",
    "vehicle.carlamotors.carlacola":  "truck",
    "vehicle.ford.ambulance":         "truck",
    "vehicle.volkswagen.t2":          "bus",
    "vehicle.mitsubishi.fusorosa":    "bus",
    "vehicle.diamondback":            "bicycle",
    "vehicle.gazelle":                "bicycle",
    "vehicle.bh":                     "bicycle",
    "vehicle.harley":                 "motorcycle",
    "vehicle.kawasaki":               "motorcycle",
    "vehicle.yamaha":                 "motorcycle",
    "vehicle.vespa":                  "motorcycle",
}


# ─── Utilitários de coordenadas ───────────────────────────────────────────────

def carla_xyz_to_nuscenes(xyz) -> np.ndarray:
    """CARLA (X-fwd, Y-right, Z-up, LH) → nuScenes (X-fwd, Y-left, Z-up, RH): inverte Y."""
    return np.array([xyz[0], -xyz[1], xyz[2]], dtype=np.float64)


def rpy_deg_carla_to_R(rpy_deg) -> np.ndarray:
    """
    RPY em graus CARLA (LH, roll-pitch-yaw) → matriz de rotação 3x3 nuScenes (RH).
    Inversão de handedness: negar pitch e yaw antes de construir a matriz ZYX.
    """
    roll  = math.radians(rpy_deg[0])
    pitch = math.radians(-rpy_deg[1])   # nega
    yaw   = math.radians(-rpy_deg[2])   # nega

    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ], dtype=np.float64)


def R_to_quat(R: np.ndarray) -> List[float]:
    """Matriz de rotação 3x3 → quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return [0.25 / s,
                (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                0.25 * s,                 (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s, 0.25 * s]


def world_to_ego_xy_dist(world_xyz_ns, ego_xyz_ns, ego_R) -> float:
    """Distância euclidiana XY (ego-frame) entre ator e ego."""
    delta = ego_R.T @ (world_xyz_ns - ego_xyz_ns)
    return math.hypot(delta[0], delta[1])


# ─── Leitura de calibração ────────────────────────────────────────────────────

def load_camera_calibration(cameras_json_path: str) -> Dict:
    """
    Lê calib/cameras.json e retorna dict { cam_name → calib }.
    Cada entrada contém:
        intrinsic              : np.ndarray (3,3)
        sensor2ego_translation : np.ndarray (3,)  nuScenes coords
        sensor2ego_rotation    : [w,x,y,z]
        img_width, img_height  : int
    """
    with open(cameras_json_path) as f:
        data = json.load(f)

    result = {}
    for cam in data["cameras"]:
        name = cam["name"]
        intr = cam["intrinsics"]
        extr = cam["extrinsics"]

        K = np.array([
            [intr["fx"], 0,          intr["cx"]],
            [0,          intr["fy"], intr["cy"]],
            [0,          0,          1         ],
        ], dtype=np.float64)

        t = carla_xyz_to_nuscenes(extr["location_xyz_m"])
        R = rpy_deg_carla_to_R(extr["rotation_rpy_deg"])

        result[name] = {
            "intrinsic":              K,
            "sensor2ego_translation": t,
            "sensor2ego_rotation":    R_to_quat(R),
            "img_width":              intr["width"],
            "img_height":             intr["height"],
        }
    return result


# ─── Leitura de actors estáticos ──────────────────────────────────────────────

def load_static_actors(actors_json_path: str) -> Dict[int, Dict]:
    """
    Lê scene_manifest/actors.json e retorna { track_id → metadados }.
    bbox_extent_xyz_m são half-extents → multiplicamos por 2.
    """
    with open(actors_json_path) as f:
        data = json.load(f)

    registry = {}
    for actor in data["actors"]:
        tid = actor["track_id"]
        ext = actor["bbox_extent_xyz_m"]   # [half_x, half_y, half_z]
        registry[tid] = {
            "blueprint_id": actor.get("blueprint_id", "").lower(),
            # [length(X)*2, width(Y)*2, height(Z)*2]
            "bbox_lwh": np.array([ext[0]*2, ext[1]*2, ext[2]*2], dtype=np.float32),
        }
    return registry


# ─── Leitura de frames.jsonl ──────────────────────────────────────────────────

def iter_frames(frames_jsonl_path: str):
    """Itera sobre frames.jsonl retornando um dict por linha."""
    with open(frames_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ─── Construção de um info-dict por frame ────────────────────────────────────

def build_frame_info(
    frame: Dict,
    cam_calib: Dict,
    static_actors: Dict[int, Dict],
    seq_path: str,
    scene_token: str,
    repo_root: str,
    max_range_m: float = 51.2,
) -> Optional[Dict]:
    """
    Converte um record de frames.jsonl no info-dict que CarlaDetDataset espera.

    Args:
        repo_root  : Caminho absoluto da raiz do repo BEVDepth.
                     Usado para gerar filenames relativos ao repo (que é o
                     data_root passado ao CarlaDetDataset).
        max_range_m: Descarta atores além desse raio XY (ego-frame).
                     Padrão: 51.2m (igual ao point_cloud_range do nuScenes).
    """
    frame_id = str(frame["frame_id"]).zfill(10)

    # ── Pose do ego em nuScenes coords ──────────────────────────────────────
    ego      = frame["ego"]
    ego_xyz  = carla_xyz_to_nuscenes(ego["world_xyz_m"])
    ego_R    = rpy_deg_carla_to_R(ego["world_rpy_deg"])
    ego_quat = R_to_quat(ego_R)

    # ── Câmeras ──────────────────────────────────────────────────────────────
    cam_infos = {}
    for cam_name, nusc_key in CAM_MAP.items():
        if cam_name not in cam_calib:
            continue

        calib = cam_calib[cam_name]

        # filename relativo à raiz do repo (= data_root do CarlaDetDataset)
        abs_rgb   = os.path.join(seq_path, "cameras", cam_name, "rgb",   f"{frame_id}.png")
        abs_depth = os.path.join(seq_path, "cameras", cam_name, "depth", f"{frame_id}.npz")
        rel_rgb   = os.path.relpath(abs_rgb,   repo_root)
        rel_depth = os.path.relpath(abs_depth, repo_root)

        cam_infos[nusc_key] = {
            # CarlaDetDataset: img_path = os.path.join(data_root, cam_info["filename"])
            "filename":       rel_rgb,
            "depth_filename": rel_depth if os.path.exists(abs_depth) else None,
            # Extrínseca câmera→ego
            "calibrated_sensor": {
                "camera_intrinsic": calib["intrinsic"].tolist(),
                "rotation":         calib["sensor2ego_rotation"],          # [w,x,y,z]
                "translation":      calib["sensor2ego_translation"].tolist(),
            },
            # Pose ego→world
            "ego_pose": {
                "rotation":    ego_quat,
                "translation": ego_xyz.tolist(),
            },
            "timestamp": frame.get("sim_time_s", 0.0),
        }

    # ── LiDAR ────────────────────────────────────────────────────────────────
    abs_lidar = os.path.join(seq_path, "lidar", "top", f"{frame_id}.npz")
    lidar_path = os.path.relpath(abs_lidar, repo_root) if os.path.exists(abs_lidar) else None

    # ── Anotações (world coords — get_gt() transforma para ego internamente) ─
    ann_infos = []
    for actor in frame.get("actors", []):
        if not actor.get("alive", True):
            continue

        tid       = actor.get("track_id")
        sem_class = actor.get("semantic_class", "").lower()

        if sem_class not in KNOWN_CLASSES:
            continue

        # Classe: pedestrian direto; vehicle refinado via blueprint
        mapped = SEMANTIC_CLASS_MAP[sem_class]
        if mapped == "car" and tid in static_actors:
            bp = static_actors[tid]["blueprint_id"]
            for prefix, cls in BLUEPRINT_PREFIX_MAP.items():
                if bp.startswith(prefix):
                    mapped = cls
                    break

        # Filtro de range XY no frame ego
        actor_xyz = carla_xyz_to_nuscenes(actor["world_xyz_m"])
        if world_to_ego_xy_dist(actor_xyz, ego_xyz, ego_R) > max_range_m:
            continue

        # Dimensões [w, l, h] — CarlaDetDataset.get_gt() usa size[[1,0,2]] → [l,w,h]
        # Nosso bbox_lwh = [length, width, height], então size = [width, length, height]
        if tid in static_actors:
            lwh  = static_actors[tid]["bbox_lwh"]      # [l, w, h]
            size = [float(lwh[1]), float(lwh[0]), float(lwh[2])]  # → [w, l, h]
        else:
            size = [0.5, 1.0, 1.7]

        # Rotação → quaternion nuScenes world
        actor_quat = R_to_quat(rpy_deg_carla_to_R(actor["world_rpy_deg"]))

        # Velocidade nuScenes world
        vel = carla_xyz_to_nuscenes(actor.get("velocity_xyz_mps", [0.0, 0.0, 0.0]))

        ann_infos.append({
            "category_name": mapped,
            "translation":   actor_xyz.tolist(),   # world coords, nuScenes
            "size":          size,                  # [w, l, h]
            "rotation":      actor_quat,            # [w, x, y, z] world
            "velocity":      vel.tolist(),
            "track_id":      tid,
        })

    return {
        "sample_token":           f"{scene_token}/{frame_id}",
        "scene_token":            scene_token,
        "frame_id":               frame_id,
        "timestamp":              frame.get("sim_time_s", 0.0),
        "ego2global_translation": ego_xyz.tolist(),
        "ego2global_rotation":    ego_quat,
        "cam_infos":              cam_infos,
        "ann_infos":              ann_infos,
        "lidar_path":             lidar_path,
    }


# ─── Processamento de uma sequência ──────────────────────────────────────────

def process_sequence(seq_path: str, repo_root: str, max_range_m: float) -> List[Dict]:
    """Processa todos os frames de uma sequência e retorna lista de info-dicts."""
    frames_jsonl = os.path.join(seq_path, "scene_manifest", "frames.jsonl")
    actors_json  = os.path.join(seq_path, "scene_manifest", "actors.json")
    cameras_json = os.path.join(seq_path, "calib", "cameras.json")

    if not os.path.exists(frames_jsonl):
        print(f"  [SKIP] frames.jsonl não encontrado: {frames_jsonl}")
        return []
    if not os.path.exists(cameras_json):
        print(f"  [SKIP] cameras.json não encontrado: {cameras_json}")
        return []

    cam_calib     = load_camera_calibration(cameras_json)
    static_actors = load_static_actors(actors_json) if os.path.exists(actors_json) else {}

    # scene_token: 2 últimos componentes do path (ex: "Town01/seq_0001")
    parts       = os.path.normpath(seq_path).replace("\\", "/").split("/")
    scene_token = "/".join(parts[-2:])

    infos = []
    for frame in iter_frames(frames_jsonl):
        info = build_frame_info(
            frame, cam_calib, static_actors,
            seq_path, scene_token, repo_root, max_range_m,
        )
        if info is not None:
            infos.append(info)
    return infos


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gera pkl de infos para BEVDepth/CarlaDetDataset (formato Town0n/CARLA)."
    )
    parser.add_argument(
        "--data-root", type=str, default=".",
        help="Raiz do repo BEVDepth (onde ficam Town01/, Town02/, etc.). "
             "Deve ser o mesmo valor passado como data_root ao CarlaDetDataset. "
             "Default: '.' (diretório corrente).",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./data/custom",
        help="Diretório de saída dos .pkl. Default: ./data/custom",
    )
    parser.add_argument(
        "--train-split", type=float, default=0.8,
        help="Fração das sequências para treino (split por sequência, não por frame). "
             "Default: 0.8",
    )
    parser.add_argument(
        "--max-range", type=float, default=51.2,
        help="Raio máximo XY (m, frame ego) para manter uma anotação. "
             "Igual ao point_cloud_range do nuScenes. Use 'inf' para desativar. "
             "Default: 51.2",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.data_root)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Descoberta de sequências (aceita 3 layouts de diretório) ─────────────
    # (A) <data_root>/seq_*               ex: --data-root ./Town01
    # (B) <data_root>/*/seq_*             ex: --data-root .  (múltiplos towns)
    # (C) <data_root>/*/*/seq_*           ex: com nível extra
    sequences = sorted(glob(os.path.join(repo_root, "seq_*")))
    if not sequences:
        sequences = sorted(glob(os.path.join(repo_root, "*", "seq_*")))
    if not sequences:
        sequences = sorted(glob(os.path.join(repo_root, "*", "*", "seq_*")))
    if not sequences:
        raise RuntimeError(f"Nenhuma sequência encontrada em {repo_root}")

    print(f"Encontradas {len(sequences)} sequências em {repo_root}")

    # ── Split por sequência (evita data-leakage) ──────────────────────────────
    split_idx  = max(1, int(len(sequences) * args.train_split))
    train_seqs = sequences[:split_idx]
    val_seqs   = sequences[split_idx:]

    print(f"  Treino : {len(train_seqs)} sequência(s)")
    print(f"  Val    : {len(val_seqs)} sequência(s)")
    print(f"  Range  : {args.max_range} m")

    # ── Processamento ─────────────────────────────────────────────────────────
    def collect(seqs, label):
        all_infos = []
        for seq in tqdm(seqs, desc=label):
            infos = process_sequence(seq, repo_root, args.max_range)
            all_infos.extend(infos)
            print(f"    {os.path.relpath(seq, repo_root)}: {len(infos)} frames")
        return all_infos

    train_infos = collect(train_seqs, "Treino")
    val_infos   = collect(val_seqs,   "Val")

    # ── Salvar ────────────────────────────────────────────────────────────────
    train_pkl = os.path.join(args.out_dir, "custom_infos_train.pkl")
    val_pkl   = os.path.join(args.out_dir, "custom_infos_val.pkl")

    with open(train_pkl, "wb") as f:
        pickle.dump(train_infos, f)
    with open(val_pkl, "wb") as f:
        pickle.dump(val_infos, f)

    print(f"\nPronto!")
    print(f"  Treino : {len(train_infos)} frames → {train_pkl}")
    print(f"  Val    : {len(val_infos)} frames → {val_pkl}")


if __name__ == "__main__":
    main()