# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
from collections import namedtuple
from os import path as osp

import mmcv
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter as nuscenes_converter

ClassInfo = namedtuple("ClassInfo", ["train_id", "id", "name", "category", "type", "color"])
id_to_trainid_info = {
    0:   ClassInfo(255, 0,"void or ignore", "void", "void", (0, 0, 0)),
    1:   ClassInfo(255, 1, "void or ignore", "void.outlier", "void", (0, 0, 0)),
    10:  ClassInfo(0,  10, "car (thing)", "vehicle.car", "thing", (100, 150, 245)),
    11:  ClassInfo(1,  11, "bicycle (thing)", "vehicle.bicycle", "thing", (100, 230, 245)),
    13:  ClassInfo(4,  13, "bus (thing)", "vehicle.bus", "thing", (100, 80, 250)),
    15:  ClassInfo(2,  15, "motorcycle (thing)", "vehicle.motorcycle", "thing", (30, 60, 150)),
    16:  ClassInfo(4,  16, "on rails (thing)", "vehicle.on_rails", "thing", (0, 0, 255)),
    18:  ClassInfo(3,  18, "truck (thing)", "vehicle.truck", "thing", (80, 30, 180)),
    20:  ClassInfo(4,  20, "other vehicle (thing)", "vehicle.other", "thing", (0, 0, 255)),
    30:  ClassInfo(5,  30, "person (thing)", "human.pedestrian", "thing", (255, 30, 30)),
    31:  ClassInfo(6,  31, "bicyclist (thing)", "human.rider.bicyclist", "thing", (255, 40, 200)),
    32:  ClassInfo(7,  32, "motorcyclist (thing)", "human.rider.motorcyclist", "thing", (150, 30, 90)),
    40:  ClassInfo(8,  40, "road (stuff)", "flat.road", "stuff", (255, 0, 255)),
    44:  ClassInfo(9,  44, "parking (stuff)", "flat.parking", "stuff", (255, 150, 255)),
    48:  ClassInfo(10, 48, "sidewalk (stuff)", "flat.sidewalk", "stuff", (75, 0, 75)),
    49:  ClassInfo(11, 49, "other ground (stuff)", "flat.other_ground", "stuff", (175, 0, 75)),
    50:  ClassInfo(12, 50, "building (stuff)", "construction.building", "stuff", (255, 200, 0)),
    51:  ClassInfo(13, 51, "fence (stuff)", "construction.fence", "stuff", (255, 120, 50)),
    52:  ClassInfo(255, 52, "void or ignore", "construction.other", "void", (0, 0, 0)),
    60:  ClassInfo(8,   60, "lane marking (stuff)", "flat.lane_marking", "stuff", (150, 255, 170)),
    70:  ClassInfo(14, 70, "vegetation (stuff)", "static.vegetation", "stuff", (0, 175, 0)),
    71:  ClassInfo(15, 71, "trunk (stuff)", "static.trunk", "stuff", (135, 60, 0)),
    72:  ClassInfo(16, 72, "terrain (stuff)", "flat.terrain", "stuff", (150, 240, 80)),
    80:  ClassInfo(17, 80, "pole (stuff)", "static.pole", "stuff", (255, 240, 150)),
    81:  ClassInfo(18, 81, "traffic sign (stuff)", "static.traffic_sign", "stuff", (255, 0, 0)),
    99:  ClassInfo(255, 99, "void or ignore", "object.other", "void", (0, 0, 0)),
    252: ClassInfo(0,   252, "moving car (thing)", "vehicle.car", "thing", (100, 150, 245)),
    253: ClassInfo(6,   253, "moving bicyclist (thing)", "human.rider.bicyclist", "thing", (255, 40, 200)),
    254: ClassInfo(5,   254, "moving person (thing)", "human.pedestrian", "thing", (255, 30, 30)),
    255: ClassInfo(7,   255, "moving motorcyclist (thing)", "human.rider.motorcyclist", "thing", (150, 30, 90)),
    256: ClassInfo(4,   256, "moving on rails (thing)", "vehicle.on_rails", "thing", (0, 0, 255)),
    257: ClassInfo(4,   257, "moving bus (thing)", "vehicle.bus", "thing", (100, 80, 250)),
    258: ClassInfo(3,   258, "moving truck (thing)", "vehicle.truck", "thing", (80, 30, 180)),
    259: ClassInfo(4,   259, "moving other vehicle (thing)", "vehicle.other", "thing", (0, 0, 255)),
}

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def read_calib(calib_path):
    calib_all = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                break
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    out = {}
    for k in ("P0", "P1", "P2", "P3"):
        M = np.eye(4)
        M[:3, :4] = calib_all[k].reshape(3, 4)
        out[k] = M

    Tr = np.eye(4)
    Tr[:3, :4] = calib_all["Tr"].reshape(3, 4)
    out["Tr"] = Tr
    return out

def decompose_K_t_from_P(P):
    # P is 3x4
    # For rectified KITTI cameras, R is identity and P = K [I | t]
    fx, fy = P[0, 0], P[1, 1]
    cx, cy = P[0, 2], P[1, 2]
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=float)
    p4 = P[:, 3].reshape(3)
    t = np.linalg.inv(K) @ p4
    R = np.eye(3, dtype=float)
    return K, R, t

def sensor2lidar_from_calib(calib_path, camera_key="P2"):
    C = read_calib(calib_path)
    # T_cam0<-lidar
    T_cam0_from_lidar = C["Tr"]

    # Build T_cami<-cam0 from Pi
    P = C[camera_key][:3, :4]
    K, R_ci_from_c0, t_ci_from_c0 = decompose_K_t_from_P(P)
    T_ci_from_c0 = np.eye(4)
    T_ci_from_c0[:3, :3] = R_ci_from_c0
    T_ci_from_c0[:3, 3] = t_ci_from_c0

    # Compose to get T_cami<-lidar
    T_ci_from_lidar = T_ci_from_c0 @ T_cam0_from_lidar

    # Invert to get T_lidar<-cami which is sensor2lidar
    T_lidar_from_ci = np.linalg.inv(T_ci_from_lidar)

    R_sensor2lidar = T_lidar_from_ci[:3, :3]
    t_sensor2lidar = T_lidar_from_ci[:3, 3]
    return R_sensor2lidar, t_sensor2lidar, K

def sensor_xforms_from_calib(calib_path, camera_key="P2"):
    """
    Returns:
      R_sensor2lidar  3x3
      t_sensor2lidar  3,
      K               3x3 intrinsics of the selected camera
      R_sensor2ego    3x3  where ego is cam0
      t_sensor2ego    3,
    """
    C = read_calib(calib_path)

    # T_cam0_from_lidar
    T_cam0_from_lidar = C["Tr"]

    # Build T_cami_from_cam0 from Pi for the requested camera
    P = C[camera_key][:3, :4]
    K, R_ci_from_c0, t_ci_from_c0 = decompose_K_t_from_P(P)
    T_ci_from_c0 = np.eye(4)
    T_ci_from_c0[:3, :3] = R_ci_from_c0
    T_ci_from_c0[:3, 3] = t_ci_from_c0

    # Compose to get T_cami_from_lidar
    T_ci_from_lidar = T_ci_from_c0 @ T_cam0_from_lidar

    # Invert to get T_lidar_from_cami which is sensor to lidar
    T_lidar_from_ci = np.linalg.inv(T_ci_from_lidar)
    R_sensor2lidar = T_lidar_from_ci[:3, :3]
    t_sensor2lidar = T_lidar_from_ci[:3, 3]

    # Invert T_cami_from_c0 to get T_cam0_from_cami which is sensor to ego
    T_c0_from_ci = np.linalg.inv(T_ci_from_c0)
    R_sensor2ego = T_c0_from_ci[:3, :3]
    t_sensor2ego = T_c0_from_ci[:3, 3]

    return R_sensor2lidar, t_sensor2lidar, K, R_sensor2ego, t_sensor2ego

def parse_poses(filename):
  """ read poses file with per-scan poses from given filename

      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []


  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    poses.append(pose)

  return poses

def parse_times(filename):
    """Read KITTI times.txt and return list of timestamps (in seconds)."""
    with open(filename, "r") as f:
        times = [float(line.strip()) for line in f if line.strip()]
    return np.array(times)

def parse_ego2global(T_global_from_ego):
    R_sensor2global =  T_global_from_ego[:3, :3]
    t_sensor2global =  T_global_from_ego[:3, 3]
    return R_sensor2global, t_sensor2global

def semantic_kitti_data_prep(root_path, info_prefix, version, max_sweeps=10, debug=True):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    split = version.split('-')[1]
    if split == 'train':
        seq_ids = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    elif split == 'val':
        seq_ids = [8]
    elif split == 'trainval':
        seq_ids = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 8]
    
    camera_names = {'image_2':'P2', 'image_3':'P3'}
    
    
    data_infos = {'infos': [], 'metadata': {'version':version}}
    for seq_id in seq_ids:
        print(f'Processing seq {seq_id:02d}')
        lidar_dir = os.path.join(root_path, 'dataset', 'sequences', f'{seq_id:02d}', 'velodyne')
        lidar_files = sorted(os.listdir(lidar_dir))
        calib_path = os.path.join(root_path, 'dataset', 'sequences', f'{seq_id:02d}', 'calib.txt')
        poses_path = os.path.join(root_path, 'dataset', 'sequences', f'{seq_id:02d}', 'poses.txt')
        times_path = os.path.join(root_path, 'dataset', 'sequences', f'{seq_id:02d}', 'times.txt')
        parsed_poses = parse_poses(poses_path)
        parsed_times = parse_times(times_path)
        calib = read_calib(calib_path)
        for i, lidar_file in enumerate(lidar_files):
            info = {}
            lidar_path = os.path.join(lidar_dir, lidar_file)
            info['lidar_path'] = lidar_path
            info['scene_token'] = f'{seq_id:02d}'
            frame_id = lidar_file.split('.')[0]
            info['token'] = frame_id + f'_{seq_id:02d}'
            info['lidar_token'] = info['token']
            seq_id_ = f'{seq_id:02d}'
            label_id =  frame_id
            pan_gt_path =  osp.join("/home/mohan/hpcgpu11/occformer_data/SemanticKITTI/panoptic_labels/", seq_id_, "labels",
                                                f"{label_id}.label")

            pan_gt  = np.fromfile(pan_gt_path, dtype=np.uint32)
            pan_gt = pan_gt & 0xFFFF
            unique_pan_gt = np.unique(pan_gt)
            # add_flag = False
            # class_gts = set()
            # for ix in unique_pan_gt:
            #     if id_to_trainid_info[ix].train_id == 255:
            #         continue
            #     if id_to_trainid_info[ix].type == "stuff" and np.sum(pan_gt == ix) > 50 and id_to_trainid_info[ix].id in [49,]:
            #         add_flag = True
            #     class_gts.add(id_to_trainid_info[ix].name)

            # if not add_flag:
            #     continue

            info['sweeps'] = []
            info['timestamp'] = parsed_times[i] * 1e6  # in microseconds
            info['cams'] = {}
            for cam, cam_key in camera_names.items():
                R_sensor2lidar, t_sensor2lidar, K, R_sensor2ego, t_sensor2ego = sensor_xforms_from_calib(calib_path, camera_key=cam_key)
                R_ego2global, t_ego2global = parse_ego2global(parsed_poses[i])
                cam_path = lidar_file.replace('velodyne', cam).replace('.bin', '.png')
                cam_path = os.path.join(root_path, 'dataset', 'sequences', f'{seq_id:02d}', cam, cam_path)
                info['cams'][f'CAM_{cam[-1]}'] = {
                    'data_path': cam_path,
                    'type': cam,
                    'sample_data_token': info['token'],
                    'sensor2lidar_rotation': R_sensor2lidar,
                    'sensor2lidar_translation': t_sensor2lidar,
                    'sensor2ego_translation': t_sensor2ego,
                    'sensor2ego_rotation': R_sensor2ego,
                    'ego2global_rotation': R_ego2global,
                    'ego2global_translation': t_ego2global,
                    'cam_intrinsic': K,
                    'timestamp': info['timestamp']
                }
            info['lidar2ego_rotation'] = calib['Tr'][:3, :3]
            info['lidar2ego_translation'] = calib['Tr'][:3, 3] 
            info['ego2global_rotation'] = parsed_poses[i][:3, :3]
            info['ego2global_translation'] = parsed_poses[i][:3, 3]
            info['gt_boxes'] = []
            info['gt_names'] = class_gts
            info['num_lidar_pts'] = []
            info['num_radar_pts'] = []
            info['valid_flag'] = []
            info['gt_velocity'] = []
            info['ann_infos'] = []
            data_infos['infos'].append(info)
    info_path = os.path.join(root_path,
                                '{}_infos_{}.pkl'.format(info_prefix, split))
    mmcv.dump(data_infos, info_path)
    return 1


# def add_ann_adj_info(extra_tag):
#     nuscenes_version = 'v1.0-trainval'
#     dataroot = './data/nuscenes/'
#     nuscenes = NuScenes(nuscenes_version, dataroot)
#     for set in ['train', 'val']:
#         dataset = pickle.load(
#             open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
#         for id in range(len(dataset['infos'])):
#             if id % 10 == 0:
#                 print('%d/%d' % (id, len(dataset['infos'])))
#             info = dataset['infos'][id]
#             # get sweep adjacent frame info
#             sample = nuscenes.get('sample', info['token'])
#             ann_infos = list()
#             for ann in sample['anns']:
#                 ann_info = nuscenes.get('sample_annotation', ann)
#                 velocity = nuscenes.box_velocity(ann_info['token'])
#                 if np.any(np.isnan(velocity)):
#                     velocity = np.zeros(3)
#                 ann_info['velocity'] = velocity
#                 ann_infos.append(ann_info)
#             dataset['infos'][id]['ann_infos'] = ann_infos
#             dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
#             dataset['infos'][id]['scene_token'] = sample['scene_token']

#             scene = nuscenes.get('scene', sample['scene_token'])
#             dataset['infos'][id]['occ_path'] = \
#                 './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
#         with open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
#                   'wb') as fid:
#             pickle.dump(dataset, fid)


if __name__ == '__main__':
    dataset = 'nuscenes'
    version = 'v2.0'
    train_version = f'{version}-val'
    root_path = '/home/mohan/hpcgpu11/occformer_data/SemanticKITTI/'
    extra_tag = 'semantic_kitti-v3.0'
    semantic_kitti_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=train_version,
        max_sweeps=0)

    # print('add_ann_infos')
    # add_ann_adj_info(extra_tag)
