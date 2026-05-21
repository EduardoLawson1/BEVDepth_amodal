import argparse
import csv
import json
import os
from glob import glob

import mmcv
import numpy as np


CAMERA_NAME_MAP = {
	'front': 'CAM_FRONT',
	'front_left': 'CAM_FRONT_LEFT',
	'front_right': 'CAM_FRONT_RIGHT',
	'rear': 'CAM_BACK',
	'rear_left': 'CAM_BACK_LEFT',
	'rear_right': 'CAM_BACK_RIGHT',
}


def euler_deg_to_quat_wxyz(roll_deg, pitch_deg, yaw_deg):
	"""Convert roll-pitch-yaw in degrees to quaternion [w, x, y, z].

	Rotation order is ZYX (yaw -> pitch -> roll), which is the common
	convention for vehicle pose composition.
	"""
	roll = np.deg2rad(roll_deg)
	pitch = np.deg2rad(pitch_deg)
	yaw = np.deg2rad(yaw_deg)

	cy = np.cos(yaw * 0.5)
	sy = np.sin(yaw * 0.5)
	cp = np.cos(pitch * 0.5)
	sp = np.sin(pitch * 0.5)
	cr = np.cos(roll * 0.5)
	sr = np.sin(roll * 0.5)

	w = cr * cp * cy + sr * sp * sy
	x = sr * cp * cy - cr * sp * sy
	y = cr * sp * cy + sr * cp * sy
	z = cr * cp * sy - sr * sp * cy
	return [float(w), float(x), float(y), float(z)]


def build_camera_intrinsic(camera_intrinsics):
	fx = float(camera_intrinsics['fx'])
	fy = float(camera_intrinsics['fy'])
	cx = float(camera_intrinsics['cx'])
	cy = float(camera_intrinsics['cy'])
	return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def load_state_by_frame(state_csv_path):
	state_by_frame = {}
	with open(state_csv_path, 'r', newline='') as f:
		reader = csv.DictReader(f)
		for row in reader:
			frame_id = int(row['frame_id'])
			pos = [
				float(row['pos_x']),
				float(row['pos_y']),
				float(row['pos_z']),
			]
			roll = float(row['roll'])
			pitch = float(row['pitch'])
			yaw = float(row['yaw'])
			quat = euler_deg_to_quat_wxyz(roll, pitch, yaw)
			state_by_frame[frame_id] = {
				'translation': pos,
				'rotation': quat,
				'yaw_deg': yaw,
			}
	return state_by_frame


def load_calib(calib_path):
	with open(calib_path, 'r') as f:
		raw = json.load(f)
	cam_data = {}
	for cam in raw['cameras']:
		cam_name = cam['name']
		ext = cam['extrinsics']
		rpy = ext['rotation_rpy_deg']
		quat = euler_deg_to_quat_wxyz(rpy[0], rpy[1], rpy[2])
		cam_data[cam_name] = {
			'height': int(cam['intrinsics']['height']),
			'width': int(cam['intrinsics']['width']),
			'camera_intrinsic': build_camera_intrinsic(cam['intrinsics']),
			'translation': [float(v) for v in ext['location_xyz_m']],
			'rotation': quat,
		}
	return cam_data


def load_lidar_calib(lidar_calib_path):
	with open(lidar_calib_path, 'r') as f:
		raw = json.load(f)
	sensor = raw['sensors'][0]
	ext = sensor['extrinsics']
	rpy = ext['rotation_rpy_deg']
	quat = euler_deg_to_quat_wxyz(rpy[0], rpy[1], rpy[2])
	return {
		'translation': [float(v) for v in ext['location_xyz_m']],
		'rotation': quat,
		'camera_intrinsic': [],
	}


def collect_frame_ids(seq_root, camera_names):
	per_cam_frame_ids = []
	for cam_name in camera_names:
		rgb_glob = os.path.join(seq_root, 'cameras', cam_name, 'rgb', '*.png')
		depth_glob = os.path.join(seq_root, 'cameras', cam_name, 'depth', '*.npz')
		rgb_ids = {int(os.path.splitext(os.path.basename(p))[0]) for p in glob(rgb_glob)}
		depth_ids = {int(os.path.splitext(os.path.basename(p))[0]) for p in glob(depth_glob)}
		per_cam_frame_ids.append(rgb_ids & depth_ids)

	lidar_glob = os.path.join(seq_root, 'lidar', 'top', '*.npz')
	lidar_ids = {int(os.path.splitext(os.path.basename(p))[0]) for p in glob(lidar_glob)}

	ann_glob = os.path.join(seq_root, 'boxes_3d', 'frames', '*.json')
	ann_ids = {int(os.path.splitext(os.path.basename(p))[0]) for p in glob(ann_glob)}

	all_sets = per_cam_frame_ids + [lidar_ids, ann_ids]
	valid_ids = sorted(set.intersection(*all_sets))
	return valid_ids


def frame_id_to_timestamp_us(frame_id, base_frame_id, hz=10.0):
	frame_offset = frame_id - base_frame_id
	return int(frame_offset * (1e6 / hz))



def is_valid_instance(inst):
    """Filter out corrupted actor states."""
    cw = inst.get('center_world_xyz_m', [0, 0, 0])
    
    # NaN check
    if any(v != v for v in cw):
        return False
    
    # Near-origin check: valid world coords in Town01 are never this close to [0,0,0]
    # if np.linalg.norm(cw) < 5.0:
    #     return False
    
    # Zero size check
    size = inst.get('size_xyz_m', [0, 0, 0])
    if any(s <= 0 for s in size):
        return False
    
    return True


def build_ann_infos(frame_json_path):
	with open(frame_json_path, 'r') as f:
		frame_data = json.load(f)

	ann_infos = []
	for inst in frame_data.get('instances', []):
		if not is_valid_instance(inst):
			continue

		category_name = str(inst.get('semantic_class', '')).lower()

		translation = inst.get('center_world_xyz_m', [0.0, 0.0, 0.0])
		size = inst.get('size_xyz_m', [0.0, 0.0, 0.0])
		yaw_world_deg = float(inst.get('yaw_world_deg', 0.0))
		rotation = euler_deg_to_quat_wxyz(0.0, 0.0, yaw_world_deg)
		velocity = inst.get('velocity_world_xyz_mps', [0.0, 0.0, 0.0])

		ann_infos.append({
			'token': f"track_{inst.get('track_id', -1)}_frame_{frame_data.get('frame_id', -1)}",
			'sample_token': None,
			'instance_token': f"track_{inst.get('track_id', -1)}",
			'visibility_token': '4',
			'attribute_tokens': [],
			'translation': [float(v) for v in translation],
			'size': [float(v) for v in size],
			'rotation': rotation,
			'prev': '',
			'next': '',
			'num_lidar_pts': 0,
			'num_radar_pts': 0,
			'category_name': category_name,
			'velocity': [float(v) for v in velocity],
		})
	return ann_infos


def frame_has_valid_annotations(sequence_root, frame_id):
	"""Return True if a frame has at least one valid instance annotation."""
	ann_path = os.path.join(sequence_root, 'boxes_3d', 'frames',
							f'{frame_id:010d}.json')
	if not os.path.exists(ann_path):
		return False
	ann_infos = build_ann_infos(ann_path)
	return len(ann_infos) > 0





def build_single_info(frame_id,
					  seq_root,
					  seq_name,
					  base_frame_id,
					  state_by_frame,
					  cam_calib,
					  lidar_calib,
					  include_lidar_info=True):
	timestamp_us = frame_id_to_timestamp_us(frame_id, base_frame_id)
	sample_token = f'{seq_name}_{frame_id:010d}'
	scene_token = seq_name

	if frame_id not in state_by_frame:
		raise KeyError(f'Frame {frame_id} not found in state.csv')

	ego_pose = {
		'token': f'ego_{sample_token}',
		'timestamp': timestamp_us,
		'rotation': state_by_frame[frame_id]['rotation'],
		'translation': state_by_frame[frame_id]['translation'],
	}

	cam_infos = {}
	for raw_cam_name, nus_cam_name in CAMERA_NAME_MAP.items():
		calib = cam_calib[raw_cam_name]
		rgb_rel = os.path.join('cameras', raw_cam_name, 'rgb', f'{frame_id:010d}.png')
		depth_rel = os.path.join('cameras', raw_cam_name, 'depth', f'{frame_id:010d}.npz')

		cam_infos[nus_cam_name] = {
			'sample_token': sample_token,
			'ego_pose': ego_pose,
			'timestamp': timestamp_us,
			'is_key_frame': True,
			'height': calib['height'],
			'width': calib['width'],
			'filename': rgb_rel,
			'depth_filename': depth_rel,
			'calibrated_sensor': {
				'token': f'calib_{nus_cam_name}',
				'sensor_token': nus_cam_name,
				'translation': calib['translation'],
				'rotation': calib['rotation'],
				'camera_intrinsic': calib['camera_intrinsic'],
			},
		}

	if include_lidar_info:
		lidar_infos = {
			'LIDAR_TOP': {
				'sample_token': sample_token,
				'ego_pose': ego_pose,
				'timestamp': timestamp_us,
				'filename': os.path.join('lidar', 'top', f'{frame_id:010d}.npz'),
				'calibrated_sensor': {
					'token': 'calib_LIDAR_TOP',
					'sensor_token': 'LIDAR_TOP',
					'translation': lidar_calib['translation'],
					'rotation': lidar_calib['rotation'],
					'camera_intrinsic': [],
				},
			}
		}
	else:
		lidar_infos = {}

	ann_path = os.path.join(seq_root, 'boxes_3d', 'frames', f'{frame_id:010d}.json')
	ann_infos = build_ann_infos(ann_path)
	for ann in ann_infos:
		ann['sample_token'] = sample_token

	return {
		'sample_token': sample_token,
		'timestamp': timestamp_us,
		'scene_token': scene_token,
		'cam_infos': cam_infos,
		'lidar_infos': lidar_infos,
		'cam_sweeps': [],
		'lidar_sweeps': [],
		'ann_infos': ann_infos,
	}


def generate_info(sequence_root,
				  train_ratio=0.8,
				  include_lidar_info=True,
				  min_frame_count=2):
	seq_name = os.path.basename(sequence_root.rstrip('/'))
	state_csv = os.path.join(sequence_root, 'state.csv')
	cam_calib_path = os.path.join(sequence_root, 'calib', 'cameras.json')
	lidar_calib_path = os.path.join(sequence_root, 'calib', 'lidar.json')
 
	state_by_frame = load_state_by_frame(state_csv)
	cam_calib = load_calib(cam_calib_path)
	lidar_calib = load_lidar_calib(lidar_calib_path)
 
	frame_ids = collect_frame_ids(sequence_root, list(CAMERA_NAME_MAP.keys()))
 
	# Filter: frame must exist in state.csv AND have at least one valid annotation
	frame_ids = [
		fid for fid in frame_ids
		if fid in state_by_frame and frame_has_valid_annotations(sequence_root, fid)
	]
 
	if len(frame_ids) < min_frame_count:
		raise RuntimeError(
			f'Not enough valid frames in {sequence_root}. Found {len(frame_ids)}')
 
	print(f'Valid frames after filtering: {len(frame_ids)}')
 
	base_frame_id = frame_ids[0]
	infos = []
	for frame_id in frame_ids:
		infos.append(
			build_single_info(frame_id,
							  sequence_root,
							  seq_name,
							  base_frame_id,
							  state_by_frame,
							  cam_calib,
							  lidar_calib,
							  include_lidar_info=include_lidar_info))
 
	split_idx = int(len(infos) * train_ratio)
	split_idx = min(max(split_idx, 1), len(infos) - 1)
	train_infos = infos[:split_idx]
	val_infos = infos[split_idx:]
	return train_infos, val_infos


def main():
	parser = argparse.ArgumentParser(
		description='Generate CARLA infos pkl for BEVDepth training.')
	parser.add_argument('--sequence-root',
						type=str,
						default='./Town01/seq_0001',
						help='Path to a CARLA sequence folder, e.g. Town01/seq_0001')
	parser.add_argument('--output-dir',
						type=str,
						default='./data/town01',
						help='Output directory for generated pkl files.')
	parser.add_argument('--train-ratio',
						type=float,
						default=0.8,
						help='Train split ratio in [0, 1].')
	parser.add_argument('--without-lidar-info',
						action='store_true',
						help='Do not include lidar_infos in generated pkl.')
	args = parser.parse_args()

	if args.train_ratio <= 0.0 or args.train_ratio >= 1.0:
		raise ValueError('--train-ratio must be in (0, 1).')

	mmcv.mkdir_or_exist(args.output_dir)
	train_infos, val_infos = generate_info(
		sequence_root=args.sequence_root,
		train_ratio=args.train_ratio,
		include_lidar_info=not args.without_lidar_info,
	)

	train_path = os.path.join(args.output_dir, 'town01_infos_train.pkl')
	val_path = os.path.join(args.output_dir, 'town01_infos_val.pkl')
	mmcv.dump(train_infos, train_path)
	mmcv.dump(val_infos, val_path)

	print(f'Generated train infos: {len(train_infos)} -> {train_path}')
	print(f'Generated val infos: {len(val_infos)} -> {val_path}')


if __name__ == '__main__':
	main()