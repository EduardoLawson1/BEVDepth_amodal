import os

import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

__all__ = ['CarlaDetDataset', 'collate_fn']


map_name_from_general_to_detection = {
	'human.pedestrian.adult': 'pedestrian',
	'human.pedestrian.child': 'pedestrian',
	'vehicle.car': 'car',
	'vehicle.motorcycle': 'motorcycle',
	'vehicle.bicycle': 'bicycle',
	'vehicle.bus.bendy': 'bus',
	'vehicle.bus.rigid': 'bus',
	'vehicle.truck': 'truck',
	'vehicle.construction': 'construction_vehicle',
	'vehicle.trailer': 'trailer',
	'movable_object.barrier': 'barrier',
	'movable_object.trafficcone': 'traffic_cone',
}


def get_rot(h):
	return torch.tensor([
		[np.cos(h), np.sin(h)],
		[-np.sin(h), np.cos(h)],
	],
						dtype=torch.float32)


def img_transform(img, resize, resize_dims, crop, flip, rotate):
	ida_rot = torch.eye(2)
	ida_tran = torch.zeros(2)
	img = img.resize(resize_dims)
	img = img.crop(crop)
	if flip:
		img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
	img = img.rotate(rotate)

	ida_rot *= resize
	ida_tran -= torch.tensor(crop[:2], dtype=torch.float32)
	if flip:
		flip_mat = torch.tensor([[-1, 0], [0, 1]], dtype=torch.float32)
		flip_bias = torch.tensor([crop[2] - crop[0], 0], dtype=torch.float32)
		ida_rot = flip_mat.matmul(ida_rot)
		ida_tran = flip_mat.matmul(ida_tran) + flip_bias
	rot_mat = get_rot(rotate / 180 * np.pi)
	rot_bias = torch.tensor([crop[2] - crop[0], crop[3] - crop[1]],
							dtype=torch.float32) / 2
	rot_bias = rot_mat.matmul(-rot_bias) + rot_bias
	ida_rot = rot_mat.matmul(ida_rot)
	ida_tran = rot_mat.matmul(ida_tran) + rot_bias
	ida_mat = ida_rot.new_zeros(4, 4)
	ida_mat[3, 3] = 1
	ida_mat[2, 2] = 1
	ida_mat[:2, :2] = ida_rot
	ida_mat[:2, 3] = ida_tran
	return img, ida_mat


def bev_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
	rotate_angle = torch.tensor(rotate_angle / 180 * np.pi, dtype=torch.float32)
	rot_sin = torch.sin(rotate_angle)
	rot_cos = torch.cos(rotate_angle)
	rot_mat = torch.tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
							[0, 0, 1]],
						   dtype=torch.float32)
	scale_mat = torch.tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
							  [0, 0, scale_ratio]],
							 dtype=torch.float32)
	flip_mat = torch.eye(3, dtype=torch.float32)
	if flip_dx:
		flip_mat = flip_mat @ torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
										   dtype=torch.float32)
	if flip_dy:
		flip_mat = flip_mat @ torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]],
										   dtype=torch.float32)
	rot_mat = flip_mat @ (scale_mat @ rot_mat)

	if gt_boxes.shape[0] > 0:
		gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
		gt_boxes[:, 3:6] *= scale_ratio
		gt_boxes[:, 6] += rotate_angle
		if flip_dx:
			gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
		if flip_dy:
			gt_boxes[:, 6] = -gt_boxes[:, 6]
		gt_boxes[:, 7:] = (
			rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
	return gt_boxes, rot_mat


def _rotation_matrix_from_quaternion(rotation):
	if rotation is None:
		return np.eye(3, dtype=np.float32)
	return Quaternion(rotation).rotation_matrix.astype(np.float32)


def _safe_translation(translation):
	if translation is None:
		return np.zeros(3, dtype=np.float32)
	return np.array(translation, dtype=np.float32)


def _rt_to_mat(calibrated_or_pose):
	mat = np.eye(4, dtype=np.float32)
	if calibrated_or_pose is None:
		return torch.from_numpy(mat)
	mat[:3, :3] = _rotation_matrix_from_quaternion(
		calibrated_or_pose.get('rotation'))
	mat[:3, 3] = _safe_translation(calibrated_or_pose.get('translation'))
	return torch.from_numpy(mat)


def _intrin_to_4x4(camera_intrinsic):
	intrin = np.eye(4, dtype=np.float32)
	if camera_intrinsic is not None and len(camera_intrinsic) > 0:
		intrin[:3, :3] = np.array(camera_intrinsic, dtype=np.float32)
	return torch.from_numpy(intrin)


def _normalize_category_name(category_name):
	if category_name in map_name_from_general_to_detection:
		return map_name_from_general_to_detection[category_name]
	if '.' in category_name:
		return category_name.split('.')[-1]
	return category_name


class CarlaDetDataset(Dataset):

	def __init__(self,
				 ida_aug_conf,
				 bda_aug_conf,
				 classes,
				 data_root,
				 info_paths,
				 is_train,
				 use_cbgs=False,
				 num_sweeps=1,
				 img_conf=dict(img_mean=[123.675, 116.28, 103.53],
							   img_std=[58.395, 57.12, 57.375],
							   to_rgb=True),
				 return_depth=False,
				 sweep_idxes=list(),
				 key_idxes=list(),
				 use_fusion=False):
		super().__init__()
		if isinstance(info_paths, list):
			self.infos = list()
			for info_path in info_paths:
				self.infos.extend(mmcv.load(info_path))
		else:
			self.infos = mmcv.load(info_paths)

		self.is_train = is_train
		self.ida_aug_conf = ida_aug_conf
		self.bda_aug_conf = bda_aug_conf
		self.data_root = data_root
		self.classes = classes
		self.use_cbgs = use_cbgs
		if self.use_cbgs:
			self.cat2id = {name: i for i, name in enumerate(self.classes)}
			self.sample_indices = self._get_sample_indices()
		self.num_sweeps = num_sweeps
		self.img_mean = np.array(img_conf['img_mean'], np.float32)
		self.img_std = np.array(img_conf['img_std'], np.float32)
		self.to_rgb = img_conf['to_rgb']
		self.return_depth = return_depth
		self.sweeps_idx = sweep_idxes
		self.key_idxes = [0] + key_idxes
		self.use_fusion = use_fusion

	def _get_sample_indices(self):
		class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
		for idx, info in enumerate(self.infos):
			ann_infos = info.get('ann_infos', [])
			gt_names = set(
				[_normalize_category_name(ann['category_name']) for ann in ann_infos
				 if 'category_name' in ann])
			for gt_name in gt_names:
				if gt_name not in self.classes:
					continue
				class_sample_idxs[self.cat2id[gt_name]].append(idx)

		duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
		if duplicated_samples == 0:
			return list(range(len(self.infos)))

		class_distribution = {
			k: max(len(v) / duplicated_samples, 1e-6)
			for k, v in class_sample_idxs.items()
		}
		sample_indices = list()
		frac = 1.0 / max(len(self.classes), 1)
		ratios = [frac / v for v in class_distribution.values()]
		for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
			if len(cls_inds) == 0:
				continue
			sample_indices += np.random.choice(
				cls_inds, int(len(cls_inds) * ratio)).tolist()
		return sample_indices if len(sample_indices) > 0 else list(
			range(len(self.infos)))

	def sample_ida_augmentation(self):
		H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
		fH, fW = self.ida_aug_conf['final_dim']
		if self.is_train:
			resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
			resize_dims = (int(W * resize), int(H * resize))
			newW, newH = resize_dims
			crop_h = int(
				(1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) *
				newH) - fH
			crop_w = int(np.random.uniform(0, max(0, newW - fW)))
			crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
			flip = False
			if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
				flip = True
			rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
		else:
			resize = max(fH / H, fW / W)
			resize_dims = (int(W * resize), int(H * resize))
			newW, newH = resize_dims
			crop_h = int(
				(1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
			crop_w = int(max(0, newW - fW) / 2)
			crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
			flip = False
			rotate_ida = 0
		return resize, resize_dims, crop, flip, rotate_ida

	def sample_bda_augmentation(self):
		if self.is_train:
			rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
			scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
			flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
			flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
		else:
			rotate_bda = 0
			scale_bda = 1.0
			flip_dx = False
			flip_dy = False
		return rotate_bda, scale_bda, flip_dx, flip_dy

	def choose_cams(self):
		if self.is_train and self.ida_aug_conf['Ncams'] < len(
				self.ida_aug_conf['cams']):
			cams = np.random.choice(self.ida_aug_conf['cams'],
									self.ida_aug_conf['Ncams'],
									replace=False)
			return list(cams)
		return list(self.ida_aug_conf['cams'])

	def _get_cam_info(self, frame_cam_infos, cam):
		if cam in frame_cam_infos:
			return frame_cam_infos[cam]
		# Fallback for lowercase camera naming.
		lower_key_map = {k.lower(): k for k in frame_cam_infos.keys()}
		cam_lower = cam.lower()
		if cam_lower in lower_key_map:
			return frame_cam_infos[lower_key_map[cam_lower]]
		raise KeyError(f'Camera {cam} not found in frame cam infos.')

	def get_image(self, cam_infos, cams):
		assert len(cam_infos) > 0
		sweep_imgs = list()
		sweep_sensor2ego_mats = list()
		sweep_intrin_mats = list()
		sweep_ida_mats = list()
		sweep_sensor2sensor_mats = list()
		sweep_timestamps = list()
		key_info = cam_infos[0]

		for cam in cams:
			imgs = list()
			sensor2ego_mats = list()
			intrin_mats = list()
			ida_mats = list()
			sensor2sensor_mats = list()
			timestamps = list()
			resize, resize_dims, crop, flip, rotate_ida = (
				self.sample_ida_augmentation())

			key_cam_info = self._get_cam_info(key_info, cam)
			key_sensor2ego = _rt_to_mat(key_cam_info.get('calibrated_sensor'))
			key_ego2global = _rt_to_mat(key_cam_info.get('ego_pose'))
			global2keyego = torch.inverse(key_ego2global)
			keyego2keysensor = torch.inverse(key_sensor2ego)

			for frame_cam_infos in cam_infos:
				cam_info = self._get_cam_info(frame_cam_infos, cam)
				img_path = os.path.join(self.data_root, cam_info['filename'])
				img = Image.open(img_path).convert('RGB')

				sweepsensor2sweepego = _rt_to_mat(
					cam_info.get('calibrated_sensor'))
				sweepego2global = _rt_to_mat(cam_info.get('ego_pose'))

				keysensor2sweepsensor = (
					keyego2keysensor @ global2keyego @ sweepego2global
					@ sweepsensor2sweepego).inverse()
				sweepsensor2keyego = global2keyego @ sweepego2global @ \
					sweepsensor2sweepego

				sensor2ego_mats.append(sweepsensor2keyego)
				sensor2sensor_mats.append(keysensor2sweepsensor)
				intrin_mats.append(
					_intrin_to_4x4(cam_info['calibrated_sensor'].get(
						'camera_intrinsic')))

				img, ida_mat = img_transform(img,
											 resize=resize,
											 resize_dims=resize_dims,
											 crop=crop,
											 flip=flip,
											 rotate=rotate_ida)
				ida_mats.append(ida_mat)
				img = mmcv.imnormalize(np.array(img), self.img_mean,
									   self.img_std, self.to_rgb)
				imgs.append(torch.from_numpy(img).permute(2, 0, 1))
				timestamps.append(cam_info.get('timestamp', 0))

			sweep_imgs.append(torch.stack(imgs))
			sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
			sweep_intrin_mats.append(torch.stack(intrin_mats))
			sweep_ida_mats.append(torch.stack(ida_mats))
			sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
			sweep_timestamps.append(torch.tensor(timestamps, dtype=torch.float32))

		ego2global_rotations = []
		ego2global_translations = []
		for cam in cams:
			cam_pose = self._get_cam_info(key_info, cam).get('ego_pose', {})
			if 'rotation' in cam_pose:
				ego2global_rotations.append(cam_pose['rotation'])
			if 'translation' in cam_pose:
				ego2global_translations.append(cam_pose['translation'])
		if len(ego2global_rotations) == 0:
			ego2global_rotation = [1.0, 0.0, 0.0, 0.0]
		else:
			ego2global_rotation = np.mean(ego2global_rotations, 0)
		if len(ego2global_translations) == 0:
			ego2global_translation = [0.0, 0.0, 0.0]
		else:
			ego2global_translation = np.mean(ego2global_translations, 0)

		img_metas = dict(
			box_type_3d=LiDARInstance3DBoxes,
			ego2global_translation=ego2global_translation,
			ego2global_rotation=ego2global_rotation,
		)

		ret_list = [
			torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
			torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
			torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
			torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
			torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
			torch.stack(sweep_timestamps).permute(1, 0),
			img_metas,
		]
		if self.return_depth:
			num_sweeps = ret_list[0].shape[0]
			num_cams = ret_list[0].shape[1]
			h, w = self.ida_aug_conf['final_dim']
			ret_list.append(ret_list[0].new_zeros((num_sweeps, num_cams, h, w)))
		return ret_list

	def get_gt(self, info, cams):
		if 'ann_infos' not in info or len(info['ann_infos']) == 0:
			return torch.zeros((0, 9), dtype=torch.float32), torch.zeros(
				(0, ), dtype=torch.long)

		ego2global_rotations = []
		ego2global_translations = []
		for cam in cams:
			cam_pose = self._get_cam_info(info['cam_infos'], cam).get(
				'ego_pose', {})
			if 'rotation' in cam_pose:
				ego2global_rotations.append(cam_pose['rotation'])
			if 'translation' in cam_pose:
				ego2global_translations.append(cam_pose['translation'])

		if len(ego2global_rotations) == 0:
			ego_q = Quaternion([1.0, 0.0, 0.0, 0.0])
		else:
			ego_q = Quaternion(np.mean(ego2global_rotations, 0))
		if len(ego2global_translations) == 0:
			ego_t = np.zeros(3, dtype=np.float32)
		else:
			ego_t = np.mean(ego2global_translations, 0).astype(np.float32)

		rot_inv = ego_q.inverse.rotation_matrix.astype(np.float32)
		trans_inv = -ego_t

		gt_boxes = list()
		gt_labels = list()
		for ann_info in info['ann_infos']:
			category_name = _normalize_category_name(
				ann_info.get('category_name', ''))
			if category_name not in self.classes:
				continue

			center_global = np.array(ann_info.get('translation', [0, 0, 0]),
									 dtype=np.float32)
			center_ego = rot_inv @ (center_global + trans_inv)

			size = np.array(ann_info.get('size', [0, 0, 0]), dtype=np.float32)
			if size.shape[0] != 3:
				size = np.zeros(3, dtype=np.float32)
			# Keep the same convention used by the nuScenes dataset path.
			box_dxdydz = size[[1, 0, 2]]

			rotation = ann_info.get('rotation', [1.0, 0.0, 0.0, 0.0])
			box_q_global = Quaternion(rotation)
			box_q_ego = ego_q.inverse * box_q_global
			box_yaw = np.array([box_q_ego.yaw_pitch_roll[0]], dtype=np.float32)

			velocity = ann_info.get('velocity', [0.0, 0.0, 0.0])
			velocity = np.array(velocity, dtype=np.float32)
			if velocity.shape[0] < 2:
				velocity = np.array([0.0, 0.0], dtype=np.float32)
			else:
				velocity = velocity[:2]

			gt_boxes.append(
				np.concatenate([center_ego, box_dxdydz, box_yaw, velocity]))
			gt_labels.append(self.classes.index(category_name))

		if len(gt_boxes) == 0:
			return torch.zeros((0, 9), dtype=torch.float32), torch.zeros(
				(0, ), dtype=torch.long)
		return torch.tensor(np.array(gt_boxes),
							dtype=torch.float32), torch.tensor(gt_labels,
															   dtype=torch.long)

	def __getitem__(self, idx):
		if self.use_cbgs:
			idx = self.sample_indices[idx]

		cams = self.choose_cams()
		cam_infos = list()
		info = self.infos[idx]

		for key_idx in self.key_idxes:
			cur_idx = key_idx + idx
			if cur_idx < 0 or cur_idx >= len(self.infos):
				cur_idx = idx
			elif self.infos[cur_idx].get('scene_token') != self.infos[idx].get(
					'scene_token'):
				cur_idx = idx
			cam_infos.append(self.infos[cur_idx]['cam_infos'])

		image_data_list = self.get_image(cam_infos, cams)
		(sweep_imgs, sweep_sensor2ego_mats, sweep_intrins, sweep_ida_mats,
		 sweep_sensor2sensor_mats, sweep_timestamps,
		 img_metas) = image_data_list[:7]

		img_metas['token'] = info.get('sample_token', str(idx))
		if self.is_train:
			gt_boxes, gt_labels = self.get_gt(info, cams)
		else:
			gt_boxes = sweep_imgs.new_zeros(0, 7)
			gt_labels = sweep_imgs.new_zeros(0, )

		rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
		)
		bda_mat = sweep_imgs.new_zeros(4, 4)
		bda_mat[3, 3] = 1
		gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda,
										  flip_dx, flip_dy)
		bda_mat[:3, :3] = bda_rot

		ret_list = [
			sweep_imgs,
			sweep_sensor2ego_mats,
			sweep_intrins,
			sweep_ida_mats,
			sweep_sensor2sensor_mats,
			bda_mat,
			sweep_timestamps,
			img_metas,
			gt_boxes,
			gt_labels,
		]
		if self.return_depth:
			ret_list.append(image_data_list[7])
		return ret_list

	def __str__(self):
		return f"""CarlaData: {len(self)} samples. Split: \
			{"train" if self.is_train else "val"}.
					Augmentation Conf: {self.ida_aug_conf}"""

	def __len__(self):
		if self.use_cbgs:
			return len(self.sample_indices)
		return len(self.infos)


def collate_fn(data, is_return_depth=False):
	imgs_batch = list()
	sensor2ego_mats_batch = list()
	intrin_mats_batch = list()
	ida_mats_batch = list()
	sensor2sensor_mats_batch = list()
	bda_mat_batch = list()
	timestamps_batch = list()
	gt_boxes_batch = list()
	gt_labels_batch = list()
	img_metas_batch = list()
	depth_labels_batch = list()
	for iter_data in data:
		(
			sweep_imgs,
			sweep_sensor2ego_mats,
			sweep_intrins,
			sweep_ida_mats,
			sweep_sensor2sensor_mats,
			bda_mat,
			sweep_timestamps,
			img_metas,
			gt_boxes,
			gt_labels,
		) = iter_data[:10]
		if is_return_depth:
			gt_depth = iter_data[10]
			depth_labels_batch.append(gt_depth)
		imgs_batch.append(sweep_imgs)
		sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
		intrin_mats_batch.append(sweep_intrins)
		ida_mats_batch.append(sweep_ida_mats)
		sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
		bda_mat_batch.append(bda_mat)
		timestamps_batch.append(sweep_timestamps)
		img_metas_batch.append(img_metas)
		gt_boxes_batch.append(gt_boxes)
		gt_labels_batch.append(gt_labels)
	mats_dict = dict()
	mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
	mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
	mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
	mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
	mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
	ret_list = [
		torch.stack(imgs_batch),
		mats_dict,
		torch.stack(timestamps_batch),
		img_metas_batch,
		gt_boxes_batch,
		gt_labels_batch,
	]
	if is_return_depth:
		ret_list.append(torch.stack(depth_labels_batch))
	return ret_list
