import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
import glob
import smplx
from utils.utils import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainLoader(data.Dataset):
    def __init__(self, clip_seconds=1, clip_fps=30, normalize=False, split='train'):
        self.clip_seconds = clip_seconds
        self.clip_len = 40            #clip_seconds * clip_fps  # T frames for each clip
        self.data_dict_list = []
        self.normalize = normalize
        self.split = split  # train/test


    def divide_clip(self, data_path):
        print("read .pkl file ...")
        data = np.load(data_path, allow_pickle=True)    #dict_keys(['J_locs', 'J_rotmat', 'J_len']), J_locs shape = (10158, 80, 31, 3), J_rotmat = (10158, 80, 31, 3, 3), J-len = (10158, 80, 31)

        num_sequenzes = data['J_locs'].shape[0]
        num_clips_per_sequenze = int(data['J_locs'].shape[0]/self.clip_len)     #should be 2

        print("reading data per person...")
        for s in tqdm(range(num_sequenzes)):
            for clipId in range(num_clips_per_sequenze):

                data_dict = {}
                start = clipId*self.clip_len
                stop = (clipId+1)*self.clip_len
                print(start, stop,s)
                print(data['J_locs'].shape)
                print(data['J_locs'][s,start:stop,0,...].shape)
                data_dict['r_locs'] = data['J_locs'][s,start:stop,0,...].reshape((self.clip_len,1,1,3))
                data_dict['J_rotmat'] = data['J_rotmat'][s,start:stop,...].reshape((self.clip_len,1,31,3,3))
                data_dict['J_shape'] = data['J_len'][s,start:stop,...].mean(axis=0).reshape((1,31))
                data_dict['J_locs_3d'] = data['J_locs'][s,start:stop,...].reshape((self.clip_len,1,31,3))
                
                self.data_dict_list.append(data_dict)

        print("...data all read")


    def read_data(self, data_path):
        self.divide_clip(data_path)
        self.n_samples = len(self.data_dict_list)
        print('[INFO] get {} sub clips in total.'.format(self.n_samples))


    def normalize_orientation(self):
        
        print("normalize orientation...")
        self.clip_img_list = []
        for i in tqdm(range(self.n_samples)):
            #rotate/translate to world origin
            transl = self.data_dict_list[i]['r_locs'][0,0,0,:]        #shape (3)
            rot = self.data_dict_list[i]['J_rotmat'][0,0,0,:,:].transpose()         #shape (3,3)

            clip_image = self.data_dict_list[i]['J_locs_3d'][:,0,:,:] - transl      #(40,31,3)
            clip_image = np.einsum('ij,...j->...i',rot,clip_image)
            clip_image = clip_image.reshape(clip_image.shape[0], -1)            #(40,93)

            self.clip_img_list.append(clip_image)

        self.clip_img_list = np.asarray(self.clip_img_list)     #(N,40,93)
        print("..orientation normalized, normalize values...")

        if self.normalize:
            prefix = 'preprocess_stats_for_our_prior'
            Xmean = self.clip_img_list.mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis, :]  # [1, 1, 93]
            Xstd = np.ones(self.clip_img_list.shape[-1]) * self.clip_img_list.std()  # [93]


            if self.split == 'train':
                np.savez_compressed('preprocess_stats/{}.npz'.format(prefix), Xmean=Xmean, Xstd=Xstd)
                self.clip_img_list = (self.clip_img_list - Xmean) / Xstd
            elif self.split == 'test':
                preprocess_stats = np.load('preprocess_stats/{}.npz'.format(prefix))
                self.clip_img_list = (self.clip_img_list - preprocess_stats['Xmean']) / preprocess_stats['Xstd']

        print("...normalization done")

        


        


    def create_body_repr(self, with_hand=False, smplx_model_path=None):
        print('[INFO] create motion clip imgs by {}...'.format(self.mode))

        smplx_model_male = smplx.create(smplx_model_path, model_type='smplx', gender='male', ext='npz',
                                        use_pca=False,  flat_hand_mean=True, # true: mean hand pose is a flat hand
                                        create_global_orient=True, create_body_pose=True, create_betas=True,
                                        create_left_hand_pose=True, create_right_hand_pose=True, create_expression=True,
                                        create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                        batch_size=self.clip_len).to(device)
        smplx_model_female = smplx.create(smplx_model_path, model_type='smplx', gender='female', ext='npz',
                                          use_pca=False, flat_hand_mean=True,
                                          create_global_orient=True, create_body_pose=True, create_betas=True,
                                          create_left_hand_pose=True, create_right_hand_pose=True, create_expression=True,
                                          create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                          batch_size=self.clip_len).to(device)

        self.clip_img_list = []
        for i in tqdm(range(self.n_samples)):
            ####################### set smplx params (gpu tensor) for each motion clip ##################
            body_param_ = {}
            body_param_['transl'] = self.data_dict_list[i]['trans']  # [T, 3]
            body_param_['global_orient'] = self.data_dict_list[i]['poses'][:, 0:3]
            body_param_['body_pose'] = self.data_dict_list[i]['poses'][:, 3:66]  # [T, 63]
            body_param_['left_hand_pose'] = self.data_dict_list[i]['poses'][:, 66:111]  # [T, 45]
            body_param_['right_hand_pose'] = self.data_dict_list[i]['poses'][:, 111:]  # [T, 45]
            body_param_['betas'] = np.tile(self.data_dict_list[i]['betas'][0:10], (len(body_param_['transl']), 1))  # [T, 10]

            for param_name in body_param_:
                body_param_[param_name] = torch.from_numpy(body_param_[param_name]).float().to(device)


            ################################ set body representations (marker/joint) ##########################3
            if self.data_dict_list[i]['gender'] == 'male':
                smplx_output = smplx_model_male(return_verts=True, **body_param_)  # generated human body mesh
            elif self.data_dict_list[i]['gender'] == 'female':
                smplx_output = smplx_model_female(return_verts=True, **body_param_)
            joints = smplx_output.joints  # [T, 127, 3]

            if self.mode in ['global_markers', 'local_markers']:
                if with_hand:
                    with open('loader/SSM2_withhand.json') as f:
                        marker_ids = list(json.load(f)['markersets'][0]['indices'].values())
                else:
                    with open('loader/SSM2.json') as f:
                        marker_ids = list(json.load(f)['markersets'][0]['indices'].values())
                markers = smplx_output.vertices[:, marker_ids, :]  # # [T(/bs), n_marker, 3]

            ############################### normalize first frame transl/gloabl_orient #############################
            # transfrom to pelvis at origin, normalize orientation of 1st frame (to face y axis)
            joints_frame0 = joints[0].detach()  # [N, 3] joints of first frame
            x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
            x_axis[-1] = 0
            x_axis = x_axis / torch.norm(x_axis)
            z_axis = torch.tensor([0, 0, 1]).float().to(device)
            y_axis = torch.cross(z_axis, x_axis)
            y_axis = y_axis / torch.norm(y_axis)
            transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
            joints = torch.matmul(joints - joints_frame0[0], transf_rotmat)  # [T(/bs), 25, 3]
            if self.mode in ['global_markers', 'local_markers']:
                markers_frame0 = markers[0].detach()
                markers = torch.matmul(markers - markers_frame0[0], transf_rotmat)  # [T(/bs), n_marker, 3]

            body_joints = joints[:, 0:25]  # [T, 25, 3]  root(1) + body(21) + jaw/leye/reye(3)
            hand_joints = joints[:, 25:55]  # [T, 30, 3]

            if self.mode in ['global_joints', 'local_joints']:
                if not with_hand:
                    cur_body = body_joints  # [T, 25, 3]
                else:
                    cur_body = torch.cat([body_joints, hand_joints], axis=1)  # [T, 55, 3]

                if self.mode == 'global_joints':
                    cur_body = cur_body.reshape(cur_body.shape[0], -1)  # [T, 75/165] with/without hand
                    cur_body = cur_body.detach().cpu().numpy()

                if self.mode == 'local_joints':
                    # body/hand joints: relative to pelvis
                    cur_body[:, 1:, :] = cur_body[:, 1:, :] - cur_body[:, 0:1, :]
                    cur_body = cur_body.reshape(cur_body.shape[0], -1)  # [T, 75/165] with/without hand
                    cur_body = cur_body.detach().cpu().numpy()

            if self.mode == 'global_markers':
                cur_body = markers
                cur_body = cur_body.reshape(cur_body.shape[0], -1)  # [T, n_marker*3]
                cur_body = cur_body.detach().cpu().numpy()

            if self.mode == 'local_markers':
                cur_body = torch.cat([joints[:, 0:1], markers], dim=1)  # [T, n_marker+1, 3] (add pelvis joint)
                # relative to pelvis joint
                cur_body[:, 1:, :] = cur_body[:, 1:, :] - cur_body[:, 0:1, :]
                cur_body = cur_body.reshape(cur_body.shape[0], -1)  # [T, (n_marker+1)*3]
                cur_body = cur_body.detach().cpu().numpy()

            self.clip_img_list.append(cur_body)

        self.clip_img_list = np.asarray(self.clip_img_list)  # [N, T-1, d]

        if self.normalize:
            prefix = 'preprocess_stats_smooth'
            if with_hand:
                prefix += '_withHand'
            Xmean = self.clip_img_list.mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis, :]  # [1, 1, d]
            Xstd = np.ones(self.clip_img_list.shape[-1]) * self.clip_img_list.std()  # [d]


            if self.mode in ['global_joints', 'global_markers']:
                if self.split == 'train':
                    np.savez_compressed('preprocess_stats/{}_{}.npz'.format(prefix, self.mode), Xmean=Xmean, Xstd=Xstd)
                    self.clip_img_list = (self.clip_img_list - Xmean) / Xstd
                elif self.split == 'test':
                    preprocess_stats = np.load('preprocess_stats/{}_{}.npz'.format(prefix, self.mode))
                    self.clip_img_list = (self.clip_img_list - preprocess_stats['Xmean']) / preprocess_stats['Xstd']

            if self.mode in ['local_joints', 'local_markers']:
                Xstd[0:3] = self.clip_img_list[:, :, 0:3].std()  # Xstd: [d]
                if self.split == 'train':
                    np.savez_compressed('preprocess_stats/{}_{}.npz'.format(prefix, self.mode), Xmean=Xmean, Xstd=Xstd)
                    self.clip_img_list[:, :, 0:3] = (self.clip_img_list[:, :, 0:3] - Xmean[:, :, 0:3]) / Xstd[0:3]
                elif self.split == 'test':
                    preprocess_stats = np.load('preprocess_stats/{}_{}.npz'.format(prefix, self.mode))
                    self.clip_img_list[:, :, 0:3] \
                        = (self.clip_img_list[:, :, 0:3] - preprocess_stats['Xmean'][:, :, 0:3]) / preprocess_stats['Xstd'][0:3]

        # print('max/min value in  motion clip: root joints', np.max(self.clip_img_list[:, :, 0:3]), np.min(self.clip_img_list[:, :, 0:3]))
        # print('max/min value in  motion clip: other joints', np.max(self.clip_img_list[:, :, 3:]), np.min(self.clip_img_list[:, :, 3:]))

        print('[INFO] motion clip imgs created.')


    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        clip_img = self.clip_img_list[index]  # [T, d] d dims of body representation
        clip_img = torch.from_numpy(clip_img).float().permute(1, 0).unsqueeze(0)  # [1, d, T]
        return [clip_img]
