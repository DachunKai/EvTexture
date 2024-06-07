import numpy as np
import torch
from os import path as osp
from torch.utils import data as data

from basicsr.utils import get_root_logger, FileClient
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import mod_crop
from basicsr.utils.img_util import img2tensor

@DATASET_REGISTRY.register()
class VideoWithEventsTestDataset(data.Dataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'folder': []}
        self.scale = opt['scale']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.name = opt['name']
        if self.io_backend_opt['type'] == 'hdf5':
            self.io_backend_opt['h5_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['LR', 'HR']
        else:
            raise ValueError(f"We don't realize {self.io_backend_opt['type']} backend")

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoWithEventsTestDataset - {opt["name"]}')

        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split(' ')[0])
                    clips_num.append(line.split(' ')[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt, self.event_lqs = {}, {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip(clips, clips_num):
            self.io_backend_opt['h5_clip'] = clip
            self.file_client = FileClient(self.io_backend_opt['type'], **self.io_backend_opt)

            img_lqs, img_gts, event_lqs = self.file_client.get(list(range(int(num))))
            # mod_crop gt image for scale
            img_gts = [mod_crop(img, self.scale) for img in img_gts]
            self.imgs_lq[clip] = torch.stack(img2tensor(img_lqs), dim=0)
            self.imgs_gt[clip] = torch.stack(img2tensor(img_gts), dim=0)
            self.event_lqs[clip] = torch.from_numpy(np.stack(event_lqs, axis=0))
            self.folders.append(clip)
            self.lq_paths.append(osp.join('vid4', osp.splitext(clip)[0]))
            self.data_info['folder'].extend([clip] * int(num))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]

        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]
        event_lq = self.event_lqs[folder]

        voxel_f = event_lq[:len(event_lq) // 2]
        voxel_b = event_lq[len(event_lq) // 2:]
        return {
            'lq': img_lq,
            'gt': img_gt,
            'voxels_f': voxel_f,
            'voxels_b': voxel_b,
            'folder': folder,
            'lq_path': lq_path
        }

    def __len__(self):
        return len(self.folders)