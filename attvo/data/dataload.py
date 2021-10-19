import os
import torch
import numpy as np
import pickle
import os.path as osp
from torchvision import transforms
from tools.utils import process_poses, calc_vos_simple, load_image
from torch.utils import data
from functools import partial
#需要确认图片与真实值是否对应

class SevenScenes(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None, target_transform=None, mode=0, seed=7, real=False,
                 skip_images=False, vo_lib='orbslam'):
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        np.random.seed(seed)


        data_dir = osp.join('/home/data/tlx/dataset_10')
       
        if train:
            split_file = osp.join( 'train_split1.txt')
        else:
            split_file = osp.join( 'test_split1.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l) for l in f if not l.startswith('#')]



        self.c_imgs = []
        self.d_imgs = []

        ps = {}

        for seq in seqs:
            print(seq)
            seq_dir = osp.join(data_dir, '{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('txt') >= 0]

            frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
            pss = [np.loadtxt(osp.join(seq_dir, '{:06d}.txt'.
                                       format(i)), delimiter=',').flatten()[:54] for i in frame_idx]  
            ps[seq] = torch.from_numpy(np.asarray(pss)).float()
            c_imgs = [osp.join(seq_dir, '{:06d}.png'.format(i)) for i in frame_idx]


            self.c_imgs.extend(c_imgs)
            print(len( self.c_imgs))
                
        # convert pose to translation + log quaternion
        self.poses = np.empty((0, 54))
        for seq in seqs:
            self.poses = np.vstack((self.poses, ps[seq]))
            print(self.poses.shape)
            print(self.poses.shape[0])



    def __getitem__(self, index):
        img = None
        while img is None:
            img = load_image(self.c_imgs[index])
            tforms = [transforms.ToTensor()]
            data_transform = transforms.Compose(tforms)

            img1 = data_transform(img)
            img2 = np.array(img1)
            
            pose1 = self.poses[index]

            index += 1
        index -= 1

        return img2,pose1


    def __len__(self):
        return self.poses.shape[0]
