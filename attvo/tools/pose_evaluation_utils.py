# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
# import pandas as pd
from path import Path
import imageio
from tqdm import tqdm


class test_framework_KITTI(object):
    def __init__(self, root, sequence_set, seq_length=3, step=1):
        self.root = root
        self.img_files, self.poses, self.sample_indices = read_scene_data(self.root, sequence_set, seq_length, step)
        print(1111111)#(1591, 3,4)pose

    def generator(self):
        for sample_list,img_list, pose_list  in zip(self.sample_indices,self.img_files, self.poses ):
            for snippet_indices in sample_list:
                #imgs = [imread(img_list[i]).astype(np.float32) for i in snippet_indices]
                try:
                    #print(snippet_indices)
                    imgs = [imageio.imread(img_list[i]).astype(np.float64) for i in snippet_indices]

                except OSError as e:
                    print('except:', e)
                    print('except:', snippet_indices)

                poses = np.stack(pose_list[i] for i in snippet_indices)
                first_pose = poses[0]
                val_pose = poses
                d = poses[:,:,-1]
                poses[:,:,-1] -= first_pose[:,-1]#3 3 4
                c=np.linalg.inv(first_pose[:,:3])
                compensated_poses = np.linalg.inv(first_pose[:,:3]) @ poses#矩阵-向量乘法*(kitti的pos是原图像是后面的dao0)
                #(370, 1226, 3)3
                yield {'imgs': imgs,
                       'path': img_list[0],
                       #'poses': compensated_poses#最终结果是1-》1，2—》1，3-》1
                       'poses':val_pose
                       }

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return len(self.sample_indices[0])#sum(len(imgs) for imgs in self.img_files)


def read_scene_data(data_root, sequence_set, seq_length=3, step=1):
    data_root = Path(data_root)
    im_sequences = []
    poses_sequences = []
    indices_sequences = []
    demi_length = (seq_length - 1) // 2#1
    shift_range = np.array([step*i for i in range(-demi_length, demi_length + 8)]).reshape(1, -1)#-1  0  1
    #[[-1  0  1]]
    sequences = set()
    for seq in sequence_set:
        corresponding_dirs = set((data_root/'sequences').dirs(seq))
        sequences = sequences | corresponding_dirs

    print('getting test metadata for theses sequences : {}'.format(sequences))
    for sequence in tqdm(sequences):
        poses = np.genfromtxt(data_root/'poses'/'{}.txt'.format(sequence.name)).astype(np.float64).reshape(-1, 3, 4)
        imgs = sorted((sequence/'image_2').files('*.png'))
        # construct 5-snippet sequences
        #tgt_indices = np.arange(demi_length, len(imgs) - 8).reshape(-1, 1)
        tgt_indices = np.arange(demi_length, len(imgs) - 8,9).reshape(-1, 1)
        snippet_indices = shift_range + tgt_indices
        im_sequences.append(imgs)
        poses_sequences.append(poses)#(1591, 3, 4)
        indices_sequences.append(snippet_indices)
    return im_sequences, poses_sequences, indices_sequences