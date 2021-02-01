import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import os.path as osp
import matplotlib
import sys
import matplotlib.pyplot as plt
from tools.options import Options
from PoseExpNet import PoseExpNet
from torchvision import transforms, models
from tools.utils import quaternion_angular_error, qexp, load_state_dict
from data.dataload import SevenScenes
from torch.utils.data import DataLoader
from torch.autograd import Variable
from fujia import  *
from inverse_warp import *

#对10帧进行测试

parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='/home/data/tlx/odmetry/', type=str, help="Dataset directory")
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default='123', type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

device = torch.device("cuda") if not torch.cuda.is_available() else torch.device("cpu")


model = PoseExpNet()
model.eval()
model.to(device)
weights_filename = osp.expanduser('/home/data/tlx/our/model_14/epoch_160.pth.tar')

checkpoint = torch.load(weights_filename, map_location=device)
load_state_dict(model, checkpoint['model_state_dict'])
print('Loaded weights from {:s}'.format(weights_filename))

@torch.no_grad()
def main():
    args = parser.parse_args()
    from pose_evaluation_utils1 import test_framework_KITTI as test_framework

    seq_length =3
    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))

    for j, sample in enumerate(tqdm(framework)):

        imgs = sample['imgs']

        h,w,_ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            #imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]
            imgs = [np.array(Image.fromarray((np.uint8(img))).convert('RGB').resize(( args.img_width,args.img_height))).astype(np.float32)  for img in imgs]
        imgs = [np.transpose(img, (2,0,1)) for img in imgs]#tiaozhengweidubianc3 128 416

        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            #img = ((img/255 - 0.5)/0.5).to(device)
            img = (img/255).to(device)
            #print( len(imgs)//2)
            if i == 0:
                tgt_img = img
            elif i == 1:
                ref_imgs = img
            elif i==2:
                c_imgs0 = img
            elif i==3:
                c_imgs1 = img
            elif i==4:
                c_imgs2 = img
            elif i==5:
                c_imgs3 = img
            elif i==6:
                c_imgs4 = img
            elif i==7:
                c_imgs5 = img
            elif i==8:
                c_imgs6 = img
            elif i==9:
                c_imgs7 = img
        d_img = torch.cat((tgt_img,ref_imgs,c_imgs0, c_imgs1, c_imgs2, c_imgs3, c_imgs4, c_imgs5, c_imgs6, c_imgs7), 2)

        poses,_,_ = model(d_img)
        poses = poses[0].cpu()#6
        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)#(3, 3, 4)

        bijiaode_poses = sample['poses']
        real_pose1 = np.zeros([9,4,4])
        for i in  range(1,10):
            curr_pose = bijiaode_poses[i - 1]
            curr_pose = np.vstack((curr_pose,[0,0,0,1]))
            next_pose = bijiaode_poses[i]
            next_pose = np.concatenate((next_pose, [[0, 0, 0, 1]]), axis=0)
            real_pose = np.dot(np.linalg.inv(curr_pose),next_pose)

            real_pose1[i - 1] = real_pose

        ATE, RE = compute_pose_error(real_pose1, inv_transform_matrices)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE','RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))
    with open ('a.txt','a') as f:
        f.write("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
        f.write("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    c = gt[:, :, -1]
    scale_factor = np.sum(gt[:,:3,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:3,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:3,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)
    return ATE/snippet_length, RE/snippet_length


if __name__ == '__main__':
    main()