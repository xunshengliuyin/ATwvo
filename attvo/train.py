# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import sys
import time
import os.path as osp
import numpy as np
from tensorboardX import SummaryWriter
from tools.options import Options
from torchvision import transforms, models
#from tools.utils import AtLocCriterion, AtLocPlusCriterion, AverageMeter, Logger
from tools.utils import AverageMeter, Logger
from data.dataload import SevenScenes
#from data_danduilie.py import SevenScenes
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PoseExpNet import PoseExpNet
from torch import nn
import sys
from torch.autograd import Variable
from inverse_warp import *
from collections import OrderedDict


def load_state_dict1(model, state_dict):
    model_names = [n for n,_ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

  # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        model_prefix = model_names[0].split('.')[0]
        state_prefix = state_names[0].split('.')[0]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, model_prefix)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)





def mat2euler(mat, axes='sxyz'):

    try:
        firstaxis, parity, repetition, frame = [0,0,0,0]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = [0,0,0,0]
    _NEXT_AXIS = [1,2,0,1]
    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    batch = mat.shape[0]
    M = mat[:,:3,:3]
    cy = torch.sqrt(M[:,i, i]*M[:,i, i] + M[:,j, i]*M[:,j, i])
    _EPS4 = np.finfo(float).eps * 4.0
    tx = mat[:,0,3]
    ty = mat[:,1,3]
    tz = mat[:,2,3]

    #rz = torch.empty(1,1)
    rx = torch.ones(batch,dtype=torch.float64,device=tx.device)
    ry = torch.ones(batch,dtype=torch.float64,device=tx.device)
    rz = torch.ones(batch,dtype=torch.float64,device=tx.device)

    for x in range(batch):
        """
        if x == 0:
            if cy[x] > _EPS4:
                ax = torch.atan2( M[x,k, j],  M[x,k, k])
                ay = torch.atan2(-M[x,k, i],  cy[x])
                az = torch.atan2( M[x,j, i],  M[x,i, i])
            else:
                ax = torch.atan2(-M[x,j, k],  M[x,j, j])
                ay = torch.atan2(-M[x,k, i],  cy[x])
                az = torch.ones(1,1)
            rx = ax
            ry = ay
            rz = az
        else:
        """

        if cy[x] > _EPS4:
            #ax = torch.atan2( M[:,k, j],  M[:,k, k])
            #c1 = M[:,k, j]
            #c2 = M[:,k, k]
            #c3 = M[x:,k, j]
            #c4 = M[x:,k, k]

            ax = torch.atan2( M[x,k, j],  M[x,k, k])
            ay = torch.atan2(-M[x,k, i],  cy[x])
            az = torch.atan2( M[x,j, i],  M[x,i, i])
        else:
            ax = torch.atan2(-M[x,j, k],  M[x,j, j])
            ay = torch.atan2(-M[x,k, i],  cy[x])
            az = torch.ones(1,1)
        rx[x] = ax
        ry[x] = ay     
        rz[x] = az
        #ry = torch.cat([ry,ay],0)
        #rx = torch.cat((rx,ax),1)
        #ry = torch.cat((ry,ay),0)
        #rz = torch.cat((rz,az),0)
    pose = torch.stack([tx,ty,tz,rx,ry,rz]).transpose(0,1)
    return pose


def pose_vec2mat(vec, rotation_mode='euler'):
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    rot_mat = euler2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def euler2mat(angle):
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat





class AtLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion, self).__init__()#-3.0
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)



        
    def forward(self, pred ,tar,pr_glpose,weight2):
        batch_sizecx = pred.shape[0]
        targ = tar.view(-1,9,6)
        real_pose1 = torch.zeros(4, 4)
        c = torch.tensor([[0.0,0.0,0.0,1.0]],dtype=torch.float64, device=pred.device).repeat(batch_sizecx,1).unsqueeze(1)
        for i in  range(1,8+1):
            if i==1:
                curr_pose = targ[:,i - 1,:].squeeze() 
                curr_pose = pose_vec2mat(curr_pose)#.numpy().astype(np.float64)
                curr_pose = torch.cat((curr_pose,c),1)

                next_pose = targ[:,i,:].squeeze()               
                next_pose = pose_vec2mat(next_pose)#.numpy().astype(np.float64)
                next_pose = torch.cat((next_pose, c),1)

                real_pose =  torch.matmul(curr_pose,next_pose)#2-》1 X 3-》2 = 3-》1
                real_pose1 = real_pose
            else: 
                next_pose = targ[:,i,:].squeeze()
                next_pose = pose_vec2mat(next_pose)
                next_pose = torch.cat((next_pose, c),1)
            
                real_pose =  torch.matmul(real_pose1,next_pose)
                real_pose1 = real_pose


        Qpose_yuce = real_pose1
        Qpose_yuce1 = mat2euler(Qpose_yuce)

        listlo = torch.tensor([1.0, 1.0, 1.0,1.0, 1.0, 1.0],dtype=torch.float64, device=pred.device)
        f,d,e = torch.svd(weight2)

        loss=self.t_loss_fn(pred,targ)+0.1*self.t_loss_fn(Qpose_yuce1,pr_glpose)+0.01*self.q_loss_fn(d,listlo)
        return loss


# Config
opt = Options().parse()
cuda = torch.cuda.is_available()
device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"


logfile = osp.join(opt.runs_dir, 'log.txt')
stdout = Logger(logfile)
print('Logging to {:s}'.format(logfile))
sys.stdout = stdout

# Mode
atloc = PoseExpNet( )

model = atloc
train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)
val_criterion = AtLocCriterion()
param_list = [{'params': model.parameters()}]

# Optimizer
param_list = [{'params': model.parameters()}]
if hasattr(train_criterion, 'sax') and hasattr(train_criterion, 'saq') and hasattr(model, 'weight2'):#返回名字
    print('learn_beta')
    param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if opt.gamma is not None and hasattr(train_criterion, 'srx') and hasattr(train_criterion, 'srq'):
    print('learn_gamma')
    param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
optimizer = torch.optim.Adam(param_list, lr=opt.lr, weight_decay=opt.weight_decay)
#stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')opt.lr
stats_file = osp.join('stats.txt')
stats = np.loadtxt(stats_file)
tforms = [transforms.Resize(opt.cropsize)]
tforms.append(transforms.RandomCrop(opt.cropsize))
if opt.color_jitter > 0:
    assert opt.color_jitter <= 1.0
    print('Using ColorJitter data augmentation')
    tforms.append(transforms.ColorJitter(brightness=opt.color_jitter, contrast=opt.color_jitter, saturation=opt.color_jitter, hue=0.5))
else:
    print('Not Using ColorJitter')
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())


kwargs = dict(scene=opt.scene, data_path=opt.data_dir, transform=data_transform, target_transform=target_transform, seed=opt.seed)
train_set = SevenScenes(train=True, **kwargs)
val_set = SevenScenes(train=False, **kwargs)

kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True, **kwargs)
val_loader = DataLoader(val_set, batch_size=opt.batchsize, shuffle=False, **kwargs)

model.to(device)
train_criterion.to(device)
val_criterion.to(device)

total_steps = 0
total_steps1 = 0
writer = SummaryWriter(log_dir=opt.runs_dir)
experiment_name = opt.exp_name
start_epoch = 0
reuse = False
if reuse :
    weights_filename = osp.expanduser('/home/data/tlx/our/model/epoch_50.pth.tar')
    if osp.isfile(weights_filename):
        checkpoint = torch.load(weights_filename, map_location=device)
        load_state_dict1(model, checkpoint['model_state_dict'])
        start_epoch =checkpoint['epoch']
        print('Loaded weights from {:s}'.format(weights_filename))


for epoch in range(start_epoch+1,opt.epochs):
    if epoch % opt.val_freq == 0 or epoch == (opt.epochs - 1):
        val_batch_time = AverageMeter()
        val_loss = AverageMeter()
        model.eval()

        end = time.time()
        val_data_time = AverageMeter()

        for batch_idx, (val_tgt,pose) in enumerate(val_loader):
            val_data_time.update(time.time() - end)
            val_pose_var = Variable(pose, requires_grad=False)
            val_tgt_var = Variable(val_tgt, requires_grad=False)



            val_tgt_var = val_tgt_var.to(device)
            val_pose_var = val_pose_var.to(device)


            with torch.set_grad_enabled(False):
                val_output,Qpose ,weight2 = model(val_tgt_var )
                val_loss_tmp = val_criterion(val_output, val_pose_var,Qpose,weight2)
                val_loss_tmp = val_loss_tmp.item()

            val_loss.update(val_loss_tmp)
            val_batch_time.update(time.time() - end)

            writer.add_scalar('val_err', val_loss_tmp, total_steps1)
            total_steps1 += 1
            if batch_idx % opt.print_freq == 0:
                print('Val {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
                      .format(experiment_name, epoch, batch_idx, len(val_loader) - 1, val_data_time.val, val_data_time.avg, val_batch_time.val, val_batch_time.avg, val_loss_tmp))
            end = time.time()

        print('Val {:s}: Epoch {:d}, val_loss {:f}'.format(experiment_name, epoch, val_loss.avg))

        if epoch % opt.save_freq == 0:
            filename = osp.join(opt.models_dir, 'epoch_{:03d}.pth.tar'.format(epoch))
            checkpoint_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optim_state_dict': optimizer.state_dict(), 'criterion_state_dict': train_criterion.state_dict()}
            torch.save(checkpoint_dict, filename)
            print('Epoch {:d} checkpoint saved for {:s}'.format(epoch, experiment_name))

    model.train()
    train_data_time = AverageMeter()
    train_batch_time = AverageMeter()
    end = time.time()
    for batch_idx, (data_tgt, data_pose) in enumerate(train_loader):
        train_data_time.update(time.time() - end)

        data_tgt_val = Variable(data_tgt, requires_grad=True)
        data_pose_val = Variable(data_pose, requires_grad=False)

        data_tgt_val = data_tgt_val.to(device)
        data_pose_val = data_pose_val.to(device)


        with torch.set_grad_enabled(True):

            output ,Qpose1 ,weight2= model(data_tgt_val)
            loss_tmp = train_criterion(output,data_pose_val,Qpose1,weight2)


        loss_tmp.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar('Loss/train_step', loss_tmp.item(), total_steps)
        writer.add_scalar('lr',optimizer.param_groups[0]['lr'], total_steps)
        total_steps +=1
        train_batch_time.update(time.time() - end)
        if batch_idx % opt.print_freq == 0:
            print('Train {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
                  .format(experiment_name, epoch, batch_idx, len(train_loader) - 1, train_data_time.val, train_data_time.avg, train_batch_time.val, train_batch_time.avg, loss_tmp.item()))
        end = time.time()           
    
writer.add_graph(model, data_tgt)
writer.close()
