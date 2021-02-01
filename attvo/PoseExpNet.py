import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_
from torchvision import transforms, models
from torch.nn import functional as F
from torch.autograd import Variable


class PoseExpNet(nn.Module):

    def __init__(self):
        super(PoseExpNet, self).__init__()
        droprate = 0.5
        in_channels = 4096
        self.droprate = droprate
        cha = 1024

        feat_dim = 2048
        #---------------------------------------------------
        feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor = feature_extractor
        self.feature_extractor.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,  bias=False)

        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, 1024)


        # ----------------------------------------------
        self.g = nn.Linear(in_channels, in_channels // 8)
        self.theta = nn.Linear(in_channels, in_channels // 8)
        self.phi = nn.Linear(in_channels, in_channels // 8)

        self.W = nn.Linear(in_channels // 8, in_channels)
        # ----------------------------------------------
        self.fc_a = nn.Linear(in_channels, in_channels // 8)
        self.fc_b = nn.Linear(in_channels, in_channels // 8)

        self.fc_xyz = nn.Linear(in_channels // 8, 27)
        self.fc_pqr = nn.Linear(in_channels // 8, 27)


        self.Qfc_a = nn.Linear(cha ,cha// 8)
        self.Qfc_b = nn.Linear(cha ,cha // 8)

        self.Qfc_xyz = nn.Linear(cha // 8, 3)
        self.Qfc_pqr = nn.Linear(cha // 8, 3)
        # ----------------------------------------------
        self.hidden_size = 1024
        self.seq_size = 128
        self.feat_size = 72
        self.lstm_rightleft = nn.LSTM(72, self.hidden_size, 2, batch_first=True, bidirectional=False)
        self.lstm_downup = nn.LSTM(128,  self.hidden_size, 2, batch_first=True, bidirectional=False)


        weight2 = torch.randn(6, 6)
        self.weight2 = nn.Parameter(torch.Tensor(weight2), requires_grad=True)

        init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)


    def forward(self,target_image):
        target_image =  torch.zeros(1, 3, 1280, 416)
        batch_size = target_image.size(0)
        c_img = torch.chunk(target_image, 10, dim=2)

        input = torch.cat((c_img[0], c_img[9]), 1)


        z1 = self.feature_extractor(torch.cat((c_img[0], c_img[1]), 1))
        z2 = self.feature_extractor(torch.cat((c_img[1], c_img[2]), 1))
        z3 = self.feature_extractor(torch.cat((c_img[2], c_img[3]), 1))
        z4 = self.feature_extractor(torch.cat((c_img[3], c_img[4]), 1))
        z5 = self.feature_extractor(torch.cat((c_img[4], c_img[5]), 1))
        z6 = self.feature_extractor(torch.cat((c_img[5], c_img[6]), 1))
        z7 = self.feature_extractor(torch.cat((c_img[6], c_img[7]), 1))
        z8 = self.feature_extractor(torch.cat((c_img[7], c_img[8]), 1))
        z9 = self.feature_extractor(torch.cat((c_img[8], c_img[9]), 1))

        Qz = self.feature_extractor(input)
        Qxyz = self.Qfc_a(Qz)
        Qpqr = self.Qfc_b(Qz)
        Qpqr = 0.01*self.Qfc_pqr(Qpqr)
        Qxyz = 0.01*self.Qfc_xyz(Qxyz)

        Qpose = torch.cat((Qpqr, Qxyz), 1)

        feature = torch.cat((z1.view(batch_size, 1,-1 ), z2.view(batch_size, 1,-1 ), z3.view(batch_size, 1,-1 ), z4.view(batch_size, 1,-1 ), z5.view(batch_size, 1,-1 ),
                          z6.view(batch_size, 1,-1 ), z7.view(batch_size, 1,-1 ), z8.view(batch_size, 1,-1 ), z9.view(batch_size, 1,-1 )), 1)


        out = feature.view(1, 128, 72)


        batch_size = out.size(0)
        h0 = torch.randn(2, batch_size, self.hidden_size).to(out.device)  # torch.Size([2, b, 512])
        c0 = torch.randn(2, batch_size, self.hidden_size).to(out.device)  # torch.Size([2, b, 512])
        h1 = torch.randn(2, batch_size, self.hidden_size).to(out.device)  # torch.Size([2, batch, 512])
        c1 = torch.randn(2, batch_size, self.hidden_size).to(out.device)  # torch.Size([2, b, 512])

        batch_size1 = out.size(0)
        x_rightleft = out.view(batch_size1, self.seq_size, self.feat_size)

        x_downup = x_rightleft.transpose(1, 2)

        o1, (hidden_state_lr, cn1) = self.lstm_rightleft(x_rightleft, (h0, c0))

        o2, (hidden_state_ud, cn1) = self.lstm_downup(x_downup, (h1, c1))
        torch.Size([2, 64, 256])
        hlr_fw = hidden_state_lr[0, :, :]
        hlr_bw = hidden_state_lr[1, :, :]
        hud_fw = hidden_state_ud[0, :, :]
        hud_bw = hidden_state_ud[1, :, :]
        z = torch.cat([hlr_fw, hlr_bw, hud_fw, hud_bw], dim=1)  # torch.Size([b, 4096])

        z = torch.flatten(z, 1)
        out_channels = z.size(1)  # 2048

        g_x = self.g(z).view(batch_size, out_channels // 8, 1)
        # torch.Size([64, 256, 1])
        theta_x = self.theta(z).view(batch_size, out_channels // 8, 1)
        # 64.256.1
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(z).view(batch_size, out_channels // 8, 1)
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)

        pose = W_y + z
        pose = pose.view(batch_size,-1)

        xyz1 = self.fc_a(pose)
        pqr1 = self.fc_b(pose)
        pqr1 = 0.01 * self.fc_pqr(pqr1).view(batch_size,9,-1)
        xyz1 = 0.01 * self.fc_xyz(xyz1).view(batch_size,9,-1)

        poses = torch.cat((xyz1, pqr1), 2)

        poses = torch.matmul(poses, self.weight2)
        Qpose = torch.matmul(Qpose,self.weight2)

        return poses,Qpose, self.weight2
