import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torchvision import transforms, models


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.g = nn.Linear(in_channels, in_channels // 8)
        self.theta = nn.Linear(in_channels, in_channels // 8)
        self.phi = nn.Linear(in_channels, in_channels // 8)

        self.W = nn.Linear(in_channels // 8, in_channels)

    def forward(self, x):
        batch_size = x.size(0)
        out_channels = x.size(1)

        g_x = self.g(x).view(batch_size, out_channels // 8, 1)

        theta_x = self.theta(x).view(batch_size, out_channels // 8, 1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, out_channels // 8, 1)
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)
        z = W_y + x
        return z

class FourDirectionalLSTM(nn.Module):
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()
        self.feat_size = origin_feat_size // seq_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.lstm_rightleft = nn.LSTM(self.feat_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_downup = nn.LSTM(self.seq_size, self.hidden_size, batch_first=True, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_size).to(device),
                torch.randn(2, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        batch_size = x.size(0)
        x_rightleft = x.view(batch_size, self.seq_size, self.feat_size)
        x_downup = x_rightleft.transpose(1, 2)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        _, (hidden_state_lr, _) = self.lstm_rightleft(x_rightleft, hidden_rightleft)
        _, (hidden_state_ud, _) = self.lstm_downup(x_downup, hidden_downup)
        hlr_fw = hidden_state_lr[0, :, :]
        hlr_bw = hidden_state_lr[1, :, :]
        hud_fw = hidden_state_ud[0, :, :]
        hud_bw = hidden_state_ud[1, :, :]
        return torch.cat([ hlr_fw, hlr_bw, hud_fw, hud_bw], dim=1)

class deepvo(nn.Module):
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()
        self.feat_size = origin_feat_size // seq_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.lstm_rightleft = nn.LSTM(self.feat_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_downup = nn.LSTM(self.seq_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.fc123 = nn.Linear(49152, 2048)


    def init_hidden_(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_size).to(device),
                torch.randn(2, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        batch_size = x.size(0)
        x_rightleft = x.view(batch_size, self.seq_size, self.feat_size)
        x_downup = x_rightleft.transpose(1, 2)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        out1, (_, _) = self.lstm_rightleft(x_rightleft, hidden_rightleft)#torch.Size([8, 32, 512])
        out2, (_, _) = self.lstm_downup(x_downup, hidden_downup)#torch.Size([8, 64, 512])  #torch.Size([8, 96, 512])

        x = torch.cat([ out1, out2], dim=1)
        x = torch.flatten(x, 1)
    
        x = self.fc123(x)
        
        return x 

class PoseNet(nn.Module):

    def __init__(self):
        super(PoseNet, self).__init__()
        droprate = 0.5
        self.droprate = droprate


        feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor = feature_extractor
        self.feature_extractor.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3,  bias=False)

        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)

        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, 2048)


        self.fc_xyz = nn.Linear(2048, 27)
        self.fc_pqr = nn.Linear(2048, 27)


    def forward(self,target_image):

        c_img =  torch.chunk(target_image,10,dim=2)
        input = torch.cat((c_img[0],c_img[1],c_img[2],c_img[3],c_img[4],c_img[5],c_img[6],c_img[7],c_img[8],c_img[9]),1)

        z = self.feature_extractor(input)

        xyz = self.fc_xyz(z)
        pqr = self.fc_pqr(z)

        poses = torch.cat((xyz, pqr), 1)


        return poses

class PoseExpNet(nn.Module):
    def __init__(self):
        super(PoseExpNet, self).__init__()
        self.droprate = 0
     
        self.lstm = False
        droprate = 0
        pretrained = True
        feat_dim = 2048
      
        # replace the last FC layer in feature extractor
        self.feature_extractor  = models.resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        if self.lstm:
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=feat_dim, hidden_size=256)
            self.fc_xyz = nn.Linear(1024 , 27)#deepvo 2048,mapnet 1024
            self.fc_wpqr = nn.Linear(1024, 27)
        else:
            self.att = AttentionBlock(feat_dim)
            self.fc_xyz = nn.Linear(feat_dim, 27)
            self.fc_wpqr = nn.Linear(feat_dim, 27)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)

        if self.lstm:
            x = self.lstm4dir(x)
        else:
            x = self.att(x.view(x.size(0), -1))

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        #print(x.shape)
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)

