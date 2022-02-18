#naive icp implement to check accuracy and set as baseline
#will implement a cuda-version/fast version if needed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algs import BaseRefiner
relu = nn.LeakyReLU(0.01)
class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 3,1,1)
        self.conv2 = torch.nn.Conv1d(64, 128, 3,1,1)

        self.e_conv1 = torch.nn.Conv1d(3, 64, 3,1,1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 3,1,1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 3,1,1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
        self.relu = relu

    def forward(self, x, emb):
        x = self.relu(self.conv1(x))
        emb = self.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = self.relu(self.conv2(x))
        emb = self.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = self.relu(self.conv5(pointfeat_3))
        x = self.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x
class DFRefiner(BaseRefiner):
    def __init__(self, cfg):
        super(DFRefiner, self).__init__()
        num_points = cfg.channel
        self.mode = cfg.decode
        self.cfg = cfg
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        self.relu = relu
        self.conv_e = torch.nn.Conv1d(cfg.num_points_mesh,num_points,1)
        self.conv = torch.nn.Conv1d(cfg.num_points,num_points,1)
        self.conv1_r = nn.Linear(1024, 512)
        self.conv1_t = nn.Linear(1024, 512)

        self.conv2_r = nn.Linear(512, 128)
        self.conv2_t = nn.Linear(512, 128)
        self.mode = cfg.decode
        if self.mode=='quat':
            self.conv3_r = nn.Linear(128, 4) #quaternion
        else:
            self.conv3_r = nn.Linear(128, 6) # rotation paramterization
        self.conv3_t = nn.Linear(128, 3)
        self.relu = relu

    def forward(self, model,dmap,pose):
        model = BaseRefiner.project_pose(model,pose,scale=self.cfg.scale) 
        x = relu(self.conv(dmap))
        emb= relu(self.conv_e(model)) #re-order the model points

        x = x.transpose(2, 1).contiguous()
        emb = emb.transpose(2, 1).contiguous()     
        ap_x = self.feat(x, emb)

        rx = self.relu(self.conv1_r(ap_x))
        tx = self.relu(self.conv1_t(ap_x))   

        rx = self.relu(self.conv2_r(rx))
        tx = self.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx)
        tx = self.conv3_t(tx)
        
        out_rx,out_tx = self.decode(rx,tx,pose)
        

        return out_rx, out_tx