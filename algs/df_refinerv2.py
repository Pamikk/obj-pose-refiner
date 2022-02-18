#naive icp implement to check accuracy and set as baseline
#will implement a cuda-version/fast version if needed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algs import BaseRefiner
relu = nn.LeakyReLU(0.01)
class PoseRefineNetFeat(nn.Module):
    def __init__(self):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 3,1,1)
        self.conv2 = torch.nn.Conv1d(64, 128, 3,1,1)

        self.conv3 = torch.nn.Conv1d(192, 256, 3,1,1)

        self.relu = relu

    def forward(self, x):
        x1 = self.relu(self.conv1(x))

        x2 = self.relu(self.conv2(x1))
        pointfeat = torch.cat([x1, x2], dim=1)

        x = self.relu(self.conv3(pointfeat))
        return x
class WeightingLayer(nn.Module):
    def __init__(self):
        super(WeightingLayer, self).__init__()
        self.fc1 =nn.Conv1d(256, 64,1,True)
        self.fc2 = nn.Conv1d(64, 16,1,True)
        self.fc3 = nn.Conv1d(16, 1,1,True)
    
    def forward(self, x, K = 64):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        
        _,topk_indices = torch.topk(x, K, dim = -1)
        topk_indices = topk_indices.flatten()
        return topk_indices
class DFRefiner(BaseRefiner):
    def __init__(self, cfg):
        super(DFRefiner, self).__init__()
        self.mode = cfg.decode
        self.cfg = cfg
        self.feat = PoseRefineNetFeat()
        self.weight_layer = WeightingLayer()
        self.relu = relu
        self.fuse = nn.Conv1d(512,256,1)
        self.conv_r = nn.Linear(256, 128)
        self.conv_t = nn.Linear(256, 128)

        self.mode = cfg.decode
        if self.mode=='quat':
            self.pred_r = nn.Linear(128, 4) #quaternion
        else:
            self.pred_r = nn.Linear(128, 6) # rotation paramterization
        self.pred_t = nn.Linear(128, 3)
        self.relu = relu

    def forward(self, src, tgt,pose):
        src_trans = BaseRefiner.project_pose(src,pose,scale=self.cfg.scale)         
        src_feats = self.feat(src_trans.transpose(1,2).contiguous())
        tgt_feats = self.feat(tgt.transpose(1,2).contiguous())
        B = src.shape[0]
        K_topk = self.cfg.topk
        batch_mask = torch.arange(B)
        batch_mask = batch_mask.unsqueeze(1).repeat(1, B)
        batch_mask = batch_mask.flatten()
        
        src_keypts_idx = self.weight_layer(src_feats,K_topk)
        src_keypts_idx[src_keypts_idx>src.shape[1]] = src.shape[1]-1
        src_keyfeats = src_feats[batch_mask,:, src_keypts_idx].view(B,K_topk,-1).contiguous()
        tgt_keypts_idx = self.weight_layer(tgt_feats,K_topk)

        tgt_keypts_idx[tgt_keypts_idx>tgt.shape[1]] = tgt.shape[1]-1
        tgt_keyfeats = tgt_feats[batch_mask, :,tgt_keypts_idx].view(B,K_topk,-1).contiguous()

        diff = torch.cat((src_keyfeats,tgt_keyfeats),dim=2)
        diff = relu(self.fuse(diff.transpose(1,2).contiguous()))
        ap_x = diff.mean(dim=-1)

        rx = self.conv_r(ap_x)
        tx = self.conv_t(ap_x)

        rx = self.pred_r(rx)
        tx = self.pred_t(tx)
        
        out_rx,out_tx = self.decode(rx,tx,pose)
        

        return out_rx, out_tx