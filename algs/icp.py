#naive icp implement to check accuracy and set as baseline
#will implement a cuda-version/fast version if needed
import numpy as np
import torch.nn as nn
import torch
from .cor_layers import PoseRegressNet

from algs import BaseRefiner

def nearest_neighbor(pt1,pt2):
    """batchwise find nearest neighbor in pt set 2for points set 1
       pt1:bxnx3,pt2:bxmx3
       return idx:bxnxk
    """
    distance = pt1.pow(2).sum(dim=-1).unsqueeze(-1)+pt2.pow(2).sum(dim=-1).unsqueeze(-2)
    distance -= 2*torch.bmm(pt1,pt2.transpose(1,2))
    distance = torch.sqrt(distance)
    dist,idx = torch.min(distance,dim=-1)
    return dist,idx
class ICP(BaseRefiner):
    def __init__(self,cfg):
        super(ICP,self).__init__()
        self.cfg = cfg

        if cfg.decode =='svd':
            self.pred = BaseRefiner.compute_pose
        else:
            self.pred = PoseRegressNet(cfg.decode)
        self.mode = cfg.decode
        self.terminate = self.cfg.threshold
    def forward(self,src,tgt,pose):
        B,num= src.shape[:2]
        assert B==1
        scale = self.cfg.scale
        batch_mask = torch.arange(B)
        batch_mask = batch_mask.unsqueeze(1).repeat(1, B)
        batch_mask = batch_mask.flatten()
        prev = None
        out_rx = pose[:,:3,:3]
        out_tx = pose[:,:3,3]
        for _ in range(self.cfg.iters):
            src_trans = BaseRefiner.project_pose(src,pose,scale=scale)
            dist, idx = nearest_neighbor(src_trans,tgt)
            cor = tgt[batch_mask,idx.flatten(),:].view(B,-1,3)
            err = torch.norm(src_trans-cor,p=2,dim=-1).mean()
            if prev:
                if abs(prev-err)<self.terminate:
                    break
                else:
                    out_rx = rx 
                    out_tx = tx 
            rx,tx = BaseRefiner.computer_pose(src_trans,cor)
            pose = torch.cat(rx,tx.view(-1,3,1),dim=-1)
            prev = err
        return src,cor,out_rx,out_tx/scale

        



        


