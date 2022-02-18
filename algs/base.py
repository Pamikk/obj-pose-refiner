#define base refiner
from torch import nn
from abc import abstractmethod
from typing import TypeVar
import torch
Tensor = TypeVar('torch.tensor')
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias!=None:
            m.bias.data.fill_(0.0)
    elif type(m) == nn.Conv1d:
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias!=None:
            m.bias.data.fill_(0.0)
class BaseRefiner(nn.Module):
    def __init__(self)->None:
        super(BaseRefiner,self).__init__()
    @staticmethod
    def ortho2rot(e1,e2)->Tensor:
        #perform rotation parameterizaion
        #return recover 3x3 rotation matrix
        x = e1/torch.norm(e1,p=2,dim=-1,keepdim=True)
        z = torch.cross(x,e2,dim=-1)
        z = z/torch.norm(z,p=2,dim=-1, keepdim=True)
        y = torch.cross(z,x,dim=-1)
        rot = torch.stack((x,y,z),dim=-1)
        x = x.to(dtype=torch.float64)
        return rot
    @staticmethod
    def quat2rot(vec)->Tensor:
        assert vec.shape[-1] == 4
        vec = vec / torch.norm(vec, p=2, dim=-1, keepdim=True)
        q0 = vec[:,0]
        q1 = vec[:,1]
        q2 = vec[:,2]
        q3 = vec[:,3]
        x = torch.stack((2.0*(q0.pow(2)+q1.pow(2))-1,2.0*(q1*q2+q0*q3),2.0*(q1*q3-q0*q2)),dim=-1)
        y = torch.stack((2.0*(q1*q2-q0*q3),2.0*(q0.pow(2)+q2.pow(2))-1,2.0*(q2*q3+q0*q1)),dim=-1)
        z = torch.stack((2.0*(q1*q3+q0*q2),2.0*(q2*q3-q0*q1),2.0*(q0.pow(2)+q3.pow(2))-1),dim=-1)
        rot = torch.stack((x,y,z),dim=-1)
        return rot
    def decode(self,rx,tx,pose)->Tensor:
        scale = self.cfg.scale
        pred_r = pose[:,:3,:3]
        pred_t = pose[:,:3,3]*scale
        if self.mode =='quat':
            out_rx = BaseRefiner.quat2rot(rx)
        elif self.mode =='ortho':
            out_rx = BaseRefiner.ortho2rot(rx[:,:3],rx[:,3:])
        else:
            out_rx = rx
        out_rx = torch.bmm(out_rx,pred_r)
        if self.cfg.dtrans =='im':
            z = pred_t[:,2]*torch.exp(tx[:,2])
            x = (tx[:,0] + pred_t[:,0]/pred_t[:,2])*z
            y =  (tx[:,1] + pred_t[:,1]/pred_t[:,2])*z
            out_tx = torch.stack([x,y,z],dim=-1).view(-1,3,1).contiguous()
        elif self.cfg.dtrans =='cosy':
            z = pred_t[:,2]*tx[:,2]
            x = (tx[:,0] + pred_t[:,0]/pred_t[:,2])*z
            y =  (tx[:,1] + pred_t[:,1]/pred_t[:,2])*z
            out_tx = torch.stack([x,y,z],dim=-1).view(-1,3,1).contiguous()
        else:
            out_tx = pred_t.view(-1,3,1) + tx.view(-1,3,1).contiguous()
        if self.mode == 'svd':
            out_tx = torch.bmm(out_rx,pred_t.view(-1,3,1))+tx
        return out_rx,out_tx/scale
    def initialization(self):
        for m in self.modules():
            init_weights(m)
    @staticmethod
    def project_pose(pts,pose,inv=False,scale=1.0):
        """pts:Bx3xN,pose:bx3x4"""
        rot = pose[:,:3,:3]
        trans = pose[:,:3,3].view(-1,1,3).contiguous()*scale
        if inv:
            return torch.bmm(pts-trans,rot)
        else:
            return torch.bmm(pts,rot.transpose(1,2).contiguous())+trans
    @staticmethod
    def compute_pose(p,q):
        #p - source points BxNx3: 3d models
        #q - corresponding target points BxNx3: RGBD data
        #initial pose: coarse pose from previous network
        bs = p.shape[0]
        p_mean = p.mean(dim=1)
        q_mean = q.mean(dim=1)
        q-=q_mean
        p-=p_mean
        H = torch.bmm(q.transpose(1,2).contiguous(),p)
        [U,_,V] = torch.svd(H)
        rot= torch.bmm(U,V.transpose(1,2).contiguous())
        trans = q_mean.view(-1,3,1).contiguous()-torch.bmm(rot,p_mean.view(-1,3,1).contiguous())
        return rot,trans.view(bs,3,1)