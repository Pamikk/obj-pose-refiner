import torch
import torch.nn as nn
# input: B x N x 3 
# output:B x N x C x 3
relu = nn.LeakyReLU(0.01,inplace=True)
def voxelize(points,search_radius,voxel_len):
    volume = int(2*search_radius/voxel_len +1)
    candidates = points.unsqueeze(-2).repeat(1,1,volume**3,1)
    # create 3D grid for each of x,y,z
    device = points.device
    xrange = torch.arange(-search_radius-voxel_len/2, search_radius, voxel_len).to(device)
    yrange = torch.arange(-search_radius-voxel_len/2, search_radius, voxel_len).to(device)
    zrange = torch.arange(-search_radius-voxel_len/2, search_radius, voxel_len).to(device)
    xgrid, ygrid, zgrid = torch.meshgrid(xrange, yrange, zrange)
    candidates[...,0]+=xgrid.reshape(1,1,-1)
    candidates[...,1]+=ygrid.reshape(1,1,-1)
    candidates[...,2]+=zgrid.reshape(1,1,-1)
    return candidates
class PoseRefineNetFeat(nn.Module):
    def __init__(self,in_channel =3):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channel, 16, 3,1,1)
        self.bn1 = torch.nn.InstanceNorm1d(16,affine=True)
        self.conv2 = torch.nn.Conv1d(16, 32, 3,1,1)

        self.conv3 = torch.nn.Conv1d(48, 32, 1)
        self.relu = relu

    def forward(self, x):

        x1 = self.relu(self.conv1(x))

        x2 = self.relu(self.conv2(x1))

        pointfeat = torch.cat([x1,x2], dim=1)

        x = self.relu(self.conv3(pointfeat))
        return x

class PoseRegressNet(nn.Module):
    def __init__(self,mode,topk=64):
        super(PoseRegressNet, self).__init__()
        self.feat = PoseRefineNetFeat(6)
        self.conv1_r = nn.Linear(32, 16)
        self.conv1_t = nn.Linear(32, 16)
        self.ap = torch.nn.AvgPool1d(topk)

        if mode=='quat':
            self.pred_r = nn.Linear(16, 4) #quaternion
        elif mode=='ortho':
            self.pred_r = nn.Linear(16, 6) # rotation paramterization
        self.pred_t = nn.Linear(16,3)
        self.relu = relu
    def forward(self,src,tgt=None):
        #BxNx3,BxNx3
        if (tgt==None):
            x = src.transpose(1,2).contiguous()
        else:
            x = torch.cat((src.transpose(1,2).contiguous(),tgt.transpose(1,2).contiguous()),dim = 1)
        if x.shape[1]==6:
            feat = self.feat(x)
        else:
            feat = x
        feat = feat.mean(dim=-1)
        feat = feat.view(-1,32).contiguous()
        rx = self.relu(self.conv1_r(feat))
        tx = self.relu(self.conv1_t(feat))   

        rx = self.pred_r(rx)
        tx = self.pred_t(tx)
        return rx,tx