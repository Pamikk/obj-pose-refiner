import torch
from torch import nn
relu = nn.LeakyReLU(0.01,inplace=True)

class CPG(nn.Module):
    def __init__(self):
        super(CPG, self).__init__()
        self.in_channel = 32
        self.layers = self.make_layers([16,4])
        # softmax 
        self.softmax = nn.Tanh()#nn.Softmax(dim=-1)
    def make_layers(self,channels):
        layers = []
        for i in channels:
            layers.append(nn.Conv3d(self.in_channel,i,kernel_size=3,padding=1))
            layers.append(relu)
            self.in_channel = i
        layers.append(nn.Conv3d(self.in_channel,1,kernel_size=3,padding=1))
        return nn.Sequential(*layers)
    def forward(self, src_feat, tgt_feat, candidates, r, s):
        B, N, C, _ = candidates.shape
        grid_size = int((2*r)/s+1)

        src_feat_volume = src_feat.reshape(B, N, 1, 1, 1, 32).repeat(1, 1, grid_size, grid_size, grid_size, 1)
        tgt_feat_volume = tgt_feat.reshape(B, N,grid_size, grid_size, grid_size, 32)
        cost_volume = src_feat_volume - tgt_feat_volume


        x = cost_volume.permute(0, 1, 5, 2, 3, 4).contiguous()
        x = x.flatten(start_dim=0,end_dim=1)

        x = self.layers(x)
        x = x.reshape(B, N, C)

        weights = self.softmax(x)
        weights = weights.unsqueeze(-1).repeat(1,1,1,3).contiguous()
        # weights,  candidates: B x N x C x 3 
        refined_tgt = torch.sum(torch.mul(weights,candidates), -2) 
        # vcp: B x N x 3
        return refined_tgt




