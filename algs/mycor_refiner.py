#refiner with correspondence layer
from .base import *
from .cor_layers import PoseRefineNetFeat,WeightingLayer,PoseRegressNet,voxelize
relu = relu = nn.LeakyReLU(0.01)
class CPG(nn.Module):
    def __init__(self):
        super(CPG, self).__init__()
        self.in_channel = 35
        self.layers = self.make_layers([16,4])
        # softmax 
        self.softmax = nn.Softmax(dim=-1)
    def make_layers(self,channels):
        layers = []
        for i in channels:
            layers.append(nn.Conv3d(self.in_channel,i,kernel_size=3,padding=1))
            layers.append(relu)
            self.in_channel = i
        layers.append(nn.Conv3d(self.in_channel,3,kernel_size=3,padding=1))
        return nn.Sequential(*layers)
    def forward(self, diff, candidates, r, s):
        B, N, C, _ = candidates.shape
        grid_size = int((2*r)/s+1)
        diff = torch.cat((candidates,diff),dim=-1)
        cost_volume = diff.view(B,N,grid_size,grid_size,grid_size,-1)


        x = cost_volume.permute(0, 1, 5, 2, 3, 4).contiguous()
        x = x.flatten(start_dim=0,end_dim=1)

        x = self.layers(x)
        x = nn.functional.max_pool3d(x,kernel_size = grid_size)
        return x.view(B,3,N).transpose(1,2).contiguous()

        '''weights = x
        weights = weights.unsqueeze(-1).repeat(1,1,1,3).contiguous()
        # weights,  candidates: B x N x C x 3 
        refined_tgt = torch.sum(torch.mul(weights,candidates), -2) 
        # vcp: B x N x 3
        return refined_tgt'''

class CorRefiner(BaseRefiner):
    def __init__(self,cfg) -> None:
        super(CorRefiner,self).__init__()
        self.feat_net = PoseRefineNetFeat()
        self.weight_layer = WeightingLayer()
        self.cfg = cfg
        self.mode = cfg.decode
        self.cpg = CPG()
        if cfg.decode =='svd':
            self.pred = BaseRefiner.compute_pose
        else:
            self.pred = PoseRegressNet(cfg.decode,cfg.topk)
    def forward(self,src,tgt,pose,grid):
        device = src.device
        scale =self.cfg.scale
        src_trans = BaseRefiner.project_pose(src,pose,scale=self.cfg.scale)         
        src_feats = self.feat_net(src_trans.transpose(1,2).contiguous())
        tgt_feats = self.feat_net(tgt.transpose(1,2).contiguous())
        B,num1,_ = src.shape
        num2 = tgt.shape[1]
        K_topk = self.cfg.topk
        batch_mask = torch.arange(B)
        batch_mask = batch_mask.unsqueeze(1).repeat(1, B)
        batch_mask = batch_mask.flatten()
        if num1 > K_topk:
            src_keypts_idx = self.weight_layer(src_feats,K_topk)
            
            assert src_keypts_idx.max()<num1
            src_keypts_trans = src_trans[batch_mask, src_keypts_idx,:].view(B,K_topk,-1).contiguous()
            src_keypts = src[batch_mask, src_keypts_idx,:].view(B,K_topk,-1).contiguous()
            src_keyfeats = src_feats[batch_mask, :, src_keypts_idx].view(B,K_topk,-1)
        else:
            src_keypts = src
            src_keypts_trans = src_trans
            src_keyfeats = src_feats.transpose(1,2).contiguous()
        if num2> K_topk*2:
            tgt_keypts_idx = self.weight_layer(tgt_feats,K_topk*2)
            assert tgt_keypts_idx.max()<num2
            tgt_keypts = tgt[batch_mask, tgt_keypts_idx,:].view(B,K_topk*2,-1).contiguous()
            tgt_keyfeats = tgt_feats[batch_mask, :, tgt_keypts_idx].view(B,K_topk*2,-1)
        else:
            tgt_keypts = tgt
            tgt_keyfeats = tgt_feats.transpose(1,2).contiguous()
        
        
        r,s = grid
        candidate_pts = voxelize(src_keypts_trans, r, s)
        cand_num = candidate_pts.shape[2]
        pt1= candidate_pts.flatten(start_dim=1,end_dim=2)
        pt2 = tgt_keypts
        pt1 -= pt1.mean(dim=1,keepdim=True)
        pt2 -= pt2.mean(dim=1,keepdim=True)
        distance = pt1.pow(2).sum(dim=-1).unsqueeze(-1)+pt2.pow(2).sum(dim=-1).unsqueeze(-2)
        distance -= 2*torch.bmm(pt1,pt2.transpose(1,2).contiguous())
        idx = torch.min(torch.sqrt(distance),dim=-1).indices
        src_feats_vol = src_keyfeats.unsqueeze(2).repeat(1,1,cand_num,1)
        tgt_feats_vol = tgt_keyfeats[batch_mask,idx.flatten(),:].view(B,K_topk,cand_num,-1)
        assert (tgt_feats_vol.shape==src_feats_vol.shape)

        diff = src_feats_vol-tgt_feats_vol

        tgt_ref = self.cpg(diff, candidate_pts, r, s)
        rx,tx = self.pred(src_keypts, tgt_ref)
        out_rx,out_tx = self.decode(rx,tx,pose)

        return src_keypts/scale, tgt_ref/scale,out_rx,out_tx

