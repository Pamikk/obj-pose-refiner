#refiner with correspondence layer
from .base import *
from .cor_layers import *

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
            self.pred = PoseRegressNet(cfg.decode)
    def forward(self,src,target,pose,grid):
        device = src.device
        scale =self.cfg.scale
        B, num, _ = src.shape
        src_pts = src.transpose(1,2)
        tgt_pts = target.transpose(1,2)

          
        src_feats = self.feat_net(src_pts)
        if num > 64:
            K_topk = 64
            src_keypts_idx = self.weight_layer(src_feats)
            batch_mask = torch.arange(B)
            batch_mask = batch_mask.unsqueeze(1).repeat(1, B)
            batch_mask = batch_mask.flatten()
            
            assert src_keypts_idx.max()<num
            src_keypts = src_pts[batch_mask, :, src_keypts_idx].view(B,K_topk,-1)
            src_keyfeats = src_feats[batch_mask, :, src_keypts_idx].view(B,K_topk,-1).transpose(1,2).contiguous()
        else:
            src_keypts = src
            src_keyfeats = src_feats
        tgt_deep_feat_pts = self.feat_net(tgt_pts)
        #tgt_deep_feat_pts = torch.rand_like(tgt_deep_feat_pts).to(device)
        src_transformed = BaseRefiner.project_pose(src_keypts,pose,scale=self.cfg.scale)
        #print(torch.norm(target.mean(dim=1)-src_transformed.mean(dim=1),2))
        
        
        r,s = grid
        src_grid = voxelize(src_transformed, r, s)

        # group the tgt_pts
        tgt_gcf = SelFeatTgt()
        candidate_pts,tgt_keyfeats = tgt_gcf(src_grid, src_keypts, target, tgt_deep_feat_pts)

        candidate_pts =  src_grid
        #tgt_keyfeats = torch.rand(B,64,216,32).to(device)
        tgt_vcp = self.cpg(src_keyfeats, tgt_keyfeats, candidate_pts, r, s)
        rx,tx = self.pred(src_keypts, tgt_vcp)
        out_rx,out_tx = self.decode(rx,tx,pose)

        return src_keypts/scale, tgt_vcp/scale,out_rx,out_tx

