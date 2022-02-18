import torch
import torch.nn as nn
import sys
sys.path.append("../ffb6d")
from models.RandLA.helper_tool import DataProcessing
def knn(pt1,pt2,k=1):
    """batchwise find nearest neighbor in pt set 2for points set 1
       pt1:bxnx3,pt2:bxmx3
       return idx:bxnxk
    """
    distance = pt1.pow(2).sum(dim=-1).unsqueeze(-1)+pt2.pow(2).sum(dim=-1).unsqueeze(-2)
    distance -= 2*torch.bmm(pt1,pt2.transpose(1,2).contiguous())
    distance = torch.sqrt(distance)
    #print(pt1.shape,pt2.shape)
    #distance = torch.norm(pt1-pt2,p=2,dim=-1)
    dist,idx = torch.topk(distance,k,dim=-1,largest=False)
    return dist,idx
class SelFeatTgt(nn.Module):
    def __init__(self):
        super(SelFeatTgt, self).__init__()
        self.nsample =32
        self.pool = nn.MaxPool1d(kernel_size = self.nsample)

    def forward(self, candidate_pts, src_keypts, tgt_pts_xyz, tgt_deep_feat_pts):
        """
        Input:
            candidate_pts: candidate corresponding points (B x C x 3)
            src_keypts: keypoints in src point cloud (B x K_topk x 3)
            tgt_pts_xyz: original points in target point cloud (B x N x 3)
            tgt_deep_feat_pts: deep features for tgt point cloud (B x N x num_feats)
        
        Output: 
            tgt_keyfeats_cat: concatenated local coordinates of candidate points and 
                              normalized deep features (B x K_topk x C x nsample x (3 + num_feats))
        """
        B, _, _ = src_keypts.shape
        device = src_keypts.device
        
        # sample and group the candidate points
        # candidate_pts_grouped_xyz: B x K_topk x C x nsample x 3
        nsample = self.nsample
    


        # use KNN to find nearest neighbors of the candidates in tgt_pts 
        # dist: (B x (K_topk x C) x k_nn)
        # idx: (B x (K_topk x C) x k_nn)
        # candidate_pts_k: (B x K_topk x C x nsample x 3)
        candidate_pts_flat = torch.flatten(candidate_pts, start_dim = 1, end_dim = 2)
        query_pts = candidate_pts_flat
        ref_pts = tgt_pts_xyz + candidate_pts_flat.mean(dim=1) - tgt_pts_xyz.mean(dim=1)
        #idx =  DataProcessing.knn_search( ref_pts.cpu(),query_pts.cpu(), nsample)
        _, idx = knn(query_pts,ref_pts,k = nsample)
        #print(query_pts.shape,ref_pts.shape)
        #print(idx.max())
        assert idx.max()< ref_pts.shape[1]

        
        
        # pick deep features of tgt_pts with idx
        N_keypts = src_keypts.shape[1]
        C_candidates = candidate_pts.shape[2]
        C_deep_feat = tgt_deep_feat_pts.shape[1]
        
        idx_1_mask = torch.arange(B)
        idx_1_mask = idx_1_mask.unsqueeze(1).repeat(1, B)
        idx_1_mask = idx_1_mask.flatten()
        idx_2_mask = idx.flatten()
        
        #tgt_deep_feat_pts = torch.rand_like(tgt_deep_feat_pts).to(device)#
        tgt_feat_picked = tgt_deep_feat_pts.transpose(1,2).contiguous()
        tgt_feat_picked = tgt_feat_picked[idx_1_mask, idx_2_mask, :].contiguous().view(B, N_keypts, C_candidates,  nsample, C_deep_feat).contiguous()

        tgt_pts_picked = tgt_pts_xyz[idx_1_mask, idx_2_mask, :].view(B, N_keypts, C_candidates,  nsample, 3).contiguous()
        tgt_pts_picked = tgt_pts_picked.mean(dim=3)

        # normalize the picked deep features from tgt_pts
        tgt_feat_norm = tgt_feat_picked
         
        B, N, C, _,out_channel= tgt_feat_norm.shape
        X = tgt_feat_norm.permute(0, 1, 2, 4, 3).contiguous()
        X = torch.flatten(X, start_dim = 1, end_dim = 3)
        X = self.pool(X)
        tgt_keyfeats= X.view(B, N, C, out_channel, 1).squeeze(-1)
        
        return tgt_pts_picked,tgt_keyfeats
