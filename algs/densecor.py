#refiner with correspondence layer
from .base import *
from .cor_layers import PoseRegressNet,WeightingLayer,PoseRefineNetFeat
relu = nn.LeakyReLU(0.01)
class CPG(nn.Module):
    #2D version
    def __init__(self,in_channel):
        super(CPG, self).__init__()
        self.in_channel = in_channel
        self.layers = self.make_layers([16,4])
        self.wconv = nn.Conv2d(self.in_channel,1,1)
        self.bconv = nn.Conv2d(self.in_channel,3,1)
        self.tanh = nn.Tanh()
        # softmax 
        self.softmax = nn.Tanh()
    def make_layers(self,channels):
        layers = []
        for i in channels:
            layers.append(nn.Conv2d(self.in_channel,i,kernel_size=1))
            layers.append(relu)
            self.in_channel = i
        return nn.Sequential(*layers)
    def forward(self, src_feat, tgt_feat, candidates):
        N1 = src_feat.shape[1]
        N2 = tgt_feat.shape[1]
        src_feat_plat =src_feat.unsqueeze(2).repeat(1,1,N2,1)
        tgt_feat_plat = tgt_feat.unsqueeze(1).repeat(1,N1,1,1)

        cost = src_feat_plat-tgt_feat_plat


        x = cost.permute(0,3,1,2).contiguous()

        x = self.layers(x)
        weights = self.softmax(self.wconv(x)).squeeze(dim=1)
        bias = self.tanh(self.bconv(x)).sum(dim=-1).transpose(1,2)
        #weights = self.softmax(x)
        weights = weights.unsqueeze(-1).repeat(1,1,1,3).contiguous()
        # weights,  candidates: B x N x C x 3 
        refined_tgt = torch.sum(torch.mul(weights,candidates), -2) + bias 
        # vcp: B x N x 3
        return refined_tgt
class DCorRefiner(BaseRefiner):
    def __init__(self,cfg) -> None:
        super(DCorRefiner,self).__init__()
        self.feat_net = PoseRefineNetFeat()
        self.weight_layer = WeightingLayer()
        self.cfg = cfg
        self.mode = cfg.decode
        self.cpg = CPG(35)
        if cfg.decode =='svd':
            self.pred = BaseRefiner.compute_pose
        else:
            self.pred = PoseRegressNet(cfg.decode,cfg.topk)
    def forward(self,src,tgt,pose):
        scale =self.cfg.scale
        src_trans = BaseRefiner.project_pose(src,pose,scale=self.cfg.scale)         
        src_feats = self.feat_net(src_trans.transpose(1,2).contiguous())
        tgt_feats = self.feat_net(tgt.transpose(1,2).contiguous())
        B,num1,_ = src.shape
        K_topk = self.cfg.topk
        if num1 > K_topk:
            src_keypts_idx = self.weight_layer(src_feats,K_topk)
            batch_mask = torch.arange(B)
            batch_mask = batch_mask.unsqueeze(1).repeat(1, B)
            batch_mask = batch_mask.flatten()
            
            assert src_keypts_idx.max()<num1
            src_keypts_trans = src_trans[batch_mask, src_keypts_idx,:].view(B,K_topk,-1).contiguous()
            src_keypts = src[batch_mask, src_keypts_idx,:].view(B,K_topk,-1).contiguous()
            src_keyfeats = src_feats[batch_mask, :, src_keypts_idx].view(B,K_topk,-1)
        else:
            src_keypts = src
            src_keypts_trans = src_trans
            src_keyfeats = src_feats.contiguous()
        src_keyfeats = torch.cat((src_keypts_trans,src_keyfeats),dim=-1)
        tgt_keyfeats = torch.cat((tgt,tgt_feats.transpose(1,2).contiguous()),dim=-1)
        tgt_ref = self.cpg(src_keyfeats, tgt_keyfeats,tgt)
        rx,tx = self.pred(src_keypts_trans, tgt_ref)
        out_rx,out_tx = self.decode(rx,tx,pose)

        return src_keypts/scale, tgt_ref/scale,out_rx,out_tx

