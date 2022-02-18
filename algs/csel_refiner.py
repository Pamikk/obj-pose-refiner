#refiner with correspondence layer
from .base import *
from .cor_layers import PoseRegressNet,PoseRefineNetFeat
relu = nn.LeakyReLU(0.01)
#Correspondence Selection Refiner
class WeightingLayer(nn.Module):
    def __init__(self):
        super(WeightingLayer, self).__init__()
        self.fc1 =nn.Conv1d(32, 16,1,True)
        self.fc2 = nn.Conv1d(16, 8,1,True)
        self.fc3 = nn.Conv1d(8, 1,1,True)
    
    def forward(self, x, K = 64):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        
        _,topk_indices = torch.topk(x, K, dim = -1)
        topk_indices = topk_indices.flatten()
        return topk_indices
class CPG(nn.Module):
    #2D version
    def __init__(self,in_channel):
        super(CPG, self).__init__()
        self.in_channel = in_channel
        self.layers = self.make_layers([16,8])
        self.tanh = nn.Tanh()
        # softmax 
        self.softmax = nn.Tanh()
    def make_layers(self,channels):
        layers = []
        for i in channels:
            layers.append(nn.Conv1d(self.in_channel,i,kernel_size=1))
            layers.append(relu)
            self.in_channel = i
        layers.append(nn.Conv1d(self.in_channel,1,1))
        return nn.Sequential(*layers)
    def forward(self, src_feat, tgt_feat, src,tgt):
        B,N1 = src_feat.shape[:2]
        N2 = tgt_feat.shape[1]
        src_feat_plat =src_feat.unsqueeze(2).repeat(1,1,N2,1)
        tgt_feat_plat = tgt_feat.unsqueeze(1).repeat(1,N1,1,1)
        point_pairs = torch.cat((src.unsqueeze(2).repeat(1,1,N2,1),tgt.unsqueeze(1).repeat(1,N1,1,1)),dim=-1)
        cost = src_feat_plat-tgt_feat_plat
        x = torch.cat((point_pairs,cost),dim=-1)
        x = x.flatten(start_dim=1,end_dim=2).transpose(1,2).contiguous()
        x = self.tanh(self.layers(x))
        idx = torch.topk(x,k=N1,dim=-1,largest=False).indices.flatten()
        point_pairs = point_pairs.flatten(start_dim=1,end_dim=2)
        batch_mask = torch.arange(B)
        batch_mask = batch_mask.unsqueeze(1).repeat(1, B)
        batch_mask = batch_mask.flatten()
        chosen = point_pairs[batch_mask, idx,:].view(B,N1,-1).contiguous()
        # vcp: B x N x 3
        return chosen
class CSelRefiner(BaseRefiner):
    def __init__(self,cfg) -> None:
        super(CSelRefiner,self).__init__()
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
        num2 = tgt.shape[1]
        K_topk = self.cfg.topk
        batch_mask = torch.arange(B)
        batch_mask = batch_mask.unsqueeze(1).repeat(1, B)
        batch_mask = batch_mask.flatten()
        if num1!=K_topk:
            src_keypts_idx = self.weight_layer(src_feats,K_topk)
            assert src_keypts_idx.max()<num1
            src_keypts_trans = src_trans[batch_mask, src_keypts_idx,:].view(B,K_topk,-1).contiguous()
            src_keypts = src[batch_mask, src_keypts_idx,:].view(B,K_topk,-1).contiguous()
        else:
            src_keypts_trans = src_trans
            src_keyfeats = src_feats.transpose(1,2).contiguous()
        if num2!=K_topk:
            tgt_keypts_idx = self.weight_layer(tgt_feats,K_topk)
            
            assert tgt_keypts_idx.max()<num2
            tgt_keypts = tgt[batch_mask, tgt_keypts_idx,:].view(B,K_topk,-1).contiguous()
            tgt_keyfeats = tgt_feats[batch_mask, :, tgt_keypts_idx].view(B,K_topk,-1).contiguous()
        else:
            tgt_keypts=tgt
        rx,tx = self.pred(src_keypts_trans, tgt_keypts)
        out_rx,out_tx = self.decode(rx,tx,pose)

        return src_keypts/scale, tgt_keypts/scale,out_rx,out_tx

