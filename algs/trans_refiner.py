#refiner with correspondence layer
from .base import *
from .cor_layers import PoseRegressNet
relu = nn.ReLU()
class PoseRefineNetFeat(nn.Module):
    def __init__(self):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 3,1,1)
        self.conv2 = torch.nn.Conv1d(64, 128, 3,1,1)

        self.conv3 = torch.nn.Conv1d(192, 256, 1)

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
class TransRefiner(BaseRefiner):
    def __init__(self,cfg) -> None:
        super(TransRefiner,self).__init__()
        self.feat_net = PoseRefineNetFeat()
        self.cfg = cfg
        self.mode = cfg.decode
        self.weight_layer = WeightingLayer()
        self.cpg = nn.Transformer(d_model=3, nhead=3, num_encoder_layers=3, num_decoder_layers=3,activation='gelu')
        if cfg.decode =='svd':
            self.pred = BaseRefiner.compute_pose
        else:
            self.pred = PoseRegressNet(cfg.decode,cfg.topk)
        self.relu = relu
    def forward(self,src,tgt,pose):
        scale =self.cfg.scale
        src_trans = BaseRefiner.project_pose(src,pose,scale=self.cfg.scale)         
        src_feats = self.feat_net(src_trans.transpose(1,2).contiguous())
        tgt_feats = self.feat_net(tgt.transpose(1,2).contiguous())
        B,num1,_ = src.shape
        K_topk = self.cfg.topk
        batch_mask = torch.arange(B)
        batch_mask = batch_mask.unsqueeze(1).repeat(1, B)
        batch_mask = batch_mask.flatten()
        src_keypts_idx = self.weight_layer(src_feats,K_topk)
        src_keypt_trans = src_trans[batch_mask, src_keypts_idx,:].view(B,K_topk,-1).contiguous()
        src_keypts = src[batch_mask, src_keypts_idx,:].view(B,K_topk,-1).contiguous()
        tgt_keypts_idx = self.weight_layer(tgt_feats,K_topk)
        tgt_keypts = tgt[batch_mask, tgt_keypts_idx,:].view(B,K_topk,-1).contiguous()  
        src_keypt_trans_ = src_keypt_trans.permute(1,0,2).contiguous()
        tgt_keypts = tgt_keypts.permute(1,0,2).contiguous()
        tgt_ref = self.cpg(src_keypt_trans_,tgt_keypts).transpose(0,1).contiguous()
        rx,tx = self.pred(src_keypt_trans,tgt_ref)
        out_rx,out_tx = self.decode(rx,tx,pose)
        return src_keypts/scale,tgt_ref/scale,out_rx,out_tx
