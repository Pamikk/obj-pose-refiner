from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch

def nearest_neighbor(pt1,pt2):
    """batchwise find nearest neighbor in pt set 2for points set 1
       pt1:bxnx3,pt2:bxmx3
       return idx:bxn
    """
    distance = pt1.pow(2).sum(dim=-1).unsqueeze(-1)+pt2.pow(2).sum(dim=-1).unsqueeze(-2)
    distance -= 2*torch.bmm(pt1,pt2)
    _,idx = torch.min(torch.sqrt(distance),dim=-1)
    return idx
def loss_calculation(pred_r, pred_t, target, model_points,symmetric,inference=False):
    pred_r = pred_r.transpose(2, 1).contiguous()
    bs = pred_r.shape[0]
    if len(model_points.shape)==2:
        model_points = model_points.view(1,-1,3)
    if model_points.shape[0] ==1:
        model_points = model_points.repeat(bs,1,1)
    projection=torch.add(torch.bmm(model_points, pred_r), pred_t.view(-1,1,3).contiguous())
    if not inference:
        #val,_ = torch.topk(dis,k=10,dim=-1)
        dis = torch.norm((projection - target),p=1, dim=2)
        dis = torch.mean(dis,dim=1) #+ 10*torch.mean(val,dim=1)
    else:
        dis = torch.norm((projection - target),p=1, dim=2)
        dis = torch.mean(dis,dim=1)
    return dis


class Loss_refine(_Loss):

    def __init__(self,mesh=None,if_sum= False,bs=1):
        super(Loss_refine, self).__init__(True)
        self.if_sum = if_sum
        if len(mesh.shape)==2:
            mesh = mesh.view(1,-1,3)
        if bs>1:
            self.mesh = mesh.repeat(bs,1,1).cuda()
        else:
            self.mesh = mesh.cuda()
        self.eval = False
    def forward(self, pred_r, pred_t, target, points,symmetric,eval=True):
        return loss_calculation(pred_r,pred_t, target,points,symmetric,self.eval)
