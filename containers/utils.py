import os
import time
import shutil
import numpy as np
import torch
#import amp
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel) or \
                isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    
    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }


def save_checkpoint(
        state, is_best, filename="checkpoint", bestname="model_best",
        bestname_pure='ffb6d_best'
):
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "{}.pth.tar".format(bestname))
        shutil.copyfile(filename, "{}.pth.tar".format(bestname_pure))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        ck = torch.load(filename)
        epoch = ck.get("epoch", 0)
        it = ck.get("it", 0.0)
        best_prec = ck.get("best_prec", None)
        if model is not None and ck["model_state"] is not None:
            ck_st = ck['model_state']
            if 'module' in list(ck_st.keys())[0]:
                tmp_ck_st = {}
                for k, v in ck_st.items():
                    tmp_ck_st[k.replace("module.", "")] = v
                ck_st = tmp_ck_st
            model.load_state_dict(ck_st)
        if optimizer is not None and ck["optimizer_state"] is not None:
            optimizer.load_state_dict(ck["optimizer_state"])
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> ck '{}' not found".format(filename))
        return None