import numpy as np
#first axis, parity,repetition,frame
# parity transforamtion - symmetric, right hand to left hand
# frame, switch x,z axis or not 
import math
from scipy.linalg import logm
import cv2
import torch
_AXES2TUPLE = { 
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0

_MAX_FLOAT = np.maximum_sctype(np.float)
_FLOAT_EPS = np.finfo(np.float).eps
def isrotMat(rot):
    # justify if is valid rotation mat
    tmp = np.matmul(rot,rot.T)
    tmp = tmp-np.identity(3,dtype = rot.dtype)
    return np.linalg.norm(tmp)<1e-4
def buildRotmat(j,k,ang,if_degree = True):
    #j,k:next axis and next next axi
    M = np.eye(3)
    ang = np.deg2rad(ang) if if_degree else ang
    M[j,j] = math.cos(ang)
    M[k,k] = math.cos(ang)
    M[j,k] = -math.sin(ang)
    M[k,j] = math.sin(ang)
    return M
def Euler2Mat(ai,aj,ak,axes="sxyz",if_degree=True):
    # input ai,aj,ak angles for first axis,second axis, third axis
    # axes determine corresponding axes and rotation order
    # output: rotation matrix
    startaxis,parity,rep,frame = _AXES2TUPLE[axes.lower()]
    i = startaxis
    j = (startaxis+1+parity)%3
    k = (startaxis+2-parity)%3
    if frame:
        ai,ak = ak,ai
    if parity:
        ai,aj,ak = -ai,-aj,-ak
    if rep:
        return np.matmul(buildRotmat(j,k,ak,if_degree),np.matmul(buildRotmat(k,i,aj,if_degree),buildRotmat(j,k,ai,if_degree)))
    else:
        return np.matmul(buildRotmat(i,j,ak,if_degree),np.matmul(buildRotmat(k,i,aj,if_degree),buildRotmat(j,k,ai,if_degree)))

def Mat2Euler(rot,axes="sxyz",if_degree=True):
    # input rotation mat and rotation axes
    # output: euler angles in order of rot axes
    # by default it will be row-x, pitch-y, yall-z
    # if_degree, output will in degrees otherwise in radius
    if not(isrotMat(rot)):
        print("Rotation Matrix is not valid.")
    startaxis,parity,rep,frame = _AXES2TUPLE[axes.lower()] 
    i = startaxis
    j = (startaxis+1+parity)%3
    k = (startaxis+2-parity)%3
    M = np.array(rot,dtype=np.float64,copy=False)[:3,:3]
    if rep:
        sinj = math.sqrt(M[i,j]**2+M[i,k]**2)
        if sinj>_EPS4:
            ai = math.atan2(M[i,j],M[i,k])
            aj = math.atan2(sinj,M[i,i])
            ak = math.atan2(M[j,i],-M[k,i])
        else:
            ai = math.atan2(M[k,j],M[k,k])
            aj = math.atan2(sinj,M[i,i])
            ak = 0.0
    else:
        cosj = math.sqrt(M[i,i]*M[i,i]+M[j,i]*M[j,i])
        if cosj >_EPS4:
            ai = math.atan2(M[k,j],M[k,k])
            aj = math.atan2(-M[k,i],cosj)
            ak = math.atan2(M[j,i],M[i,i])
        else:
            ai = math.atan2(-M(j,k),M[j,j])
            aj = math.atan2(-M[k,i],cosj)
            ak = 0.0
    if parity:
        ai,aj,ak = -ai,-aj,-ak
    if frame:
        ai,ak = ak,ai
    if if_degree:
        return tuple(map(np.rad2deg,[ai,aj,ak]))
    else:
        return ai,aj,ak
################ torch version tensor batch version ###########################
def buildRotmat_torch(j,k,ang,if_degree = True):
    #j,k:next axis and next next axi
    bs = len(ang)
    M = torch.eye(3,3).view(1,3,3).repeat(bs,1,1)
    ang = np.radians(ang) if if_degree else ang
    M[:,j,j] = torch.cos(ang)
    M[:,k,k] = torch.cos(ang)
    M[:,j,k] = -torch.sin(ang)
    M[:,k,j] = torch.sin(ang)
    return M
def Euler2Mat_torch(ai,aj,ak,axes="sxyz",if_degree=True):
    # input ai,aj,ak angles for first axis,second axis, third axis
    # axes determine corresponding axes and rotation order
    # output: rotation matrix
    startaxis,parity,rep,frame = _AXES2TUPLE[axes.lower()]
    i = startaxis
    j = (startaxis+1+parity)%3
    k = (startaxis+2-parity)%3
    if frame:
        ai,ak = ak,ai
    if parity:
        ai,aj,ak = -ai,-aj,-ak
    if rep:
        return torch.bmm(buildRotmat_torch(j,k,ak,if_degree),torch.bmm(buildRotmat_torch(k,i,aj,if_degree),buildRotmat_torch(j,k,ai,if_degree)))
    else:
        return torch.bmm(buildRotmat_torch(i,j,ak,if_degree),torch.bmm(buildRotmat_torch(k,i,aj,if_degree),buildRotmat_torch(j,k,ai,if_degree)))
def Mat2Euler_torch(rot,axes="sxyz",if_degree=True):
    # input rotation mat and rotation axe
    # output: euler angles in order of rot axes
    # by default it will be row-x, pitch-y, yall-z
    # if_degree, output will in degrees otherwise in radius
    startaxis,parity,rep,frame = _AXES2TUPLE[axes.lower()] 
    i = startaxis
    j = (startaxis+1+parity)%3
    k = (startaxis+2-parity)%3
    if len(rot.shape)==3:
        M = rot[:,:3,:3]
    else:
        M = rot[:3,:3]
        M = M.unsqueeze(dim=0)
    if rep:
        sinj = torch.sqrt(M[:,i,j]**2+M[:,i,k]**2)
        mask = sinj<=_EPS4
        ai = torch.atan2(M[:,i,j],M[:,i,k])
        aj = torch.atan2(sinj,M[:,i,i])
        ak = torch.atan2(M[:,j,i],-M[:,k,i])
        ai[mask] = torch.atan2(M[mask,k,j],M[mask,k,k])
        aj[mask] = torch.atan2(sinj[mask],M[mask,i,i])
        ak[mask] = 0.0
    else:
        cosj = torch.sqrt(M[:,i,i]*M[:,i,i]+M[:,j,i]*M[:,j,i])
        mask = cosj <=_EPS4
        ai = torch.atan2(M[:,k,j],M[:,k,k])
        aj = torch.atan2(-M[:,k,i],cosj)
        ak = torch.atan2(M[:,j,i],M[:,i,i])
        ai[mask] = torch.atan2(-M[mask,j,k],M[mask,j,j])
        aj[mask] = torch.atan2(-M[mask,k,i],cosj[mask])
        ak[mask] = 0.0
    if parity:
        ai,aj,ak = -ai,-aj,-ak
    if frame:
        ai,ak = ak,ai
    if if_degree:
        return np.degrees(torch.stack((ai,aj,ak),dim=-1))
    else:
        return torch.stack((ai,aj,ak),dim=-1)
