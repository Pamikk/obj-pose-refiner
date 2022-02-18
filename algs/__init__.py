from .base import *
from .icp import *
from .loss import Loss_refine
from .df_refiner import *
from .df_refinerv2 import DFRefiner as DFRefinerv2
from .mycor_refiner import CorRefiner
from .densecor import DCorRefiner
from .csel_refiner import CSelRefiner
from .trans_refiner import TransRefiner
Refiners = {'icp':ICP,'df':DFRefiner,'dfv2':DFRefinerv2,'cor':CorRefiner,'dcor':DCorRefiner,'csel':CSelRefiner,'transf':TransRefiner}
ref_loss = Loss_refine