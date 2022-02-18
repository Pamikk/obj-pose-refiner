#configurations for refiners
import numpy
class Config(object):
    def __init__(self,model="df"):
        #generation params
        angle_std = 15
        angle_max = 45
        x_std = 0.01
        y_std  =0.01
        z_std = 0.05
        self.gen_params = [angle_std, angle_max, x_std, y_std, z_std]
        self.exp_description = "" #comment for current experiment

        #network params
        self.model = model#[icp,df]
        self.decode = 'ortho'#'svd'|'ortho'|'quat'
        self.dtrans = 'cosy' #'im'|'cosy'
        self.disentangle = True
        self.if_sum = False
        self.update_sep = True
        self.lr = 0.01
        self.schedule = [500,1000,2500,3000]
        self.bs = 1
        self.schedule = list((numpy.array(self.schedule)/self.bs).astype(int))
        # icp params
        self.fx = 572.4114
        self.fy = 573.57043
        self.scale = 1.0
        if self.model =="icp":
            self.iters = 200
            self.threshold = 1e-3
            self.outlier = 0.1
            self.decode = 'svd'
            self.pred = 'none' #[v1,v2,v3]
            self.num_points = 12800#linemod
            self.num_points_mesh =5000 
            self.scale = 1.0
        #densefusion
        if  self.model in ['df','dfv2']:
            self.num_points = 8000#linemod
            self.iters = 4
            self.threshold = 0.01
            self.num_points_mesh =4000
            self.channel = 1000
            self.lr = 0.001
            self.topk = 512
            self.schedule = [1000,2000,3000]
            #128000 rgbd pt cloud
            #5841 model pt cloud
        if self.model=='cor':
            self.num_points = 8000#linemod
            self.num_points_mesh =5000
            self.grids = (0.5,0.2)
            self.topk = 128
            self.iters = 1
        if self.model == 'dcor':
            self.num_points = 5000#linemod
            self.num_points_mesh =2000
            self.topk = 128
            self.iters = 1
            self.lr = 0.01
            self.schedule = [1200,2400,3000]
        if self.model == 'csel':
            self.num_points = 8000#linemod
            self.num_points_mesh =5000
            self.topk = 256
            self.iters = 1
        if self.model == 'transf':
            self.num_points = 8000
            self.num_points_mesh =5000
            self.topk = 256
            self.iters = 1
            self.lr = 0.01
            self.schedule = [1000,2000,3000]
    def print_properties(self):
        print("Generation Parameters")
        print(f'current parameter for sampling distribution')
        angle_std, angle_max, x_std, y_std, z_std = self.gen_params
        print(f'Euler Angles -- std:{angle_std}, max:{angle_max}')
        print(f'Translation stds-- x:{x_std},y:{y_std},z:{z_std}')
        print(f'decode mode:{self.decode}')
        if self.model in (""):
            print(f"Experiment for {self.model}:")
        elif self.model =="icp":
            print("Network Paramters:")
            print(f"max iters:{self.iters}, terminate threshold:{self.threshold}, outlier threshold:{self.outlier}")
        else:
            print(f"lr:{self.lr}")
        if 'df' in self.model :
            print("Network Parameters:")
            print(f"max iters:{self.iters}, cloud point:{self.num_points}, mesh points:{self.num_points_mesh},network channel:{self.channel}")
            print("Train Parameters:")
        else:
            print("Network Parameters:")
            print(f" cloud point:{self.num_points}, mesh points:{self.num_points_mesh},iters:{self.iters}")
            



