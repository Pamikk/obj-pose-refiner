# Object Pose Refiner
Refine coarse pose from previous output using RGB-D/RGB frames
## Codes Structure
+ algs
  + folders for different methods implementation
  + avoid confilict with ffb6d
  + ```base.py```
    + refiner base class definition
    + Some general use and post-processing/decode function implements
  + ```icp.py```
     + ICP algorithms/related functions
     + Problems
       + Outlier Problem
       + If need sample from point set
       + Necessary to make it batchify
  + ```loss.py```
      + different loss API
      + currently only implement L2-ADD   
  + ```df_refiner.py``` 
      + similar to densefusion but doing matches between transformed model and dmap
      + add one layer to try to help with matching
      + idea is matching -> featurea extraction -> calculate new (residual) pose from features  
      + Problems
        + if change the first matching layer to fully connected
        + batch normalization
          + current layers all with bias      
+ configs
  + ```linemod.yaml```
    + configuration file for linemode data set
+ ref_utils
  + ```utils.py```
    + helper functions
    + ``` isrotMat(rot) -> bool```
    + ``` Euler2Mat(ai,aj,ak,axes='sxyz',if_dgree=True)->Matrix```
    + ``` CalRotMatDist(rotA,rotB,if_degree=True)->angle difference```
      + with support from scipy of logm
        + can switch to cv2.Rodrigues()
      + no torch version currently
        + guess need to wait for logm-support for torch
        + or implement logm by myself :(
    + ``` CalPoseMatDist(rotA,rotB,if_degree=True)->d_rot,d_trans```
      + no torch version
    + ``` Mat2Euler(rot,axes='sxyz',if_degree=True)-> ai,aj,ak```
    + Has add torch-version to support batch size >1
      +  with appendix _torch, same parameters
  + ```pose_utils.py```
    + With Linemod default parameters
    + ```gen_coarse_pose_rand(src_pose,intrinsic=K_default,params=default_params,gen_num = gen_num_default) -> target pose, error```
      + stimulate coarse pose using normal distribution
    +   ```dmap2cloud(dmap,intrinsic=K_default)->point cloud```
      + convert depthmap to point cloud
    + ```gen_coarse_pose_rand_batch(src_pose,parameters)-> poses```
      + generate poses without test validity to support batch size > 1 without loop
    + ```project_pose_batch(rot,trans,points)-> projected target point```
      + project model points with different rotation and translation
      + support batch size >1
      + batch size determined by bs of rot an trans
      + rot:[bs,3,3],trans:[bs,3],points:[N,3] or [1,N,3]
    +  ```sample_points_rand_batch(points,num_points)->sampled point sets```
      + points: [bs,M,3] or [M,3], num_poinst:N
      + sampled:[bs,min(M,N),3]
      + add padding to support situation M<N for later 
+ containers
  + trainers and dataset from ffb6d
+ com_tuils
  + util functions from ffb6d
+ scripts
  + bash scripts to start experiments for all classes
+ ```train_ref.py```
   + general training pipeline for algorithms
+ ```ref_config.py```
  + hyperparmeters for experiments and models
  + modify self.model for different models
  + save by yaml.dump for every experiment
   
## Dependencies
  + see in ope_ffb6d/requirement.txt
  + ope_ffb6d
    + many classes and functions are imported or inherited from ope_ffb6d
  + open3d 0.9.0
## Running
  + Before running 
    + modify configs(experiment name, wandb project name)
    + modify ref_config.py
    + set up experimetns parameters in main fuction of train_ref.py 
  + Train
    + train all class of linemod at once
      ```
      bash train_lm.sh
      ```
    + train one specific class
      + modify class name in configs
      ```
      python train_ref.py configs/linemod.yaml
  + Evaluate
    ```
    python eval_refiner.py
    ```
    + no torch distributed for evaluation currently     
    + add batch evaluation script later
    + using train script
    + set up model before running              