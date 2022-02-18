set -e

#!/bin/bash
n_gpu=1
cls_lst=('ape' 'benchvise' 'cam' 'can' 'cat' 'driller' 'duck' 'eggbox' 'glue' 'holepuncher' 'iron' 'lamp' 'phone')
refiner_lst=('df','df_full')
cls='ape'
name="df_15_ortho_2000_1000"
echo "Train for class $cls"
sed -i 's/cls_type.*$/cls_type: \"'$cls'\"/g' configs/linemod.yaml

sed -i 's/^name.*$/name: \"'$name'\"/g' configs/linemod.yaml
python train_ref.py configs/linemod.yaml
    
