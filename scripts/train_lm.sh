set -e

#!/bin/bash
n_gpu=1
if [[ $# -eq 0 ]]
then
  cls_lst=('ape' 'benchvise' 'cam' 'can' 'cat' 'driller' 'duck' 'eggbox' 'glue' 'holepuncher' 'iron' 'lamp' 'phone')
else
  cls_lst=($1)
fi
name="dfv2"
sed -i 's/^name.*$/name: \"'$name'\"/g' configs/linemod.yaml
for cls in ${cls_lst[@]}
do
    echo "Train for class $cls"
    sed -i 's/cls_type.*$/cls_type: \"'$cls'\"/g' configs/linemod.yaml
    sed -i 's/^name.*$/name: \"'$name'\"/g' configs/linemod.yaml
    CUDA_VISIBLE_DEVICES=0 python train_ref.py configs/linemod.yaml
done
    
