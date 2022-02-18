set -e

#!/bin/bash
n_gpu=1
if [[ $# -eq 0 ]]
then
  cls_lst=('ape' 'benchvise' 'cam' 'can' 'cat' 'driller' 'duck' 'eggbox' 'glue' 'holepuncher' 'iron' 'lamp' 'phone')
else
  cls_lst=($1)
fi
for cls in ${cls_lst[@]}
do
    echo "Eval for class $cls"
    sed -i 's/cls_type.*$/cls_type: \"'$cls'\"/g' configs/linemod-test.yaml
    python eval_refiner.py
done
    
