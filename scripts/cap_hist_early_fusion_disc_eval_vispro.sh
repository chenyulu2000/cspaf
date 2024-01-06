#!/bin/sh
source activate cspaf
cd /home/luchenyu/cspaf
pwd

net=cap_hist_early
phase=eval
config=configs/cap_hist_early_fusion_disc_eval_vispro.yaml

now=$(date +"%Y%m%d_%H%M%S")

exp_dir=exps/${net}_${phase}_$now

mkdir $exp_dir
cp scripts/cap_hist_early_fusion_disc_eval_vispro.sh evaluate.py ${config} ${exp_dir}

python evaluate.py --config=${config} --datetime=$now --exp_dir=$exp_dir
