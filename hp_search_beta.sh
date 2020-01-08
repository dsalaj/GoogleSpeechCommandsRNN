#!/usr/bin/env bash

TIME=`date "+%Y%m%d-%H:%M:%S"`
COMMON="python3 -u train.py --model_architecture=lsnn --n_hidden=1024"

#betas=(1.2 2 3.5 5)
#lif_fracs=(0 0.2 0.5 0.9)
#
#for b in "${betas[@]}";
#do
#    for lf in "${lif_fracs[@]}";
#    do
#        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $COMMON --comment=HPsearch_beta${b}_lf${lf} --beta=$b --n_lif_frac=$lf | tee so_gpu_${TIME}_beta${b}_lf${lf}.txt &
#        wait
#    done
#done

betas=(1.2 2 3.5 5)

for b in "${betas[@]}";
do
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $COMMON --comment=HPsearch --beta=$b | tee so_gpu_${TIME}_beta${b}.txt &
    wait
done