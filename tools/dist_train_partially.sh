#!/usr/bin/env bash
set -x
FOLD=$1
PERCENT=$2
GPUS=$3
PORT=${PORT:-29860}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH


python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py configs/dota1.5/s3od_dota1.5_1percent_test.py --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT} ${@:4}

# if [[ ${TYPE} == 'baseline' ]]; then
#     python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#         $(dirname "$0")/train.py configs/SAOD_dior/V1_sup_dior_roretinanet_5p_bs2_lr0.005.py --launcher pytorch \
#         --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5} --seed 3407 --deterministic
# else
#     python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#         $(dirname "$0")/train.py configs/dota1.5/s3od_dota1.5_1percent.py --launcher pytorch \
#         --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}
# fi
