ulimit -n 4096
set -x

swapon --show
free -h
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=8

EXP_DIR=exps/serv
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 28418 \
        --use_env \
        main.py \
        --output_dir ${EXP_DIR} \
        --root_dir dataset/ERA5 \
        --output_mode serv \
        --num_workers 4 \
        --epochs 1000 \
        --batch_size 4 \
        --model_name EconClimNet \
        --lr 5e-5 \









