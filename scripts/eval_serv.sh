
EXP_DIR=exps/serv
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 28438 \
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
        --resume exps/serv/checkpoint_best.pth \
        --analysis \
        --analysis_dir dataset/CMIP5/1ptco2_adjusted \
        --save_path results/1ptco2_adjusted \

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 28438 \
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
        --resume exps/serv/checkpoint_best.pth \
        --analysis \
        --analysis_dir dataset/CMIP5/cdr_adjusted \
        --save_path results/cdr_adjusted \

