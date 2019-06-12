#!/usr/bin/env bash
DATA=cahn-hilliard
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u run.py \
    --is_training True \
    --dataset_name mnist \
    --train_data_paths ~/data/$DATA/train.npz \
    --valid_data_paths ~/data/$DATA/valid.npz \
    \ #--pretrained_model pretrain_model/$DATA/model.ckpt-80000 \
    --save_dir checkpoints/$DATA \
    --gen_frm_dir results/$DATA \
    --model_name e3d_lstm \
    --allow_gpu_growth True \
    --img_channel 1 \
    --img_width 64 \
    --input_length 10 \
    --total_length 20 \
    --filter_size 5 \
    --num_hidden 64,64,64,64 \
    --patch_size 4 \
    --layer_norm True \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_delta_per_iter 0.00002 \
    --lr 0.0001 \
    --batch_size 4 \
    --max_iterations 60000 \
    --display_interval 100 \
    --test_interval 1000 \
    --snapshot_interval 5000
