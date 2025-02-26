# CUDA_VISIBLE_DEVICES=0 python train_qcnet.py \
#     --root /data/xiaodliu/av2 \
#     --train_batch_size 6 \
#     --val_batch_size 2 \
#     --test_batch_size 2 \
#     --devices 1 \
#     --dataset argoverse_v2 \
#     --num_historical_steps 50 \
#     --num_future_steps 60 \
#     --num_recurrent_steps 3 \
#     --pl2pl_radius 150 \
#     --time_span 10 \
#     --pl2a_radius 50 \
#     --a2a_radius 50 \
#     --num_t2m_steps 30 \
#     --pl2m_radius 150 \
#     --a2m_radius 150 \
#     --sample_interval 1 \
#     --exp_name av2_train_100p_no_map \
#     --no_map \
#     --max_epochs 32 \
#     --submission_dir submission/av2 \
#     --submission_file_name 100p_no_map_e32_bs6 


CUDA_VISIBLE_DEVICES=0 python train_qcnet.py \
    --root ~/data/av1 \
    --train_batch_size 8 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --devices 1 \
    --dataset argoverse_v1 \
    --num_historical_steps 20 \
    --num_future_steps 30 \
    --num_recurrent_steps 3 \
    --pl2pl_radius 150 \
    --time_span 10 \
    --pl2a_radius 50 \
    --a2a_radius 50 \
    --num_t2m_steps 30 \
    --pl2m_radius 150 \
    --a2m_radius 150 \
    --sample_interval 1 \
    --exp_name av1_100p_no_map_mode32 \
    --num_workers 8 \
    --no_map \
    --data_to_ram \
    --num_modes 32 \
    --max_epochs 32 \
    --submission_dir submission/av1 \
    --submission_file_name 10p_no_map_e32_bs8_m32
