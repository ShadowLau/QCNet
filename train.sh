# python train_qcnet.py \
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
#     --sample_interval 5 \
#     --exp_name train_20p_no_map \
#     --no_map


python train_qcnet.py \
    --root /data/xiaodliu/av1 \
    --train_batch_size 6 \
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
    --sample_interval 20 \
    --exp_name av1_5p_no_map \
    --no_map
