
# CUDA_VISIBLE_DEVICES=1 python val.py \
#     --model QCNet \
#     --root /data/xiaodliu/av2 \
#     --ckpt_path exps/train_20p_no_map/lightning_logs/version_0/checkpoints/epoch=60-step=406504.ckpt


CUDA_VISIBLE_DEVICES=1 python val.py \
    --model QCNet \
    --root /data/xiaodliu/av1 \
    --ckpt_path exps/av1_10p_no_map_mode32/lightning_logs/version_0/checkpoints/epoch=31-step=82400.ckpt
