
# CUDA_VISIBLE_DEVICES=1 python test.py \
#     --model QCNet \
#     --root /data/xiaodliu/av2 \
#     --ckpt_path weights/20p_no_map_epoch32_bs16_2gpu.ckpt


CUDA_VISIBLE_DEVICES=0 python test.py \
    --model QCNet \
    --root /data/xiaodliu/av1 \
    --ckpt_path exps/av1_10p_no_map_mode32/lightning_logs/version_0/checkpoints/epoch=31-step=82400.ckpt
