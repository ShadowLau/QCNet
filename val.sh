
CUDA_VISIBLE_DEVICES=1 python val.py \
    --model QCNet \
    --root /data/xiaodliu/av2 \
    --ckpt_path exps/train_20p_no_map/lightning_logs/version_0/checkpoints/epoch=60-step=406504.ckpt
