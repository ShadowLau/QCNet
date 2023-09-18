
CUDA_VISIBLE_DEVICES=1 python test.py \
    --model QCNet \
    --root /data/xiaodliu/av2 \
    --ckpt_path weights/20p_no_map_epoch32_bs16_2gpu.ckpt