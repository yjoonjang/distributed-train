export WANDB_PROJECT="llama-3-Korean-Bllossom-8B"
export WANDB_NAME="llama-3-Korean-Bllossom-8B-5ep"
EPOCH=5
LR=2e-05
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 run/train.py \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --epoch 5 \
    --lr 2e-5 \
    --warmup_steps 20 \
    --deepspeed ds_config/stage2_fp16.json