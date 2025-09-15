#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# 统一训练的FSDP脚本
# 复用pretrain_unified_navit.py的成熟FSDP框架来训练统一生成模型
# 支持预训练和微调两种模式

# ====== 配置参数 ======
# 请根据你的环境修改以下变量

# 分布式训练配置
export num_nodes=${num_nodes:-1}
export node_rank=${node_rank:-0}
export master_addr=${master_addr:-"localhost"}
export master_port=${master_port:-"12345"}
export nproc_per_node=${nproc_per_node:-8}

# 模型路径配置
# export model_path=${model_path:-"hf/BAGEL-7B-MoT"}
export llm_path=${llm_path:-"/workspace/bagel/models/Qwen2.5-0.5B-Instruct"}
export vae_path=${vae_path:-"/workspace/bagel/models/flux/ae.safetensors"}
export vit_path=${vit_path:-"/workspace/bagel/models/siglip-so400m-14-980-flash-attn2-navit/"}

# 数据路径配置
export train_data_path=${train_data_path:-"data/unified_train.jsonl"}
# 只有在val_data_path未设置时才使用默认值，如果设置为空字符串则保持为空
if [ -z "${val_data_path+x}" ]; then
    export val_data_path="data/unified_val.jsonl"
fi

# 输出路径配置
export output_path=${output_path:-"results/unified_training"}
export ckpt_path=${ckpt_path:-"results/unified_training/checkpoints"}

# 训练超参数
export batch_size=${batch_size:-1}
export gradient_accumulation_steps=${gradient_accumulation_steps:-8}
export total_steps=${total_steps:-100000}
export learning_rate=${learning_rate:-1e-5}
export warmup_steps=${warmup_steps:-2000}
export save_every=${save_every:-2000}
export log_every=${log_every:-10}

# W&B配置
export wandb_project=${wandb_project:-"bagel-unified"}
export wandb_name=${wandb_name:-"unified-fsdp-run"}
export wandb_runid=${wandb_runid:-"0"}
export wandb_offline=${wandb_offline:-"False"}

# 检查点配置
export resume_from=${resume_from:-""}
export auto_resume=${auto_resume:-"False"}

# Fine-tuning配置（如果需要的话）
export is_finetune=${is_finetune:-"False"}
export finetune_from_hf=${finetune_from_hf:-"False"}
export resume_model_only=${resume_model_only:-"False"}
export finetune_from_ema=${finetune_from_ema:-"False"}

# FSDP配置
export sharding_strategy=${sharding_strategy:-"HYBRID_SHARD"}
export num_shard=${num_shard:-8}
export cpu_offload=${cpu_offload:-"False"}

# 模型配置
export max_latent_size=${max_latent_size:-64}
export max_sequence_length=${max_sequence_length:-2048}
export max_image_tokens=${max_image_tokens:-1024}

# 冻结配置
export freeze_vae=${freeze_vae:-"True"}
export freeze_llm=${freeze_llm:-"False"}
export freeze_vit=${freeze_vit:-"False"}

# 模型结构配置
export layer_module=${layer_module:-"Qwen2MoTDecoderLayer"}

# 损失权重
export text_loss_weight=${text_loss_weight:-1.0}
export image_loss_weight=${image_loss_weight:-1.0}

# 内存管理配置
export clear_cache=${clear_cache:-"True"}

# ====== 启动训练 ======
echo "====== 统一训练FSDP配置 ======"
echo "节点数量: $num_nodes"
echo "当前节点rank: $node_rank"
echo "每节点GPU数: $nproc_per_node"
echo "主节点地址: $master_addr:$master_port"
echo "模型路径: $model_path"
echo "训练数据: $train_data_path"
echo "验证数据: $val_data_path"
echo "输出路径: $output_path"
echo "检查点路径: $ckpt_path"
echo "批次大小: $batch_size"
echo "梯度累积步数: $gradient_accumulation_steps"
echo "总训练步数: $total_steps"
echo "学习率: $learning_rate"
echo "FSDP策略: $sharding_strategy"
echo "=============================="

# 创建输出目录
mkdir -p "$output_path"
mkdir -p "$ckpt_path"

# 构建resume参数
resume_args=""
if [ -n "$resume_from" ]; then
    resume_args="--resume_from $resume_from"
fi

# 构建验证数据参数
val_args=""
if [ -n "$val_data_path" ] && [ "$val_data_path" != "" ]; then
    val_args="--val_data_path $val_data_path"
fi

# 统一使用unified_fsdp_trainer.py进行训练
training_script="train/unified_fsdp_trainer.py"

if [ "$is_finetune" = "True" ]; then
    echo "====== 使用unified_fsdp_trainer.py进行Fine-tuning ======"
    torchrun \
        --nnodes=$num_nodes \
        --node_rank=$node_rank \
        --nproc_per_node=$nproc_per_node \
        --master_addr=$master_addr \
        --master_port=$master_port \
        $training_script \
        \
        --model_path "$model_path" \
        \
        --train_data_path "$train_data_path" \
        $val_args \
        --max_sequence_length $max_sequence_length \
        --max_image_tokens $max_image_tokens \
        \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --total_steps $total_steps \
        --lr $learning_rate \
        --ema ${ema:-"0.999"} \
        --warmup_steps $warmup_steps \
        --save_every $save_every \
        --log_every $log_every \
        \
        --text_loss_weight $text_loss_weight \
        --image_loss_weight $image_loss_weight \
        \
        --results_dir "$output_path" \
        --checkpoint_dir "$ckpt_path" \
        \
        --wandb_project "$wandb_project" \
        --wandb_name "$wandb_name" \
        --wandb_runid "$wandb_runid" \
        --wandb_offline $wandb_offline \
        \
        --auto_resume $auto_resume \
        $resume_args \
        \
        --sharding_strategy "$sharding_strategy" \
        --num_shard $num_shard \
        --cpu_offload $cpu_offload \
        \
        --max_latent_size $max_latent_size \
        --freeze_vae $freeze_vae \
        --freeze_llm $freeze_llm \
        --freeze_vit $freeze_vit \
        \
        --layer_module "$layer_module" \
        --finetune_from_hf $finetune_from_hf \
        --resume_model_only $resume_model_only \
        --finetune_from_ema $finetune_from_ema \
        --copy_init_moe True \
        --clear_cache $clear_cache \
        \
        --num_workers 1

else
    echo "====== 使用unified_fsdp_trainer.py进行预训练 ======"
    torchrun \
        --nnodes=$num_nodes \
        --node_rank=$node_rank \
        --nproc_per_node=$nproc_per_node \
        --master_addr=$master_addr \
        --master_port=$master_port \
        $training_script \
        \
        --model_path "$model_path" \
        --llm_path "$llm_path" \
        --vae_path "$vae_path" \
        --vit_path "$vit_path" \
        \
        --train_data_path "$train_data_path" \
        $val_args \
        --max_sequence_length $max_sequence_length \
        --max_image_tokens $max_image_tokens \
        \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --total_steps $total_steps \
        --lr $learning_rate \
        --ema ${ema:-"0.999"} \
        --warmup_steps $warmup_steps \
        --save_every $save_every \
        --log_every $log_every \
        \
        --text_loss_weight $text_loss_weight \
        --image_loss_weight $image_loss_weight \
        \
        --results_dir "$output_path" \
        --checkpoint_dir "$ckpt_path" \
        \
        --wandb_project "$wandb_project" \
        --wandb_name "$wandb_name" \
        --wandb_runid "$wandb_runid" \
        --wandb_offline $wandb_offline \
        \
        --auto_resume $auto_resume \
        $resume_args \
        \
        --sharding_strategy "$sharding_strategy" \
        --num_shard $num_shard \
        --cpu_offload $cpu_offload \
        \
        --max_latent_size $max_latent_size \
        --freeze_vae $freeze_vae \
        --freeze_llm $freeze_llm \
        --freeze_vit $freeze_vit \
        \
        --layer_module "$layer_module" \
        --llm_qk_norm True \
        --visual_gen True \
        --visual_und True \
        --finetune_from_hf $finetune_from_hf \
        --copy_init_moe True \
        --clear_cache $clear_cache \
        \
        --num_workers 1

fi

# 启动分布式训练



