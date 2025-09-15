#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# 统一模型Fine-tuning示例脚本
# 基于用户提供的命令格式，复用train_unified_fsdp.sh的FSDP框架

echo "====== 统一模型Fine-tuning示例 ======"

# GPU设置 - 明确指定使用两张GPU
export CUDA_VISIBLE_DEVICES=0,1,2  # 使用GPU 0和1

# PyTorch内存管理优化 - 移除不支持的选项
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:8

# 添加CUDA设备绑定配置
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# 分布式训练配置
export num_nodes=1
export node_rank=0
export master_addr="localhost"
export master_port=12345
export nproc_per_node=3  # 根据你的GPU数量调整

# 模型路径配置 - 请根据你的实际路径修改
export model_path="/workspace/models/BAGEL-7B-MoT"  # 预训练模型路径
export resume_from="$model_path"    # 从预训练模型开始fine-tune

# 数据配置 - 使用train_data_path格式（你之前的方式）
export train_data_path="/workspace/bagel/dataset/demo/demo_sample/anno.json"
export val_data_path=""  # 可以设置验证数据路径，或留空

# Fine-tuning特定配置
export is_finetune="True"
export finetune_from_hf="True"  # 改为True，表示从已保存的BAGEL模型进行finetune
export auto_resume="True"
export resume_model_only="True"
export finetune_from_ema="True"
export ema=0.98  # 从默认的0.9999降低到0.999，提高学习速度
export freeze_vit="True"
export freeze_llm="False"
export freeze_vae="True"


# 训练超参数
export learning_rate=2e-5
export log_every=1
export layer_module="Qwen2MoTDecoderLayer"
export max_latent_size=64
export expected_num_tokens=1024
export max_num_tokens=1152
export max_num_tokens_per_sample=1024

# EMA设置 - 修复过大的EMA值问题

# 其他训练配置 - 优化显存使用
export batch_size=1
export gradient_accumulation_steps=4  # 减少梯度累积步数
export total_steps=100000
export warmup_steps=2000
export save_every=2000

# 内存管理配置 - 防止保存checkpoint时内存不足
export clear_cache="True"  # 在保存checkpoint前后清理CUDA缓存

# FSDP配置 - 激进显存优化
export sharding_strategy="FULL_SHARD"  # 使用完全分片以最大化显存节省 HYBRID_SHARD
export num_shard=3  # 修改为实际GPU数量
export num_replicate=1  # 复制数量
export cpu_offload="False"

# 输出路径
export output_path="results/unified_finetune_$(date +%Y%m%d_%H%M%S)"
export ckpt_path="$output_path/checkpoints"

# W&B配置
export wandb_project="bagel-unified-finetune"
export wandb_name="unified-finetune-$(date +%Y%m%d_%H%M%S)"
export wandb_offline="False"

echo "Fine-tuning配置："
echo "  GPU设置: $CUDA_VISIBLE_DEVICES"
echo "  GPU数量: $nproc_per_node"
echo "  训练程序: unified_fsdp_trainer.py"
echo "  模型路径: $model_path"
echo "  训练数据: $train_data_path"
echo "  学习率: $learning_rate"
echo "  批次大小: $batch_size"
echo "  总训练步数: $total_steps"
echo "  输出路径: $output_path"
echo "  从HF加载: $finetune_from_hf"
echo "=============================="

# 调用统一训练脚本
bash scripts/train_unified_fsdp.sh