#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练的FSDP适配器

这个模块将你的UnifiedGenerationDataset适配到pretrain_unified_navit.py的FSDP训练框架中，
实现对统一生成训练的完整支持。

主要功能：
1. 将UnifiedGenerationDataset包装成兼容PackedDataset接口的数据加载器
2. 添加统一训练的参数配置
3. 保持FSDP、EMA、检查点等所有成熟功能
"""

import functools
import os
import sys
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from datetime import timedelta

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)
from training.unified_data_processor import UnifiedGenerationDataset
from data.transforms import ImageTransform


@dataclass
class ModelArguments:
    """模型相关参数 - 复用原有配置"""
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )


@dataclass
class UnifiedDataArguments:
    """统一训练数据参数"""
    train_data_path: str = field(
        default="data/unified_train.jsonl",
        metadata={"help": "JSONL file containing unified training examples."}
    )
    val_data_path: str = field(
        default=None,
        metadata={"help": "JSONL file containing validation examples."}
    )
    max_sequence_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for text."}
    )
    max_image_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum tokens per image."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    """训练参数 - 在原有基础上添加统一训练相关配置"""
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel-unified",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="unified-run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: int = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    
    # --- 统一训练损失权重 ---
    text_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the text cross-entropy loss term."}
    )
    image_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image MSE loss term."}
    )
    
    # --- 统一训练特定参数 ---
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for unified training."}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of steps to accumulate gradients before update."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don't fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    
    # --- 内存管理配置 ---
    clear_cache: bool = field(
        default=True,
        metadata={"help": "Clear CUDA cache before saving checkpoints to prevent out of memory errors."}
    )


class UnifiedDatasetWrapper:
    """
    将UnifiedGenerationDataset包装成兼容PackedDataset接口的数据加载器
    这样可以在不修改原有FSDP训练循环的情况下使用统一训练数据
    """
    
    def __init__(
        self,
        dataset: UnifiedGenerationDataset,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_step = 0
        
    def __iter__(self):
        # 无限循环数据加载器，支持多epoch训练
        while True:
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # 统一数据集通常比较复杂，使用单进程
                pin_memory=True,
                drop_last=False,  # 改为False确保所有数据都被使用
            )
            
            for batch in dataloader:
                # 将批次数据转换为兼容原训练循环的格式
                converted_batch = self._convert_batch_format(batch)
                yield converted_batch
                self.current_step += 1
    
    def _convert_batch_format(self, batch):
        """
        将UnifiedGenerationDataset的批次格式转换为兼容原训练循环的格式
        """
        device = torch.cuda.current_device()
        
        # 由于DataLoader可能添加了batch维度，我们需要移除它
        # 因为每个样本已经是完整的打包序列
        def remove_batch_dim(tensor_or_list):
            if isinstance(tensor_or_list, torch.Tensor):
                if tensor_or_list.dim() > 1:
                    return tensor_or_list.squeeze(0)  # 移除第一个维度
                return tensor_or_list
            elif isinstance(tensor_or_list, list) and len(tensor_or_list) == 1:
                return tensor_or_list[0]  # 如果是包含一个元素的列表，返回该元素
            return tensor_or_list
        
        # UnifiedGenerationDataset的输出格式处理
        # 将packed_text_ids映射为input_ids和labels
        packed_text_ids = batch.get('packed_text_ids')
        if packed_text_ids is None:
            raise ValueError("batch中缺少 'packed_text_ids' 字段")
        
        # 移除batch维度
        packed_text_ids = remove_batch_dim(packed_text_ids)
        input_ids = packed_text_ids.to(device)
        
        # 获取真实的序列长度（来自数据集）
        real_seq_len = remove_batch_dim(batch.get('sequence_length', input_ids.shape[0]))
        if isinstance(real_seq_len, torch.Tensor):
            real_seq_len = real_seq_len.item()  # 转换为标量
        text_seq_len = input_ids.shape[0]  # 文本tokens的数量
        
        # 调试信息
        if dist.get_rank() == 0:
            print(f"Debug: real_seq_len={real_seq_len}, text_seq_len={text_seq_len}")
            if 'packed_text_indexes' in batch:
                text_indexes = remove_batch_dim(batch['packed_text_indexes'])
                # print(f"Debug: text_indexes range=[{text_indexes.min()}, {text_indexes.max()}]")
                # print(f"Debug: text_indexes shape={text_indexes.shape}")
        
        # 处理注意力掩码
        if 'nested_attention_masks' in batch and batch['nested_attention_masks']:
            # 使用数据集提供的注意力掩码
            nested_masks = remove_batch_dim(batch['nested_attention_masks'])
            if isinstance(nested_masks, list):
                attention_mask = nested_masks[0].to(device)
            else:
                attention_mask = nested_masks.to(device)
        else:
            # 创建默认的因果注意力掩码（基于真实序列长度）
            attention_mask = torch.tril(torch.ones(real_seq_len, real_seq_len, device=device))
        
        # 标签与input_ids相同，但可能需要移位
        labels = input_ids.clone()
        
        # 验证索引的有效性
        if 'packed_text_indexes' in batch:
            text_indexes = remove_batch_dim(batch['packed_text_indexes'])
            if text_indexes.max() >= real_seq_len:
                print(f"Error: text_indexes超出范围! max_index={text_indexes.max()}, seq_len={real_seq_len}")
                # 过滤掉无效的索引
                valid_mask = text_indexes < real_seq_len
                if torch.any(valid_mask):
                    valid_text_indexes = text_indexes[valid_mask]
                    valid_text_ids = input_ids[valid_mask]
                    print(f"Warning: 过滤后的文本tokens: {len(valid_text_ids)}/{len(input_ids)}")
                    
                    # 更新input_ids和indexes
                    input_ids = valid_text_ids
                    labels = input_ids.clone()
                    # 使用过滤后的索引更新batch
                    batch['packed_text_indexes'] = valid_text_indexes
                else:
                    print(f"Error: 所有text_indexes都无效，无法继续处理")
                    raise ValueError("所有文本索引都超出序列范围")
        
        # 构建兼容Bagel模型的数据字典
        data_dict = {
            'sequence_length': real_seq_len,
            'packed_text_ids': input_ids,
            'packed_text_indexes': remove_batch_dim(batch.get('packed_text_indexes', torch.arange(text_seq_len, device=device))),
            'packed_position_ids': remove_batch_dim(batch.get('packed_position_ids', torch.arange(text_seq_len, device=device))),
            'sample_lens': remove_batch_dim(batch.get('sample_lens', [real_seq_len])),
            'split_lens': remove_batch_dim(batch.get('split_lens', [real_seq_len])),
            'attn_modes': remove_batch_dim(batch.get('attn_modes', ['causal'])),
            'nested_attention_masks': [attention_mask],
            'packed_label_ids': labels,  # 用于计算文本损失
        }
        
        # 如果有VIT图像数据
        if 'packed_vit_tokens' in batch and 'vit_token_seqlens' in batch:
            # 检查vit_token_seqlens是否为空或包含无效值
            vit_seqlens = remove_batch_dim(batch['vit_token_seqlens'])
            if len(vit_seqlens) > 0 and torch.all(vit_seqlens >= 0):
                # 验证VIT索引的有效性
                vit_indexes = remove_batch_dim(batch['packed_vit_token_indexes'])
                if vit_indexes.max() >= real_seq_len:
                    print(f"Warning: VIT索引超出范围! max_index={vit_indexes.max()}, seq_len={real_seq_len}")
                    # 过滤掉无效的索引
                    valid_vit_mask = vit_indexes < real_seq_len
                    if torch.any(valid_vit_mask):
                        valid_vit_indexes = vit_indexes[valid_vit_mask]
                        valid_vit_tokens = remove_batch_dim(batch['packed_vit_tokens'])[valid_vit_mask]
                        valid_vit_positions = remove_batch_dim(batch['packed_vit_position_ids'])[valid_vit_mask]
                        
                        data_dict.update({
                            'packed_vit_tokens': valid_vit_tokens.to(device),
                            'packed_vit_token_indexes': valid_vit_indexes.to(device),
                            'packed_vit_position_ids': valid_vit_positions.to(device),
                            'vit_token_seqlens': vit_seqlens.to(device),
                        })
                    else:
                        print(f"Warning: 所有VIT索引都无效，跳过VIT处理")
                else:
                    data_dict.update({
                        'packed_vit_tokens': remove_batch_dim(batch['packed_vit_tokens']).to(device),
                        'packed_vit_token_indexes': vit_indexes.to(device),
                        'packed_vit_position_ids': remove_batch_dim(batch['packed_vit_position_ids']).to(device),
                        'vit_token_seqlens': vit_seqlens.to(device),
                    })
            else:
                # VIT数据无效，跳过VIT相关字段
                if dist.get_rank() == 0:
                    print(f"Warning: Invalid or empty vit_token_seqlens: {vit_seqlens}")
        elif 'packed_vit_tokens' in batch:
            # 有VIT tokens但没有seqlens，这是数据不一致的情况
            if dist.get_rank() == 0:
                print("Warning: Found packed_vit_tokens but missing vit_token_seqlens")
        
        # 如果有VAE图像数据
        if 'padded_vae_images' in batch:
            # 修复图像张量形状：移除batch维度
            padded_images = remove_batch_dim(batch['padded_vae_images'])
            
            vae_token_indexes = remove_batch_dim(batch['packed_vae_token_indexes'])
            vae_position_ids = remove_batch_dim(batch['packed_vae_position_ids'])
            vae_latent_shapes = remove_batch_dim(batch['patchified_vae_latent_shapes'])
            
            data_dict.update({
                'padded_images': padded_images.to(device),  # 这个将在main中转换为padded_latent
                'packed_vae_token_indexes': vae_token_indexes.to(device),
                'packed_latent_position_ids': vae_position_ids.to(device), 
                'patchified_vae_latent_shapes': vae_latent_shapes,
                'packed_timesteps': remove_batch_dim(batch.get('packed_timesteps', torch.zeros(len(vae_token_indexes), device=device))),
            })
        
        if 'image_attention_mask' in batch:
            data_dict['image_attention_mask'] = remove_batch_dim(batch['image_attention_mask']).to(device)
        
        # 添加损失计算需要的索引信息
        if 'ce_loss_indexes' in batch:
            data_dict['ce_loss_indexes'] = remove_batch_dim(batch['ce_loss_indexes']).to(device)
        else:
            # 创建文本损失索引：基于文本在整个序列中的位置
            ce_mask = torch.zeros(real_seq_len, dtype=torch.bool, device=device)
            if 'packed_text_indexes' in data_dict:
                # 获取文本token的位置索引
                text_positions = data_dict['packed_text_indexes']
                # 确保索引在有效范围内
                valid_text_positions = text_positions[text_positions < real_seq_len]
                if len(valid_text_positions) > 0:
                    ce_mask[valid_text_positions] = True
            data_dict['ce_loss_indexes'] = ce_mask
        
        if 'mse_loss_indexes' in batch:
            data_dict['mse_loss_indexes'] = remove_batch_dim(batch['mse_loss_indexes']).to(device)
        elif 'packed_vae_token_indexes' in batch:
            # 创建图像损失索引：对VAE token位置计算MSE损失
            vae_indexes = remove_batch_dim(batch['packed_vae_token_indexes'])
            
            # 调试信息：检查索引范围
            if dist.get_rank() == 0:
                # print(f"Debug: real_seq_len={real_seq_len}, vae_indexes.max()={vae_indexes.max()}, vae_indexes.min()={vae_indexes.min()}")
                # print(f"Debug: vae_indexes shape={vae_indexes.shape}, real_seq_len={real_seq_len}")
                pass
            # 确保VAE索引不超出序列长度
            valid_vae_indexes = vae_indexes[vae_indexes < real_seq_len]
            if len(valid_vae_indexes) < len(vae_indexes):
                print(f"Warning: Some VAE indexes ({len(vae_indexes) - len(valid_vae_indexes)}) exceed sequence length")
            
            mse_mask = torch.zeros(real_seq_len, dtype=torch.bool, device=device)
            if len(valid_vae_indexes) > 0:
                mse_mask[valid_vae_indexes] = True
            data_dict['mse_loss_indexes'] = mse_mask
        else:
            # 没有图像数据时，设置为全False的mask
            data_dict['mse_loss_indexes'] = torch.zeros(real_seq_len, dtype=torch.bool, device=device)
        
        # 修复 sample_lens - 使用数据集提供的值
        data_dict['sample_lens'] = batch.get('sample_lens', [real_seq_len])
        
        # 添加虚拟的数据索引信息（用于检查点恢复）
        data_dict['batch_data_indexes'] = [{
            'dataset_name': 'unified_dataset',
            'worker_id': 0,
            'data_indexes': list(range(len(input_ids))),
        }]
        
        
        return self._to_cuda_dict(data_dict, device)
    
    def _to_cuda_dict(self, data_dict, device):
        """将数据字典转换为CUDA字典类"""
        class CudaDict:
            def __init__(self, data_dict, device):
                self._data = data_dict
                self._device = device
            
            def cuda(self, device):
                return self
            
            def to_dict(self):
                return self._data
            
            def pop(self, key, default=None):
                return self._data.pop(key, default)
            
            def __getitem__(self, key):
                return self._data[key]
            
            def __contains__(self, key):
                return key in self._data
            
            def keys(self):
                return self._data.keys()
        
        return CudaDict(data_dict, device)


def create_unified_dataset(data_args, model_args, tokenizer, new_token_ids):
    """创建统一训练数据集"""
    # 创建图像变换器
    vae_transform = ImageTransform(1024, 512, 16)  # VAE需要的变换
    vit_transform = ImageTransform(980, 224, 14)   # ViT需要的变换
    
    # 创建训练数据集
    dataset = UnifiedGenerationDataset(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        max_sequence_length=data_args.max_sequence_length,
        max_image_tokens=data_args.max_image_tokens,
    )
    
    # 验证数据集（可选）
    val_dataset = None
    if data_args.val_data_path:
        val_dataset = UnifiedGenerationDataset(
            data_path=data_args.val_data_path,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
            max_sequence_length=data_args.max_sequence_length,
            max_image_tokens=data_args.max_image_tokens,
        )
    
    return dataset, val_dataset


def main():
    """主训练函数"""
    assert torch.cuda.is_available(), "CUDA不可用"
    
    # 获取本地rank，确保每个进程使用不同的GPU（在init_process_group之前）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device_count = torch.cuda.device_count()
    
    print(f"Rank {local_rank}: CUDA设备数量: {device_count}, World size: {world_size}")
    
    # 确保有足够的GPU
    if device_count < world_size:
        raise RuntimeError(f"可用GPU数量({device_count})少于进程数量({world_size})")
    
    device = local_rank % device_count
    print(f"Rank {local_rank}: 使用设备 {device}")
    
    # 设置CUDA设备（在init_process_group之前）
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    
    # 设置分布式训练超时时间，指定设备
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    
    # 使用指定设备进行barrier同步
    try:
        dist.barrier(device_ids=[device])
        print(f"Rank {local_rank}: barrier同步成功")
    except Exception as e:
        print(f"Rank {local_rank}: barrier同步失败: {e}")
        raise
    parser = HfArgumentParser((ModelArguments, UnifiedDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=300)  # 增加超时时间到5分钟
        )
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:  
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # === 关键修改：使用统一数据集 ===
    # 创建统一训练数据集
    unified_dataset, val_dataset = create_unified_dataset(data_args, model_args, tokenizer, new_token_ids)
    
    if dist.get_rank() == 0:
        logger.info(f"训练数据路径: {data_args.train_data_path}")
        if val_dataset is not None:
            logger.info(f"验证数据路径: {data_args.val_data_path}")
        else:
            logger.info("未使用验证数据集")
    
    # 使用包装器将其适配到原有训练循环
    dataset_wrapper = UnifiedDatasetWrapper(
        unified_dataset,
        batch_size=training_args.batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    ema_model.eval()

    # === 修改后的训练循环 ===
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    
    gradient_accumulation_counter = 0
    
    for curr_step, data in enumerate(dataset_wrapper, start=train_step):
        if curr_step >= training_args.total_steps:
            break
            
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen and 'padded_images' in data:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
            loss_dict = fsdp_model(**data)

        # 统一训练的损失计算
        loss = 0
        
        # 文本损失
        ce = loss_dict.get("ce")
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data.get('ce_loss_indexes', [])), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if total_ce_tokens > 0:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
                loss_dict["ce"] = ce.detach()
                loss = loss + ce * training_args.text_loss_weight
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        # 图像损失
        if training_args.visual_gen:
            mse = loss_dict.get("mse")
            if mse is not None:
                total_mse_tokens = torch.tensor(len(data.get('mse_loss_indexes', [])), device=device)
                dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
                if total_mse_tokens > 0:
                    mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
                    loss_dict["mse"] = mse.detach()
                    loss = loss + mse * training_args.image_loss_weight
            else:
                loss_dict["mse"] = torch.tensor(0, device=device)
                total_mse_tokens = torch.tensor(0, device=device)
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        # 梯度累积
        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()
        gradient_accumulation_counter += 1
        
        # 梯度更新
        if gradient_accumulation_counter >= training_args.gradient_accumulation_steps:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            gradient_accumulation_counter = 0
            
            # 定期清理缓存以防止内存积累（可选，每10个梯度更新步骤）
            if training_args.clear_cache and curr_step % (10 * training_args.gradient_accumulation_steps) == 0:
                torch.cuda.empty_cache()

        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data.get('sample_lens', [1])), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            logger.info(message)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['gradient_accumulation_counter'] = gradient_accumulation_counter

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()

        if data_status is None:
            data_status = {}
        if data_indexes:
            for item in data_indexes:
                if item['dataset_name'] not in data_status.keys():
                    data_status[item['dataset_name']] = {}
                data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            # 在保存checkpoint前清理CUDA缓存以避免内存不足错误
            if training_args.clear_cache:
                torch.cuda.empty_cache()
                if dist.get_rank() == 0:
                    logger.info("Cleared CUDA cache before saving checkpoint")
            
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )
            
            # 保存完成后再次清理缓存
            if training_args.clear_cache:
                torch.cuda.empty_cache()
                if dist.get_rank() == 0:
                    logger.info("Cleared CUDA cache after saving checkpoint")

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
