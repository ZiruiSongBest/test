#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一图像编辑推理引擎

基于BAGEL模型的图像编辑推理，正确的模型加载方式参考inferencer.py
专注于图像编辑模式的推理，不包含复杂的自回归交错生成逻辑。
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any
from PIL import Image
import numpy as np
import logging
import argparse
from copy import deepcopy

# 添加项目根目录到路径
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# 导入必要的模块
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from inferencer import InterleaveInferencer
from modeling.bagel.qwen2_navit import NaiveCache

# 如果需要使用量化或多GPU
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class UnifiedAutoregressiveInferencer:
    """
    统一自回归推理器 - 真正对应训练时的forward_autoregressive_training逻辑
    
    与训练的主要对应关系：
    1. 统一序列建模：文本token + 特殊token + 图像patches
    2. 逐token自回归生成，包括特殊token预测
    3. patch级别的图像Flow Matching生成
    4. 保持训练推理一致的数据流
    """
    
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.device = next(model.parameters()).device
        
        # 特殊token IDs
        self.start_of_image = new_token_ids.get('start_of_image')
        self.end_of_image = new_token_ids.get('end_of_image')
        self.bos_token_id = new_token_ids.get('bos_token_id')
        self.eos_token_id = new_token_ids.get('eos_token_id')
        
        print(f"🔧 统一自回归推理器初始化完成")
        print(f"   - start_of_image: {self.start_of_image}")
        print(f"   - end_of_image: {self.end_of_image}")
    
    @torch.no_grad()
    def unified_autoregressive_inference(
        self,
        input_text: str,
        input_image: Optional[Image.Image] = None,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        image_shapes: tuple = (1024, 1024),
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
        timestep_shift: float = 3.0,
        **kwargs
    ) -> List[Union[str, Image.Image]]:
        """
        统一的自回归推理，完全对应训练时的逻辑
        
        Args:
            input_text: 输入文本（对应训练时的input_text）
            input_image: 输入图像（对应训练时的input_image）
            max_length: 最大生成长度
            do_sample: 是否采样
            temperature: 温度参数
            image_shapes: 生成图像尺寸
            cfg_text_scale: 文本CFG强度
            cfg_img_scale: 图像CFG强度
            num_timesteps: Flow Matching步数
            timestep_shift: 时间步偏移
            
        Returns:
            生成结果列表（文本和图像交错）
        """
        print(f"🚀 开始统一自回归推理")
        print(f"📝 输入文本: {input_text}")
        print(f"🖼️  输入图像: {input_image.size if input_image else 'None'}")
        
        # 1. 处理输入阶段（对应训练的_process_input_stage）
        input_embeddings, input_sequence_length = self._process_input_stage(
            input_text, input_image
        )
        
        # 2. 统一自回归生成（对应训练的_process_unified_autoregressive_training）
        generated_sequence = self._unified_autoregressive_generation(
            input_embeddings=input_embeddings,
            input_sequence_length=input_sequence_length,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            image_shapes=image_shapes,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            num_timesteps=num_timesteps,
            timestep_shift=timestep_shift,
            **kwargs
        )
        
        # 3. 解析输出序列
        output_results = self._parse_generated_sequence(generated_sequence)
        
        print(f"✅ 统一自回归推理完成，生成了 {len(output_results)} 个结果")
        return output_results
    
    def _process_input_stage(
        self, 
        input_text: str, 
        input_image: Optional[Image.Image]
    ) -> Tuple[torch.Tensor, int]:
        """
        处理输入阶段，对应训练时的_process_input_stage
        
        训练时的顺序：文本在前，图像在后
        
        Returns:
            (input_embeddings, sequence_length)
        """
        sequence_parts = []
        total_length = 0
        
        # 1. 处理输入文本 - 与训练保持一致，文本在前
        text_ids = self.tokenizer.encode(input_text)
        text_embedding = self.model.language_model.model.embed_tokens(
            torch.tensor(text_ids, device=self.device)
        )
        
        sequence_parts.append(text_embedding)
        total_length += len(text_ids)
        
        # 2. 处理输入图像（如果存在）- 图像在文本之后
        if input_image is not None:
            # 转换图像格式
            from data.data_utils import pil_img2rgb, patchify
            
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            # 使用VIT变换处理图像
            image_tensor = self.vit_transform(pil_img2rgb(input_image))
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # 确保数据类型与模型权重一致
            model_dtype = next(self.model.parameters()).dtype
            image_tensor = image_tensor.to(dtype=model_dtype, device=self.device)
            
            # 使用ViT处理输入图像（对应训练逻辑）
            vit_position_ids = self.model.get_flattened_position_ids(
                image_tensor.size(2), image_tensor.size(3),
                self.model.vit_patch_size,
                max_num_patches_per_side=self.model.vit_max_num_patch_per_side
            ).to(self.device)
            
            vit_tokens = patchify(image_tensor.squeeze(0), self.model.vit_patch_size)
            
            # 使用ViT模型处理
            cu_seqlens = torch.tensor([0, vit_tokens.shape[0]], dtype=torch.int32, device=self.device)
            vit_embeddings = self.model.vit_model(
                packed_pixel_values=vit_tokens,
                packed_flattened_position_ids=vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=vit_tokens.shape[0],
            )
            
            # 应用连接器和位置编码
            vit_embeddings = self.model.connector(vit_embeddings)
            vit_pos_emb = self.model.vit_pos_embed(vit_position_ids)
            vit_embeddings = vit_embeddings + vit_pos_emb
            
            sequence_parts.append(vit_embeddings)
            total_length += len(vit_embeddings)
        
        # 3. 构建初始序列（与训练时的顺序一致：text_embedding + vit_embeddings）
        input_embeddings = torch.cat(sequence_parts, dim=0)
        
        print(f"✅ 输入处理完成，序列长度: {total_length}")
        print(f"📄 文本tokens: {len(text_ids)}")
        if input_image is not None:
            print(f"🖼️  图像patches: {len(vit_embeddings)}")
        return input_embeddings, total_length
    
    def _unified_autoregressive_generation(
        self,
        input_embeddings: torch.Tensor,
        input_sequence_length: int,
        max_length: int,
        do_sample: bool,
        temperature: float,
        image_shapes: tuple,
        cfg_text_scale: float,
        cfg_img_scale: float,
        num_timesteps: int,
        timestep_shift: float,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        统一自回归生成，完全对应训练时的逐token处理逻辑
        
        训练时的核心逻辑：
        1. 逐token预测，包括文本token和特殊token（<vision_start>, <vision_end>）
        2. 模型自己决定何时输出<vision_start>开始图像生成
        3. 在<vision_start>和<vision_end>之间，逐patch生成图像
        4. 模型自己决定何时输出<vision_end>结束图像生成
        
        Returns:
            生成序列列表，每个元素包含token类型和内容
        """
        print(f"🎯 开始统一自回归生成，最大长度: {max_length}")
        
        # 当前序列状态
        current_embeddings = input_embeddings.clone()
        generated_sequence = []
        step = 0
        
        # 生成状态跟踪
        in_image_generation = False
        current_image_patches = []
        current_image_shape = None
        patches_generated_in_current_image = 0
        max_patches_per_image = None
        
        # 初始化 past_key_values 和相关参数
        past_key_values = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        
        while step < max_length:
            # 计算当前步骤的KV cache参数
            if step == 0:
                # 第一步：处理完整输入序列，没有past key values
                kv_lens = [0]
                kv_indexes = torch.tensor([], dtype=torch.long, device=self.device)
            else:
                # 后续步骤：有past key values
                kv_lens = [len(current_embeddings) - 1]  # past序列长度
                kv_indexes = torch.arange(len(current_embeddings) - 1, device=self.device)
            # 1. 预测下一个token（可能是文本、特殊token或图像patch）
            if step == 0:
                # 第一步：使用完整输入序列
                query_embeddings = current_embeddings
            else:
                # 后续步骤：只使用最后一个token
                query_embeddings = current_embeddings[-1:, :]
                
            next_token_info = self._predict_next_token_unified(
                query_embeddings, 
                in_image_generation,
                patches_generated_in_current_image,
                max_patches_per_image,
                do_sample, 
                temperature,
                past_key_values,
                kv_lens,
                kv_indexes
            )
            
            token_id = next_token_info.get('token_id')
            token_type = next_token_info.get('token_type')
            
            print(f"第 {step+1} 步: 预测 {token_type}, token_id: {token_id}")
            if token_type == 'special':
                if token_id == self.start_of_image:
                    print(f"   -> 预测到 start_of_image")
                elif token_id == self.end_of_image:
                    print(f"   -> 预测到 end_of_image")
                elif token_id == self.eos_token_id:
                    print(f"   -> 预测到 EOS")
                else:
                    print(f"   -> 预测到其他特殊token: {token_id}")
            
            # 2. 根据预测结果处理
            if token_type == 'text':
                # 普通文本token
                token_embedding = self.model.language_model.model.embed_tokens(
                    torch.tensor([token_id], device=self.device)
                )
                current_embeddings = torch.cat([current_embeddings, token_embedding], dim=0)
                
                generated_sequence.append({
                    'type': 'text_token',
                    'content': token_id
                })
                
            elif token_type == 'special' and token_id == self.start_of_image:
                # 模型预测要开始生成图像
                print(f"🖼️  模型决定开始图像生成")
                in_image_generation = True
                current_image_shape = image_shapes
                patches_generated_in_current_image = 0
                
                # 计算当前图像的最大patch数
                H, W = image_shapes
                h = H // self.model.latent_downsample
                w = W // self.model.latent_downsample
                max_patches_per_image = h * w
                
                # 添加<vision_start> token到序列
                token_embedding = self.model.language_model.model.embed_tokens(
                    torch.tensor([token_id], device=self.device)
                )
                current_embeddings = torch.cat([current_embeddings, token_embedding], dim=0)
                
                generated_sequence.append({
                    'type': 'special_token',
                    'content': token_id,
                    'token_name': 'start_of_image'
                })
                
            elif token_type == 'special' and token_id == self.end_of_image:
                # 模型预测要结束图像生成
                print(f"🖼️  模型决定结束图像生成，已生成 {patches_generated_in_current_image} 个patches")
                
                # 完成图像生成
                if current_image_patches:
                    generated_image = self._finalize_image_generation(
                        current_image_patches, current_image_shape,
                        cfg_text_scale, cfg_img_scale, num_timesteps, timestep_shift
                    )
                    generated_sequence.append({
                        'type': 'image',
                        'content': generated_image
                    })
                    current_image_patches = []
                
                in_image_generation = False
                patches_generated_in_current_image = 0
                max_patches_per_image = None
                
                # 添加<vision_end> token到序列
                token_embedding = self.model.language_model.model.embed_tokens(
                    torch.tensor([token_id], device=self.device)
                )
                current_embeddings = torch.cat([current_embeddings, token_embedding], dim=0)
                
                generated_sequence.append({
                    'type': 'special_token',
                    'content': token_id,
                    'token_name': 'end_of_image'
                })
                
            elif token_type == 'image_patch':
                # 生成图像patch（对应训练时的flow matching）
                patch_data = self._generate_image_patch_unified(
                    current_embeddings, 
                    current_image_shape,
                    patches_generated_in_current_image,
                    cfg_text_scale,
                    cfg_img_scale,
                    past_key_values,
                    kv_lens,
                    kv_indexes
                )
                current_image_patches.append(patch_data)
                patches_generated_in_current_image += 1
                
                # 添加patch embedding到序列
                current_embeddings = torch.cat([
                    current_embeddings, 
                    patch_data['embedding']
                ], dim=0)
                
                generated_sequence.append({
                    'type': 'image_patch',
                    'content': patch_data
                })
                
                # 检查是否已生成所有patches
                if patches_generated_in_current_image >= max_patches_per_image:
                    print(f"📊 已生成所有 {max_patches_per_image} 个patches")
                    # 下一步应该预测<vision_end>
                
            elif token_id == self.eos_token_id:
                # 结束生成
                print(f"🏁 遇到EOS token，停止生成")
                generated_sequence.append({
                    'type': 'special_token',
                    'content': token_id,
                    'token_name': 'eos'
                })
                break
                
            else:
                # 其他特殊token
                token_embedding = self.model.language_model.model.embed_tokens(
                    torch.tensor([token_id], device=self.device)
                )
                current_embeddings = torch.cat([current_embeddings, token_embedding], dim=0)
                
                generated_sequence.append({
                    'type': 'special_token',
                    'content': token_id
                })
            
            step += 1
        
        # 处理未完成的图像生成
        if in_image_generation and current_image_patches:
            print(f"⚠️  生成未完成，强制完成图像生成")
            generated_image = self._finalize_image_generation(
                current_image_patches, current_image_shape,
                cfg_text_scale, cfg_img_scale, num_timesteps, timestep_shift
            )
            generated_sequence.append({
                'type': 'image',
                'content': generated_image
            })
        
        print(f"✅ 自回归生成完成，共生成 {len(generated_sequence)} 个元素")
        return generated_sequence
    
    def _predict_next_token_unified(
        self, 
        current_embeddings: torch.Tensor,
        in_image_generation: bool,
        patches_generated: int,
        max_patches: Optional[int],
        do_sample: bool, 
        temperature: float,
        past_key_values: NaiveCache,
        kv_lens: List[int],
        kv_indexes: torch.Tensor
    ) -> Dict[str, Any]:
        """
        统一的下一个token预测，完全对应训练时的逻辑
        
        关键点：
        1. 如果不在图像生成中，预测文本token或特殊token（包括<vision_start>）
        2. 如果在图像生成中，生成图像patch直到完成，然后预测<vision_end>
        
        Args:
            current_embeddings: 当前序列的embeddings
            in_image_generation: 是否在图像生成过程中
            patches_generated: 当前图像已生成的patch数
            max_patches: 当前图像的最大patch数
            do_sample: 是否采样
            temperature: 采样温度
            
        Returns:
            包含预测结果的字典
        """
        # 构建位置ID
        seq_len = current_embeddings.size(0)
        position_ids = torch.arange(seq_len, device=self.device)
        
        # LLM前向传播
        output = self.model.language_model(
            packed_query_sequence=current_embeddings.unsqueeze(0),
            query_lens=torch.tensor([seq_len], device=self.device),
            packed_query_position_ids=position_ids,
            packed_query_indexes=torch.arange(seq_len, device=self.device),
            past_key_values=past_key_values,
            key_values_lens=torch.tensor(kv_lens, device=self.device),
            packed_key_value_indexes=kv_indexes,
            update_past_key_values=True,
            is_causal=True,
            mode="und"
        )
        
        # 获取最后一个token的隐藏状态
        # 根据qwen2_navit.py的BaseNavitOutputWithPast，需要获取packed_query_sequence
        hidden_states = output.packed_query_sequence
            
        last_hidden_state = hidden_states[0, -1, :]
        
        if not in_image_generation:
            # 不在图像生成中：预测文本token或特殊token
            logits = self.model.language_model.lm_head(last_hidden_state.unsqueeze(0))
            
            if do_sample:
                import torch.nn.functional as F
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs.squeeze(0), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            token_id = next_token.item()
            
            # 判断token类型
            if token_id == self.start_of_image:
                return {
                    'token_id': token_id,
                    'token_type': 'special',
                    'logits': logits,
                    'hidden_state': last_hidden_state
                }
            elif token_id == self.end_of_image:
                # 不应该在非图像生成时预测到end_of_image
                # 但为了鲁棒性还是处理
                return {
                    'token_id': token_id,
                    'token_type': 'special',
                    'logits': logits,
                    'hidden_state': last_hidden_state
                }
            elif token_id == self.eos_token_id:
                return {
                    'token_id': token_id,
                    'token_type': 'special',
                    'logits': logits,
                    'hidden_state': last_hidden_state
                }
            else:
                return {
                    'token_id': token_id,
                    'token_type': 'text',
                    'logits': logits,
                    'hidden_state': last_hidden_state
                }
        else:
            # 在图像生成中
            if patches_generated < max_patches:
                # 还需要生成更多patches
                return {
                    'token_type': 'image_patch',
                    'hidden_state': last_hidden_state
                }
            else:
                # 已生成所有patches，应该预测<vision_end>
                # 强制预测<vision_end>或让模型自己决定
                logits = self.model.language_model.lm_head(last_hidden_state.unsqueeze(0))
                
                # 这里可以选择：
                # 1. 强制输出<vision_end>
                # 2. 让模型自己预测（可能继续生成或结束）
                
                # 方案1：强制输出（更稳定）
                return {
                    'token_id': self.end_of_image,
                    'token_type': 'special',
                    'logits': logits,
                    'hidden_state': last_hidden_state
                }
                
                # 方案2：让模型决定（注释掉的代码）
                # if do_sample:
                #     import torch.nn.functional as F
                #     probs = F.softmax(logits / temperature, dim=-1)
                #     next_token = torch.multinomial(probs.squeeze(0), num_samples=1)
                # else:
                #     next_token = torch.argmax(logits, dim=-1)
                # 
                # token_id = next_token.item()
                # if token_id == self.end_of_image:
                #     return {
                #         'token_id': token_id,
                #         'token_type': 'special',
                #         'logits': logits,
                #         'hidden_state': last_hidden_state
                #     }
                # else:
                #     # 继续生成patch（可能超过预期数量）
                #     return {
                #         'token_type': 'image_patch',
                #         'hidden_state': last_hidden_state
                #     }
    
    def _generate_image_patch_unified(
        self, 
        current_embeddings: torch.Tensor, 
        image_shape: tuple, 
        patch_index: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        past_key_values: NaiveCache,
        kv_lens: List[int],
        kv_indexes: torch.Tensor,
        num_flow_steps: int = 10
    ) -> Dict[str, Any]:
        """
        生成单个图像patch，使用Flow Matching，对应训练时的逻辑
        
        与训练的对应关系：
        1. 训练时：给定noisy patch，预测velocity (noise - clean)
        2. 推理时：从纯噪声开始，通过预测的velocity逐步去噪
        
        Args:
            current_embeddings: 当前序列的embeddings
            image_shape: 图像尺寸
            patch_index: 当前patch的索引
            cfg_text_scale: 文本CFG强度
            cfg_img_scale: 图像CFG强度  
            num_flow_steps: Flow Matching的去噪步数
            
        Returns:
            包含patch信息的字典
        """
        # 计算patch的位置和尺寸信息
        H, W = image_shape
        h = H // self.model.latent_downsample
        w = W // self.model.latent_downsample
        
        # 生成位置ID
        patch_position_ids = self.model.get_flattened_position_ids(
            h * self.model.latent_downsample, w * self.model.latent_downsample,
            self.model.latent_downsample,
            max_num_patches_per_side=self.model.max_latent_size
        ).to(self.device)
        
        current_patch_pos_id = patch_position_ids[patch_index:patch_index+1]
        
        # 初始化：从纯噪声开始
        patch_dim = self.model.latent_patch_size ** 2 * self.model.latent_channel
        x_t = torch.randn(1, patch_dim, device=self.device)  # 纯噪声
        
        # Flow Matching去噪过程
        timesteps = torch.linspace(1, 0, num_flow_steps, device=self.device)
        for t in timesteps:
            # 准备时间步
            timestep = torch.tensor([t], device=self.device)
            timestep_processed = torch.sigmoid(timestep)
            timestep_processed = self.model.timestep_shift * timestep_processed / (1 + (self.model.timestep_shift - 1) * timestep_processed)
            
            # 构建当前patch的embedding
            timestep_embed = self.model.time_embedder(timestep)
            latent_pos_embed = self.model.latent_pos_embed(current_patch_pos_id)
            patch_embedding = self.model.vae2llm(x_t) + timestep_embed + latent_pos_embed
            
            # 将patch embedding添加到当前序列末尾（临时）
            temp_embeddings = torch.cat([current_embeddings, patch_embedding.squeeze(0)], dim=0)
            
            # 构建位置ID
            seq_len = temp_embeddings.size(0)
            position_ids = torch.arange(seq_len, device=self.device)
            
            # LLM前向传播
            output = self.model.language_model(
                packed_query_sequence=temp_embeddings.unsqueeze(0),
                query_lens=torch.tensor([seq_len], device=self.device),
                packed_query_position_ids=position_ids,
                packed_query_indexes=torch.arange(seq_len, device=self.device),
                past_key_values=past_key_values,
                key_values_lens=torch.tensor(kv_lens, device=self.device),
                packed_key_value_indexes=kv_indexes,
                update_past_key_values=True,
                is_causal=True,
                mode="und"
            )
            
            # 获取patch位置的隐藏状态
            # 处理不同的模型输出格式
            if hasattr(output, 'packed_query_sequence'):
                # BaseNavitOutputWithPast 格式
                hidden_states = output.packed_query_sequence
            elif hasattr(output, 'last_hidden_state'):
                # 如果是一个包含 last_hidden_state 属性的对象
                hidden_states = output.last_hidden_state
            elif isinstance(output, tuple):
                # 如果是元组，第一个元素是 hidden states
                hidden_states = output[0]
            else:
                # 否则直接使用
                hidden_states = output
            
            patch_hidden_state = hidden_states[0, -1, :]
            
            # 预测velocity
            v_pred = self.model.llm2vae(patch_hidden_state.unsqueeze(0))
            
            # CFG（如果需要）
            if cfg_text_scale > 1.0 or cfg_img_scale > 1.0:
                # TODO: 实现CFG逻辑
                pass
            
            # 更新x_t（向clean data方向移动）
            dt = timesteps[1] - timesteps[0] if len(timesteps) > 1 else t
            x_t = x_t - v_pred * dt  # velocity指向noise到data的方向
        
        # 最终的clean latent
        clean_latent = x_t
        
        # 准备最终的embedding（使用clean latent，timestep=0）
        timestep_final = torch.zeros(1, device=self.device)
        timestep_embed_final = self.model.time_embedder(timestep_final)
        latent_pos_embed_final = self.model.latent_pos_embed(current_patch_pos_id)
        patch_embedding_final = self.model.vae2llm(clean_latent) + timestep_embed_final + latent_pos_embed_final
        
        return {
            'embedding': patch_embedding_final.squeeze(0),  # 移除batch维度
            'latent': clean_latent.squeeze(0),
            'position_id': current_patch_pos_id,
            'timestep': timestep_final,
            'patch_index': patch_index
        }
    
    def _finalize_image_generation(
        self, 
        image_patches: List[Dict[str, Any]], 
        image_shape: tuple,
        cfg_text_scale: float,
        cfg_img_scale: float,
        num_timesteps: int,
        timestep_shift: float
    ) -> Image.Image:
        """
        完成图像生成，将patches合成为完整图像
        """
        print(f"🖼️  合成图像，共 {len(image_patches)} 个patches")
        
        # 收集所有patch的latent
        patch_latents = []
        for patch_data in image_patches:
            patch_latents.append(patch_data['latent'])
        
        if not patch_latents:
            # 如果没有patch数据，生成随机图像
            H, W = image_shape
            h = H // self.model.latent_downsample
            w = W // self.model.latent_downsample
            patch_dim = self.model.latent_patch_size ** 2 * self.model.latent_channel
            total_patches = h * w
            
            combined_latent = torch.randn(total_patches, patch_dim, device=self.device)
        else:
            combined_latent = torch.stack(patch_latents, dim=0)
        
        # 解码为图像
        decoded_image = self._decode_patches_to_image(combined_latent, image_shape)
        
        return decoded_image
    
    def _decode_patches_to_image(
        self, 
        latent_patches: torch.Tensor, 
        image_shape: tuple
    ) -> Image.Image:
        """
        将latent patches解码为完整图像
        """
        H, W = image_shape
        h = H // self.model.latent_downsample
        w = W // self.model.latent_downsample
        
        # 重塑为图像格式
        latent = latent_patches.reshape(
            1, h, w, 
            self.model.latent_patch_size, 
            self.model.latent_patch_size, 
            self.model.latent_channel
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1, self.model.latent_channel, 
            h * self.model.latent_patch_size, 
            w * self.model.latent_patch_size
        )
        
        # VAE解码
        with torch.no_grad():
            image = self.vae_model.decode(latent)
        
        # 转换为PIL图像
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray(image.to(torch.uint8).cpu().numpy())
        
        return image
    
    def _parse_generated_sequence(
        self, 
        generated_sequence: List[Dict[str, Any]]
    ) -> List[Union[str, Image.Image]]:
        """
        解析生成序列，组合文本和图像输出
        """
        output_results = []
        text_buffer = []
        
        for item in generated_sequence:
            item_type = item['type']
            
            if item_type == 'text_token':
                text_buffer.append(item['content'])
                
            elif item_type == 'special_token':
                # 特殊token不包含在文本输出中，但可以作为分隔符
                if text_buffer and item['token_name'] in ['start_of_image']:
                    # 在图像开始前，输出累积的文本
                    try:
                        text = self.tokenizer.decode(text_buffer, skip_special_tokens=True)
                        if text.strip():
                            output_results.append(text.strip())
                    except:
                        output_results.append("[文本解码失败]")
                    text_buffer = []
                    
            elif item_type == 'image':
                # 添加生成的图像
                output_results.append(item['content'])
                
            elif item_type == 'image_patch':
                # patch不直接输出，由_finalize_image_generation处理
                pass
        
        # 处理剩余的文本
        if text_buffer:
            try:
                text = self.tokenizer.decode(text_buffer, skip_special_tokens=True)
                if text.strip():
                    output_results.append(text.strip())
            except:
                output_results.append("[文本解码失败]")
        
        return output_results


def load_bagel_model_for_inference(
    model_path: str,
    mode: int = 1,  # 1: 正常模式, 2: NF4量化, 3: INT8量化
    device: str = "cuda"
) -> Tuple[Bagel, Any, Qwen2Tokenizer, ImageTransform, ImageTransform, Dict[str, int]]:
    """
    正确加载BAGEL模型用于推理
    参考app.py和inferencer.py的加载方式
    
    Args:
        model_path: 模型路径
        mode: 加载模式 (1: 正常, 2: NF4量化, 3: INT8量化)
        device: 设备
        
    Returns:
        (model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    """
    print(f"📦 开始加载BAGEL模型，路径: {model_path}")
    print(f"🔧 加载模式: {mode} (1=正常, 2=NF4量化, 3=INT8量化)")
    
    # 1. 加载配置文件
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    
    # 2. 加载VAE模型
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    print("✅ VAE模型加载完成")
    
    # 3. 创建BAGEL配置
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )
    
    # 4. 创建模型（使用init_empty_weights避免内存问题）
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    
    # 5. 加载tokenizer和特殊tokens
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    print("✅ Tokenizer和特殊tokens加载完成")
    
    # 6. 创建图像变换
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    print("✅ 图像变换创建完成")
    
    # 7. 设置设备映射
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    
    # 确保相关模块在同一设备上
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            device_map[k] = first_device
    
    # 8. 根据模式加载权重
    if mode == 1:  # 正常模式
        model = load_checkpoint_and_dispatch(
            model, 
            checkpoint=os.path.join(model_path, "ema.safetensors"), 
            device_map=device_map,
            offload_folder="offload",
            dtype=torch.bfloat16,
        ).eval()
        print("✅ 模型权重加载完成 (正常模式)")
        
    elif mode == 2:  # NF4量化
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=False, 
            bnb_4bit_quant_type="nf4"
        )
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
        print("✅ 模型权重加载完成 (NF4量化模式)")
        
    elif mode == 3:  # INT8量化
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True, 
            torch_dtype=torch.float32
        )
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
        print("✅ 模型权重加载完成 (INT8量化模式)")
        
    else:
        raise NotImplementedError(f"不支持的加载模式: {mode}")
    
    print("🎉 BAGEL模型加载完成！")
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


class UnifiedImageEditingInference:
    """统一图像编辑推理引擎"""
    
    def __init__(
        self,
        model_path: str,
        mode: int = 1,
        device: str = "cuda"
    ):
        """
        初始化图像编辑推理引擎
        
        Args:
            model_path: BAGEL模型路径
            mode: 加载模式 (1: 正常, 2: NF4量化, 3: INT8量化)
            device: 设备
        """
        self.device = device
        self.model_path = model_path
        
        # 加载模型和相关组件
        (self.model, self.vae_model, self.tokenizer, 
         self.vae_transform, self.vit_transform, self.new_token_ids) = load_bagel_model_for_inference(
            model_path=model_path,
            mode=mode,
            device=device
        )
        
        # 创建推理器（保留原有的兼容性）
        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids
        )
        
        # 创建统一自回归推理器（新增）
        self.unified_inferencer = UnifiedAutoregressiveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids
        )
        
        print("🚀 图像编辑推理引擎初始化完成！")
    
    def edit_image(
        self,
        image_path: str,
        edit_prompt: str,
        think: bool = True,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
        image_shapes: Tuple[int, int] = (1024, 1024),
        **kwargs
    ) -> List[Union[str, Image.Image]]:
        """
        图像编辑的主要接口
        
        Args:
            image_path: 输入图像路径
            edit_prompt: 编辑提示词
            think: 是否启用思考模式
            cfg_text_scale: 文本CFG强度
            cfg_img_scale: 图像CFG强度
            num_timesteps: 去噪步数
            image_shapes: 输出图像尺寸
            **kwargs: 其他参数
            
        Returns:
            生成结果列表（包含文本和图像）
        """
        print(f"🖼️  开始图像编辑")
        print(f"📸 输入图像: {image_path}")
        print(f"✏️  编辑提示: {edit_prompt}")
        
        # 加载输入图像
        try:
            input_image = Image.open(image_path).convert('RGB')
            print(f"✅ 图像加载成功，尺寸: {input_image.size}")
        except Exception as e:
            print(f"❌ 图像加载失败: {e}")
            return [f"图像加载失败: {e}"]
        
        # 构建输入列表
        input_lists = [input_image, edit_prompt]
        
        # 执行推理
        try:
            results = self.inferencer.interleave_inference(
                input_lists=input_lists,
                think=think,
                understanding_output=False,  # 编辑模式，不是理解模式
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                num_timesteps=num_timesteps,
                image_shapes=image_shapes,
                **kwargs
            )
            
            print(f"✅ 编辑完成，生成了 {len(results)} 个结果")
            return results
            
        except Exception as e:
            print(f"❌ 图像编辑失败: {e}")
            import traceback
            traceback.print_exc()
            return [f"图像编辑失败: {e}"]
    
    def unified_edit_image(
        self,
        image_path: str,
        edit_prompt: str,
        use_autoregressive: bool = True,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 0.8,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
        image_shapes: Tuple[int, int] = (1024, 1024),
        **kwargs
    ) -> List[Union[str, Image.Image]]:
        """
        统一图像编辑接口 - 支持真正的自回归交错推理
        
        Args:
            image_path: 输入图像路径
            edit_prompt: 编辑提示词
            use_autoregressive: 是否使用统一自回归模式（对应训练）
            max_length: 最大生成长度
            do_sample: 是否采样
            temperature: 采样温度
            cfg_text_scale: 文本CFG强度
            cfg_img_scale: 图像CFG强度
            num_timesteps: 去噪步数
            image_shapes: 输出图像尺寸
            **kwargs: 其他参数
            
        Returns:
            生成结果列表（包含文本和图像）
        """
        print(f"🎨 开始统一图像编辑")
        print(f"📸 输入图像: {image_path}")
        print(f"✏️  编辑提示: {edit_prompt}")
        print(f"🔧 自回归模式: {use_autoregressive}")
        
        # 加载输入图像
        try:
            input_image = Image.open(image_path).convert('RGB')
            print(f"✅ 图像加载成功，尺寸: {input_image.size}")
        except Exception as e:
            print(f"❌ 图像加载失败: {e}")
            return [f"图像加载失败: {e}"]
        
        try:
            if use_autoregressive:
                # 使用统一自回归推理（对应训练逻辑）
                print("🚀 使用统一自回归推理模式")
                results = self.unified_inferencer.unified_autoregressive_inference(
                    input_text=edit_prompt,
                    input_image=input_image,
                    max_length=max_length,
                    do_sample=do_sample,
                    temperature=temperature,
                    image_shapes=image_shapes,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    num_timesteps=num_timesteps,
                    timestep_shift=3.0,  # 添加默认值
                    **kwargs
                )
            else:
                # 使用传统interleave推理（兼容性）
                print("🔄 使用传统interleave推理模式")
                input_lists = [input_image, edit_prompt]
                results = self.inferencer.interleave_inference(
                    input_lists=input_lists,
                    understanding_output=False,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    num_timesteps=num_timesteps,
                    image_shapes=image_shapes,
                    **kwargs
                )
            
            print(f"✅ 图像编辑完成，生成了 {len(results)} 个结果")
            return results
            
        except Exception as e:
            print(f"❌ 统一图像编辑失败: {e}")
            import traceback
            traceback.print_exc()
            return [f"统一图像编辑失败: {e}"]
    
    def autoregressive_multi_modal_generation(
        self,
        prompt: str,
        input_image: Optional[str] = None,
        force_image_generation: bool = False,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 0.8,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
        image_shapes: Tuple[int, int] = (1024, 1024),
        **kwargs
    ) -> List[Union[str, Image.Image]]:
        """
        自回归多模态生成 - 完全对应训练时的统一序列建模
        
        Args:
            prompt: 输入提示词
            input_image: 可选的输入图像路径
            force_image_generation: 是否强制生成图像
            max_length: 最大生成长度
            do_sample: 是否采样
            temperature: 采样温度
            cfg_text_scale: 文本CFG强度
            cfg_img_scale: 图像CFG强度
            num_timesteps: 去噪步数
            image_shapes: 输出图像尺寸
            **kwargs: 其他参数
            
        Returns:
            生成结果列表（文本和图像交错）
        """
        print(f"🎯 开始自回归多模态生成")
        print(f"📝 输入提示: {prompt}")
        print(f"🖼️  输入图像: {input_image if input_image else 'None'}")
        print(f"🔧 强制图像生成: {force_image_generation}")
        
        # 处理输入图像
        image_obj = None
        if input_image:
            try:
                image_obj = Image.open(input_image).convert('RGB')
                print(f"✅ 输入图像加载成功，尺寸: {image_obj.size}")
            except Exception as e:
                print(f"❌ 输入图像加载失败: {e}")
                return [f"输入图像加载失败: {e}"]
        
        # 构建完整的提示词
        if force_image_generation and '<|vision_start|>' not in prompt:
            # 强制添加图像生成token
            enhanced_prompt = f"{prompt} <|vision_start|> <|vision_end|>"
            print(f"🔧 强制图像生成，增强提示: {enhanced_prompt}")
        else:
            enhanced_prompt = prompt
        
        try:
            # 使用统一自回归推理
            results = self.unified_inferencer.unified_autoregressive_inference(
                input_text=enhanced_prompt,
                input_image=image_obj,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                image_shapes=image_shapes,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                num_timesteps=num_timesteps,
                **kwargs
            )
            
            print(f"✅ 自回归多模态生成完成，生成了 {len(results)} 个结果")
            return results
            
        except Exception as e:
            print(f"❌ 自回归多模态生成失败: {e}")
            import traceback
            traceback.print_exc()
            return [f"自回归多模态生成失败: {e}"]
    
    def batch_edit_images(
        self,
        image_paths: List[str],
        edit_prompts: List[str],
        output_dir: str = "edited_images",
        use_autoregressive: bool = True,
        **kwargs
    ) -> Dict[str, List[Union[str, Image.Image]]]:
        """
        批量图像编辑
        
        Args:
            image_paths: 输入图像路径列表
            edit_prompts: 编辑提示词列表
            output_dir: 输出目录
            use_autoregressive: 是否使用统一自回归模式
            **kwargs: 其他参数
            
        Returns:
            编辑结果字典
        """
        print(f"📦 开始批量图像编辑，共 {len(image_paths)} 张图像")
        print(f"🔧 自回归模式: {use_autoregressive}")
        
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for i, (image_path, edit_prompt) in enumerate(zip(image_paths, edit_prompts)):
            print(f"\n🔄 处理第 {i+1}/{len(image_paths)} 张图像")
            
            # 单张图像编辑 - 使用统一接口
            if use_autoregressive:
                edit_results = self.unified_edit_image(
                    image_path=image_path,
                    edit_prompt=edit_prompt,
                    use_autoregressive=True,
                    **kwargs
                )
            else:
                edit_results = self.edit_image(
                    image_path=image_path,
                    edit_prompt=edit_prompt,
                    **kwargs
                )
            
            # 保存结果
            results[f"image_{i+1}"] = edit_results
            
            # 保存生成的图像
            for j, result in enumerate(edit_results):
                if isinstance(result, Image.Image):
                    output_path = os.path.join(output_dir, f"image_{i+1}_result_{j}.png")
                    result.save(output_path)
                    print(f"💾 已保存: {output_path}")
                elif isinstance(result, str):
                    output_path = os.path.join(output_dir, f"image_{i+1}_text_{j}.txt")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(result)
                    print(f"💾 已保存: {output_path}")
        
        print(f"🎉 批量编辑完成！结果保存在 {output_dir}")
        return results
    
    def multi_step_edit(
        self,
        image_path: str,
        edit_steps: List[str],
        output_dir: str = "multi_step_edit",
        use_autoregressive: bool = True,
        **kwargs
    ) -> List[Union[str, Image.Image]]:
        """
        多步骤图像编辑
        
        Args:
            image_path: 输入图像路径
            edit_steps: 编辑步骤列表
            output_dir: 输出目录
            use_autoregressive: 是否使用统一自回归模式
            **kwargs: 其他参数
            
        Returns:
            所有步骤的结果
        """
        print(f"🎯 开始多步骤图像编辑，共 {len(edit_steps)} 个步骤")
        print(f"🔧 自回归模式: {use_autoregressive}")
        
        os.makedirs(output_dir, exist_ok=True)
        all_results = []
        current_image_path = image_path
        
        for i, edit_step in enumerate(edit_steps):
            print(f"\n🔄 执行步骤 {i+1}/{len(edit_steps)}: {edit_step}")
            
            # 执行当前步骤 - 使用统一接口
            if use_autoregressive:
                step_results = self.unified_edit_image(
                    image_path=current_image_path,
                    edit_prompt=edit_step,
                    use_autoregressive=True,
                    **kwargs
                )
            else:
                step_results = self.edit_image(
                    image_path=current_image_path,
                    edit_prompt=edit_step,
                    **kwargs
                )
            
            all_results.extend(step_results)
            
            # 保存当前步骤结果
            step_dir = os.path.join(output_dir, f"step_{i+1}")
            os.makedirs(step_dir, exist_ok=True)
            
            for j, result in enumerate(step_results):
                if isinstance(result, Image.Image):
                    result_path = os.path.join(step_dir, f"result_{j}.png")
                    result.save(result_path)
                    print(f"💾 步骤 {i+1} 图像已保存: {result_path}")
                    
                    # 更新current_image_path为最新生成的图像（用于下一步）
                    current_image_path = result_path
                    
                elif isinstance(result, str):
                    result_path = os.path.join(step_dir, f"text_{j}.txt")
                    with open(result_path, "w", encoding="utf-8") as f:
                        f.write(result)
                    print(f"💾 步骤 {i+1} 文本已保存: {result_path}")
        
        print(f"🎉 多步骤编辑完成！结果保存在 {output_dir}")
        return all_results


def main():
    """主函数：命令行界面"""
    parser = argparse.ArgumentParser(description="BAGEL统一图像编辑推理")
    parser.add_argument("--model_path", type=str, required=True, help="BAGEL模型路径")
    parser.add_argument("--image_path", type=str, help="输入图像路径")
    parser.add_argument("--edit_prompt", type=str, help="编辑提示词")
    parser.add_argument("--prompt", type=str, help="通用提示词（用于多模态生成）")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--mode", type=int, default=1, help="模型加载模式 (1: 正常, 2: NF4量化, 3: INT8量化)")
    
    # 推理模式选择
    parser.add_argument("--use_autoregressive", action="store_true", default=True, 
                      help="使用统一自回归推理模式（对应训练）")
    parser.add_argument("--use_legacy", action="store_true", 
                      help="使用传统interleave推理模式（兼容性）")
    parser.add_argument("--generation_mode", type=str, choices=["edit", "generate", "multi_modal"], 
                      default="edit", help="生成模式：edit=图像编辑, generate=纯生成, multi_modal=多模态生成")
    
    # 生成参数
    parser.add_argument("--max_length", type=int, default=500, help="最大生成长度")
    parser.add_argument("--do_sample", action="store_true", default=True, help="是否采样")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--force_image_generation", action="store_true", help="强制生成图像")
    
    parser.add_argument("--think", action="store_true", help="启用思考模式")
    parser.add_argument("--cfg_text_scale", type=float, default=3.0, help="文本CFG强度")
    parser.add_argument("--cfg_img_scale", type=float, default=1.5, help="图像CFG强度")
    parser.add_argument("--num_timesteps", type=int, default=50, help="去噪步数")
    parser.add_argument("--multi_step", nargs="+", help="多步骤编辑（提供多个编辑步骤）")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 处理推理模式
    use_autoregressive = args.use_autoregressive and not args.use_legacy
    
    print("🚀 初始化BAGEL统一推理引擎...")
    print(f"🔧 推理模式: {'统一自回归' if use_autoregressive else '传统交错'}")
    print(f"🎯 生成模式: {args.generation_mode}")
    
    try:
        # 创建推理引擎
        inference_engine = UnifiedImageEditingInference(
            model_path=args.model_path,
            mode=args.mode
        )
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 根据生成模式执行不同的逻辑
        if args.generation_mode == "edit":
            # 图像编辑模式
            if not args.image_path or not args.edit_prompt:
                print("❌ 图像编辑模式需要 --image_path 和 --edit_prompt 参数")
                return
                
            if args.multi_step:
                # 多步骤编辑
                print("🎯 执行多步骤图像编辑")
                results = inference_engine.multi_step_edit(
                    image_path=args.image_path,
                    edit_steps=args.multi_step,
                    output_dir=args.output_dir,
                    use_autoregressive=use_autoregressive,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    cfg_text_scale=args.cfg_text_scale,
                    cfg_img_scale=args.cfg_img_scale,
                    num_timesteps=args.num_timesteps
                )
            else:
                # 单步编辑
                print("✏️  执行单步图像编辑")
                results = inference_engine.unified_edit_image(
                    image_path=args.image_path,
                    edit_prompt=args.edit_prompt,
                    use_autoregressive=use_autoregressive,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    cfg_text_scale=args.cfg_text_scale,
                    cfg_img_scale=args.cfg_img_scale,
                    num_timesteps=args.num_timesteps
                )
                
                # 保存结果
                for i, result in enumerate(results):
                    if isinstance(result, Image.Image):
                        output_path = os.path.join(args.output_dir, f"edited_image_{i}.png")
                        result.save(output_path)
                        print(f"💾 编辑结果已保存: {output_path}")
                    elif isinstance(result, str):
                        output_path = os.path.join(args.output_dir, f"text_result_{i}.txt")
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(result)
                        print(f"💾 文本结果已保存: {output_path}")
                        
        elif args.generation_mode == "multi_modal":
            # 多模态生成模式
            if not args.prompt:
                print("❌ 多模态生成模式需要 --prompt 参数")
                return
                
            print("🎯 执行自回归多模态生成")
            results = inference_engine.autoregressive_multi_modal_generation(
                prompt=args.prompt,
                input_image=args.image_path,
                force_image_generation=args.force_image_generation,
                max_length=args.max_length,
                do_sample=args.do_sample,
                temperature=args.temperature,
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
                num_timesteps=args.num_timesteps
            )
            
            # 保存结果
            for i, result in enumerate(results):
                if isinstance(result, Image.Image):
                    output_path = os.path.join(args.output_dir, f"generated_image_{i}.png")
                    result.save(output_path)
                    print(f"💾 生成图像已保存: {output_path}")
                elif isinstance(result, str):
                    output_path = os.path.join(args.output_dir, f"generated_text_{i}.txt")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(result)
                    print(f"💾 生成文本已保存: {output_path}")
                    
        elif args.generation_mode == "generate":
            # 纯生成模式（使用传统方法作为对比）
            if not args.prompt:
                print("❌ 生成模式需要 --prompt 参数")
                return
                
            print("🎯 执行传统推理生成")
            if args.image_path:
                input_image = Image.open(args.image_path).convert('RGB')
                input_lists = [input_image, args.prompt]
            else:
                input_lists = [args.prompt]
                
            results = inference_engine.inferencer.interleave_inference(
                input_lists=input_lists,
                think=args.think,
                understanding_output=False,
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
                num_timesteps=args.num_timesteps
            )
            
            # 保存结果
            for i, result in enumerate(results):
                if isinstance(result, Image.Image):
                    output_path = os.path.join(args.output_dir, f"traditional_image_{i}.png")
                    result.save(output_path)
                    print(f"💾 传统生成图像已保存: {output_path}")
                elif isinstance(result, str):
                    output_path = os.path.join(args.output_dir, f"traditional_text_{i}.txt")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(result)
                    print(f"💾 传统生成文本已保存: {output_path}")
        
        print("🎉 推理完成！")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
