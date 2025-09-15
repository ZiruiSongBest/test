
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.cache_utils.taylorseer import cache_init

from tqdm import tqdm


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und and vit_token_seqlens is not None:
            # 数据完整性检查
            if len(vit_token_seqlens) == 0:
                # 如果没有VIT数据，跳过VIT处理
                pass
            else:
                # 检查是否有负数或异常值
                if torch.any(vit_token_seqlens < 0):
                    print(f"错误：vit_token_seqlens包含负数: {vit_token_seqlens}")
                    print(f"形状: {vit_token_seqlens.shape}, 数据类型: {vit_token_seqlens.dtype}")
                    raise ValueError(f"vit_token_seqlens包含无效的负数值")
                
                # 检查是否所有值都是0（没有实际的VIT数据）
                if torch.all(vit_token_seqlens == 0):
                    # 所有图像都没有VIT数据，跳过VIT处理
                    pass
                else:
                    max_val = torch.max(vit_token_seqlens).item()
                    if max_val > 10000:  # 设置一个合理的上限，防止内存溢出
                        print(f"警告：vit_token_seqlens包含异常大的值: {max_val}")
                        print(f"完整张量: {vit_token_seqlens}")
                    
                    cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
                    cu_seqlens = cu_seqlens.to(torch.int32)
                    max_seqlen = torch.max(vit_token_seqlens).item()
                    packed_vit_token_embed = self.vit_model(
                        packed_pixel_values=packed_vit_tokens, 
                        packed_flattened_position_ids=packed_vit_position_ids,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
                    packed_vit_token_embed = self.connector(packed_vit_token_embed)
                    vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
                    packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
                    packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen:
            # 确保索引张量与数据张量在同一设备上
            if mse_loss_indexes.device != last_hidden_state.device:
                mse_loss_indexes = mse_loss_indexes.to(last_hidden_state.device)
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        if ce_loss_indexes is not None:
            # 确保索引张量与数据张量在同一设备上
            if ce_loss_indexes.device != last_hidden_state.device:
                ce_loss_indexes = ce_loss_indexes.to(last_hidden_state.device)
            if packed_label_ids.device != last_hidden_state.device:
                packed_label_ids = packed_label_ids.to(last_hidden_state.device)
                
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            # packed_label_ids已经是对应需要计算损失位置的标签，直接使用
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)

    def forward_autoregressive_training(
        self,
        input_text: str,
        input_image: torch.Tensor,
        target_tokens: List[int],
        target_images: List[torch.Tensor],
        tokenizer,
        vae_model,
        new_token_ids: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """
        自回归序列生成的训练前向传播
        
        Args:
            input_text: 输入文本
            input_image: 输入图像 (已经过vit_transform处理)
            target_tokens: 已经tokenize的目标序列，包含文本token和vision token
            target_images: 与<|vision_start|>标记对应的图像张量列表 (已经过vae_transform处理)
            tokenizer: 分词器
            vae_model: VAE模型
            new_token_ids: 特殊token的ID映射
        
        Returns:
            包含各步骤损失的字典
        """
        device = next(self.parameters()).device
        
        # 1. 处理输入部分 (prompt文本和图像)
        input_llm_embeddings, _, _ = self._process_input_stage(
            input_text, input_image, tokenizer, device
        )
        
        # 2. 构建统一的目标序列信息 (文本token, 特殊token, 图像patch占位符, 以及真实图像latent信息)
        unified_sequence_info = self._build_unified_target_sequence(target_tokens, target_images, new_token_ids, vae_model, device)
        
        # 3. 统一自回归训练：逐token处理整个目标序列，计算损失
        losses = self._process_unified_autoregressive_training(
            input_llm_embeddings,  # LLM的初始输入是prompt部分的embedding
            unified_sequence_info, 
            tokenizer, 
            new_token_ids, 
            device
        )
        
        return losses
    
    def _process_unified_autoregressive_training(
        self,
        input_sequence: torch.Tensor,
        unified_sequence_info: Dict[str, any],
        tokenizer,
        new_token_ids: Dict[str, int],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        统一的自回归训练处理，逐token计算loss
        
        Args:
            input_sequence: 初始的输入序列embedding (prompt文本 + input_image_vit_embeddings)
            unified_sequence_info: 统一构建的目标序列信息
            tokenizer: 分词器
            new_token_ids: 特殊token的ID映射
            device: 计算设备
        
        Returns:
            包含各种loss的字典
        """
        unified_tokens = unified_sequence_info['unified_tokens']
        token_types = unified_sequence_info['token_types']
        image_embeddings_list = unified_sequence_info['image_embeddings']  # 这是一个列表，每个元素包含'latent', 'shape', 'num_patches', 'latent_position_ids'
        loss_mask = unified_sequence_info['loss_mask']
        
        # 损失统计
        text_losses = []
        special_token_losses = []  # 分开特殊token的损失
        image_flow_losses = []
        
        # 当前已生成的序列，初始为input_sequence
        current_llm_input_embeddings = input_sequence.clone()
        
        # 跟踪图像数据
        current_image_idx = 0       # 当前正在处理的图像索引
        current_patch_in_image_idx = 0  # 当前图像中的patch索引
        
        # 迭代整个目标序列，逐个token进行预测和损失计算
        for step_in_target, (target_token_id_or_placeholder, token_type, need_loss) in enumerate(zip(unified_tokens, token_types, loss_mask)):
            
            if not need_loss:
                continue
                
            # 准备LLM的输入
            llm_input_len = current_llm_input_embeddings.size(0)
            
            # 构建正确的position_ids：从0开始到当前序列长度
            llm_position_ids = torch.arange(llm_input_len, device=device)
            
            # LLM前向传播
            output = self.language_model(
                packed_sequence=current_llm_input_embeddings.unsqueeze(0),
                sample_lens=[llm_input_len],
                attention_mask=None,
                packed_position_ids=llm_position_ids,
            )
            
            # 获取最后一个token的隐藏状态，用于预测下一个token
            last_hidden_state = output[0, -1, :]  # 形状: [hidden_size]
            
            # 根据token类型计算损失并更新序列
            if token_type in ['text', 'special']:
                # 文本token或特殊token使用CE loss
                logits = self.language_model.lm_head(last_hidden_state.unsqueeze(0))  # 形状: [1, vocab_size]
                
                target_token_tensor = torch.tensor([target_token_id_or_placeholder], device=device)
                
                # 计算Cross-Entropy Loss
                loss = F.cross_entropy(logits, target_token_tensor, reduction="mean")
                
                if token_type == 'special':
                    # 给特殊token（特别是<vision_start>）更高权重
                    if target_token_id_or_placeholder == new_token_ids.get('start_of_image'):
                        loss = loss * 2.0
                    special_token_losses.append(loss)
                else:
                    text_losses.append(loss)
                
                # 将真实的token embedding添加到序列中
                token_embedding = self.language_model.model.embed_tokens(target_token_tensor)
                current_llm_input_embeddings = torch.cat([current_llm_input_embeddings, token_embedding.squeeze(0)], dim=0)
                
            elif token_type == 'image':
                # 图像token（patch）使用Flow Matching loss
                
                # 获取当前图像信息
                if current_image_idx < len(image_embeddings_list):
                    img_info = image_embeddings_list[current_image_idx]
                    all_patches_latent = img_info['latent']  # 形状: [num_patches, patch_dim]
                    latent_position_ids = img_info['latent_position_ids']  # 形状: [num_patches]
                    
                    # 获取当前patch的真实latent
                    target_patch_latent = all_patches_latent[current_patch_in_image_idx:current_patch_in_image_idx+1]
                    current_patch_pos_id = latent_position_ids[current_patch_in_image_idx:current_patch_in_image_idx+1]
                    
                    # 随机采样时间步进行Flow Matching训练
                    timestep = torch.rand(1, device=device) * 0.9 + 0.05  # 避免极端值
                    
                    # 处理时间步
                    timestep_processed = torch.sigmoid(timestep)
                    timestep_processed = self.timestep_shift * timestep_processed / (1 + (self.timestep_shift - 1) * timestep_processed)
                    
                    # 生成噪声
                    noise = torch.randn_like(target_patch_latent)
                    
                    # 构造noisy latent
                    noisy_patch_latent = (1 - timestep_processed[:, None]) * target_patch_latent + timestep_processed[:, None] * noise
                    
                    # 编码为LLM的embedding
                    timestep_embed = self.time_embedder(timestep)
                    latent_pos_embed = self.latent_pos_embed(current_patch_pos_id)
                    
                    # 这里使用noisy_patch_latent作为输入
                    llm_input_for_patch = self.vae2llm(noisy_patch_latent) + timestep_embed + latent_pos_embed
                    
                    # LLM预测velocity (v_t = dx/dt)
                    predicted_velocity = self.llm2vae(last_hidden_state.unsqueeze(0))  # 形状: [1, patch_dim]
                    
                    # 计算Flow Matching Loss
                    # Target velocity: noise - clean_latent (pointing from data to noise)
                    target_velocity = noise - target_patch_latent
                    flow_loss = ((predicted_velocity - target_velocity) ** 2).mean()
                    
                    image_flow_losses.append(flow_loss)
                    
                    # 将patch embedding添加到序列中
                    # 注意：这里使用teacher forcing，即使用真实的latent embedding而不是预测的embedding，以保持训练稳定性
                    true_patch_embedding = self.vae2llm(target_patch_latent) + timestep_embed + latent_pos_embed
                    current_llm_input_embeddings = torch.cat([current_llm_input_embeddings, true_patch_embedding.squeeze(0)], dim=0)
                    
                    # 更新patch索引
                    current_patch_in_image_idx += 1
                    if current_patch_in_image_idx >= img_info['num_patches']:
                        current_image_idx += 1
                        current_patch_in_image_idx = 0
                else:
                    raise IndexError(f"图像索引超出范围: {current_image_idx}")
        
        # 汇总损失
        total_text_loss = torch.stack(text_losses).mean() if text_losses else torch.tensor(0.0, device=device)
        total_special_loss = torch.stack(special_token_losses).mean() if special_token_losses else torch.tensor(0.0, device=device)
        total_image_loss = torch.stack(image_flow_losses).mean() if image_flow_losses else torch.tensor(0.0, device=device)
        
        # 合并文本和特殊token的loss
        total_text_and_special_loss = (total_text_loss + total_special_loss) if (text_losses or special_token_losses) else torch.tensor(0.0, device=device)
        
        return {
            "total_loss": total_text_and_special_loss + total_image_loss,
            "text_loss": total_text_loss,
            "special_token_loss": total_special_loss,
            "image_loss": total_image_loss,
            "detailed_losses": {  # 可选，用于调试
                "text_losses_list": text_losses,
                "special_token_losses_list": special_token_losses,
                "image_flow_losses_list": image_flow_losses
            }
        }
    
    def _process_input_stage(
        self, 
        input_text: str, 
        input_image: torch.Tensor, 
        tokenizer, 
        device: torch.device
    ) -> Tuple[torch.Tensor, List[int], int]:
        """处理输入阶段：文本+图像"""
        
        # 1. 处理输入文本
        text_ids = tokenizer.encode(input_text)
        text_embedding = self.language_model.model.embed_tokens(torch.tensor(text_ids, device=device))
        
        # 2. 处理输入图像
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)  # 添加batch维度
        
        # 使用ViT处理输入图像
        vit_position_ids = self.get_flattened_position_ids(
            input_image.size(2), input_image.size(3),
            self.vit_patch_size,
            max_num_patches_per_side=self.vit_max_num_patch_per_side
        ).to(device)
        
        vit_tokens = patchify(input_image.squeeze(0), self.vit_patch_size)
        
        # 使用ViT模型处理
        cu_seqlens = torch.tensor([0, vit_tokens.shape[0]], dtype=torch.int32, device=device)
        vit_embeddings = self.vit_model(
            packed_pixel_values=vit_tokens,
            packed_flattened_position_ids=vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=vit_tokens.shape[0],
        )
        
        # 应用连接器和位置编码
        vit_embeddings = self.connector(vit_embeddings)
        vit_pos_emb = self.vit_pos_embed(vit_position_ids)
        vit_embeddings = vit_embeddings + vit_pos_emb
        
        # 3. 构建初始序列
        input_sequence = torch.cat([text_embedding, vit_embeddings], dim=0)
        input_indexes = list(range(len(text_ids) + len(vit_embeddings)))
        
        return input_sequence, input_indexes, 1  # position从1开始
    
    def _build_unified_target_sequence(
        self, 
        target_tokens: List[int], 
        target_images: List[torch.Tensor],
        new_token_ids: Dict[str, int],
        vae_model,
        device: torch.device
    ) -> Dict[str, any]:
        """
        构建统一的目标序列，将文本token、特殊token和图像token统一建模
        
        Args:
            target_tokens: 已经tokenize的目标序列，包含特殊token标记
            target_images: 与<|vision_start|>标记对应的图像张量列表
            new_token_ids: 特殊token的ID映射
            vae_model: VAE模型，用于编码图像
            device: 计算设备
        
        Returns:
            统一序列信息的字典，包含：
            - unified_tokens: 完整的token序列（包括占位的图像token位置）
            - token_types: 每个位置的token类型 ('text', 'special', 'image')
            - image_embeddings: 图像token对应的embedding
            - image_positions: 图像token在序列中的位置
            - loss_mask: 哪些位置需要计算loss
        """
        start_of_image = new_token_ids.get('start_of_image')
        end_of_image = new_token_ids.get('end_of_image')
        
        # 预处理：编码所有图像为latent并转换为patches
        encoded_images = []
        for image in target_images:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            with torch.no_grad():
                latent = vae_model.encode(image.to(device))
            
            # 转换为patches
            p = self.latent_patch_size
            latent = latent[0]  # 取第一个batch
            h, w = latent.shape[1] // p, latent.shape[2] // p
            
            # Patchify latent
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            
            # 为该图像的所有latent patch生成位置ID
            all_latent_position_ids = self.get_flattened_position_ids(
                h * self.latent_downsample, w * self.latent_downsample,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size
            ).to(device)
            
            encoded_images.append({
                'latent': latent,
                'shape': (h, w),
                'num_patches': latent.shape[0],
                'latent_position_ids': all_latent_position_ids  # 存储每个patch的位置ID
            })
        
        # 构建统一序列
        unified_tokens = []
        token_types = []
        image_embeddings = []
        image_positions = []
        loss_mask = []
        
        image_idx = 0
        position = 0
        
        i = 0
        while i < len(target_tokens):
            token_id = target_tokens[i]
            
            if token_id == start_of_image:
                # 添加 <vision_start> token
                unified_tokens.append(start_of_image)
                token_types.append('special')
                loss_mask.append(True)  # 特殊token需要计算loss
                position += 1
                
                # 查找对应的 <vision_end> token
                j = i + 1
                while j < len(target_tokens) and target_tokens[j] != end_of_image:
                    j += 1
                
                if j >= len(target_tokens):
                    raise ValueError(f"找到<|vision_start|>但没有找到对应的<|vision_end|>")
                
                # 添加图像tokens（使用特殊的占位token）
                if image_idx < len(encoded_images):
                    img_info = encoded_images[image_idx]
                    num_patches = img_info['num_patches']
                    
                    # 记录图像embedding和位置信息
                    image_embeddings.append(img_info)
                    image_positions.extend(list(range(position, position + num_patches)))
                    
                    # 添加占位token和类型标记
                    for patch_idx in range(num_patches):
                        unified_tokens.append(-1)  # 占位token，-1表示图像token
                        token_types.append('image')
                        loss_mask.append(True)  # 图像token需要计算loss
                        position += 1
                    
                    image_idx += 1
                else:
                    raise ValueError(f"目标序列中有 {image_idx + 1} 个图像标记，但只提供了 {len(target_images)} 张图像")
                
                # 添加 <vision_end> token
                unified_tokens.append(end_of_image)
                token_types.append('special')
                loss_mask.append(True)  # 特殊token需要计算loss
                position += 1
                
                # 跳过到end_of_image之后
                i = j + 1
                
            else:
                # 普通文本token
                unified_tokens.append(token_id)
                token_types.append('text')
                loss_mask.append(True)  # 文本token需要计算loss
                position += 1
                i += 1
        
        # 检查是否还有未使用的图像
        if image_idx < len(target_images):
            raise ValueError(f"提供了 {len(target_images)} 张图像，但目标序列中只有 {image_idx} 个图像标记")
        
        return {
            'unified_tokens': unified_tokens,
            'token_types': token_types,
            'image_embeddings': image_embeddings,
            'image_positions': image_positions,
            'loss_mask': loss_mask,
            'sequence_length': len(unified_tokens)
        }
    
    def forward_autoregressive_training_example(self):
        """
        统一序列建模的使用示例：
        
        # 训练数据构建：
        # 输入: "用户问题" + input_image
        # 输出: "思考文本1 <|vision_start|><|vision_end|> 思考文本2 <|vision_start|><|vision_end|>"
        
        # 1. 原始目标序列（已经包含特殊token）
        target_sequence_text = "我来分析这个问题 <|vision_start|><|vision_end|> 基于上图，我认为 <|vision_start|><|vision_end|>"
        
        # 2. Tokenize目标序列（特殊token会被正确编码）
        target_tokens = tokenizer.encode(target_sequence_text)
        # 结果类似：[token1, token2, start_of_image_id, end_of_image_id, token3, token4, start_of_image_id, end_of_image_id]
        
        # 3. 对应的图像列表
        target_images = [generated_image1, generated_image2]  # 与<|vision_start|>位置对应
        
        # 4. 调用统一训练方法
        loss = model.forward_autoregressive_training(
            input_text="请分析这张图片",
            input_image=input_image,
            target_tokens=target_tokens,  # 统一的token序列
            target_images=target_images,  # 对应的图像
            tokenizer=tokenizer,
            vae_model=vae_model,
            new_token_ids=new_token_ids
        )
        
        # 返回的loss包含：
        # - text_loss: 文本token和特殊token的CE loss
        # - image_loss: 图像token的Flow Matching loss  
        # - special_token_loss: 特殊token的单独统计
        
        # 关键优势：
        # 1. 模型学会何时输出<|vision_start|>token（时序控制）
        # 2. 统一的序列建模，训练推理一致
        # 3. 分层次的loss设计，确保时序和内容质量
        """
        pass
    
    def _process_text_generation_step(
        self,
        current_sequence: torch.Tensor,
        target_tokens: List[int],
        tokenizer,
        position_id: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理文本生成步骤"""
        
        # 1. 目标token已经是ID列表，直接使用
        target_ids = target_tokens
        target_tensor = torch.tensor(target_ids, device=device)
        
        # 2. 为当前序列创建注意力掩码和位置ID
        seq_len = current_sequence.size(0)
        position_ids = torch.full((seq_len,), position_id, device=device)
        
        # 3. 前向传播
        output = self.language_model(
            packed_sequence=current_sequence.unsqueeze(0),  # 添加batch维度
            sample_lens=[seq_len],
            attention_mask=None,  # 使用默认因果掩码
            packed_position_ids=position_ids,
        )
        
        # 4. 计算文本损失（在序列末尾预测下一个文本token）
        logits = self.language_model.lm_head(output)  # [1, seq_len, vocab_size]
        
        # 对每个目标token计算损失
        text_losses = []
        updated_sequence = current_sequence.clone()
        
        for i, target_id in enumerate(target_ids):
            # 使用当前序列的最后一个位置预测下一个token
            pred_logits = logits[0, -1, :]  # 最后一个位置的预测
            loss = F.cross_entropy(pred_logits.unsqueeze(0), target_id.unsqueeze(0))
            text_losses.append(loss)
            
            # 将预测的token添加到序列中
            token_embedding = self.language_model.model.embed_tokens(target_id.unsqueeze(0))
            updated_sequence = torch.cat([updated_sequence, token_embedding], dim=0)
            
            # 如果不是最后一个token，需要重新前向传播
            if i < len(target_ids) - 1:
                seq_len = updated_sequence.size(0)
                position_ids = torch.full((seq_len,), position_id, device=device)
                output = self.language_model(
                    packed_sequence=updated_sequence.unsqueeze(0),
                    sample_lens=[seq_len],
                    attention_mask=None,
                    packed_position_ids=position_ids,
                )
                logits = self.language_model.lm_head(output)
        
        text_loss = torch.stack(text_losses).mean()
        return text_loss, updated_sequence
    
    def _process_image_generation_step(
        self,
        current_sequence: torch.Tensor,
        target_image: torch.Tensor,
        vae_model,
        position_id: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理图像生成步骤"""
        
        # 1. 编码目标图像
        if target_image.dim() == 3:
            target_image = target_image.unsqueeze(0)
        
        with torch.no_grad():
            target_latent = vae_model.encode(target_image.to(device))
        
        # 2. 处理latent为patches
        p = self.latent_patch_size
        latent = target_latent[0]  # 取第一个batch
        h, w = latent.shape[1] // p, latent.shape[2] // p
        
        # Patchify latent
        latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
        latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
        
        # 3. 创建位置编码和时间步
        latent_position_ids = self.get_flattened_position_ids(
            h * self.latent_downsample, w * self.latent_downsample,
            self.latent_downsample,
            max_num_patches_per_side=self.max_latent_size
        ).to(device)
        
        # 使用训练时间步（这里用0表示clean image）
        timesteps = torch.zeros(latent.shape[0], device=device)
        
        # 4. 处理当前序列以预测图像
        seq_len = current_sequence.size(0)
        position_ids = torch.full((seq_len,), position_id, device=device)
        
        # 前向传播
        output = self.language_model(
            packed_sequence=current_sequence.unsqueeze(0),
            sample_lens=[seq_len],
            attention_mask=None,
            packed_position_ids=position_ids,
        )
        
        # 5. 预测latent tokens
        # 创建噪声latent作为起点
        noise = torch.randn_like(latent)
        timesteps_processed = torch.sigmoid(timesteps)
        timesteps_processed = self.timestep_shift * timesteps_processed / (1 + (self.timestep_shift - 1) * timesteps_processed)
        
        # 构建输入latent
        input_latent = (1 - timesteps_processed[:, None]) * latent + timesteps_processed[:, None] * noise
        
        # 编码latent
        timestep_embeds = self.time_embedder(timesteps)
        pos_embeds = self.latent_pos_embed(latent_position_ids)
        latent_embeddings = self.vae2llm(input_latent) + timestep_embeds + pos_embeds
        
        # 6. 将latent embeddings添加到序列中进行预测
        extended_sequence = torch.cat([current_sequence, latent_embeddings], dim=0)
        extended_len = extended_sequence.size(0)
        extended_position_ids = torch.full((extended_len,), position_id, device=device)
        
        # 前向传播预测
        extended_output = self.language_model(
            packed_sequence=extended_sequence.unsqueeze(0),
            sample_lens=[extended_len],
            attention_mask=None,
            packed_position_ids=extended_position_ids,
        )
        
        # 7. 计算图像损失
        pred_latent = self.llm2vae(extended_output[0, -latent.shape[0]:, :])  # 最后N个位置的预测
        target_velocity = noise - latent  # flow matching target
        
        # 只对有效时间步计算损失
        has_mse = timesteps > 0
        if has_mse.any():
            image_loss = ((pred_latent - target_velocity)[has_mse] ** 2).mean()
        else:
            # 对于clean image (timestep=0), 直接预测latent
            image_loss = ((pred_latent - latent) ** 2).mean()
        
        # 8. 更新序列（使用预测的latent）
        updated_sequence = torch.cat([current_sequence, latent_embeddings], dim=0)
        
        return image_loss, updated_sequence


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        # 获取模型所在的设备
        device = next(self.parameters()).device
        
        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int, device=device),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long, device=device),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # 数据完整性检查
        if vit_token_seqlens is None:
            raise ValueError("vit_token_seqlens不能为None")
        
        if len(vit_token_seqlens) == 0:
            raise ValueError("vit_token_seqlens不能为空")
        
        # 检查是否有负数或异常值
        min_val = torch.min(vit_token_seqlens).item()
        max_val = torch.max(vit_token_seqlens).item()
        
        if min_val < 0:
            print(f"错误：vit_token_seqlens包含负数: {vit_token_seqlens}")
            print(f"形状: {vit_token_seqlens.shape}, 数据类型: {vit_token_seqlens.dtype}")
            raise ValueError(f"vit_token_seqlens包含无效的负数值: {min_val}")

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache_args
        enable_taylorseer=False,
    ):
        if enable_taylorseer:
            self.language_model.model.enable_taylorseer = True
            model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(self, num_timesteps)
            model_pred_img_cache_dic, model_pred_img_current = cache_init(self, num_timesteps)
        else:
            self.language_model.model.enable_taylorseer = False
            model_pred_cache_dic, model_pred_current = None, None
            model_pred_text_cache_dic, model_pred_text_current = None, None
            model_pred_img_cache_dic, model_pred_img_current = None, None
    
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                # cache
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise
        
        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_current
            del model_pred_text_cache_dic, model_pred_text_current
            del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache
        model_pred_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_current: Optional[int] = None,
        model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_text_current: Optional[int] = None,
        model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_img_current: Optional[int] = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }
        
        if self.language_model.model.enable_taylorseer:
            self.language_model.model.cache_dic = model_pred_cache_dic
            self.language_model.model.current = model_pred_current

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_text_cache_dic
                self.language_model.model.current = model_pred_text_current
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_img_cache_dic
                self.language_model.model.current = model_pred_img_current
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        # 获取模型所在的设备
        device = next(self.parameters()).device
        
        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long, device=device),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output
