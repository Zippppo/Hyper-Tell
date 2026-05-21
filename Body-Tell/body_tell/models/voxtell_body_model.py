"""VoxTell-style prompt-conditioned model adapted for Body-Tell occupancy grids."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .transformer import TransformerDecoder, TransformerDecoderLayer


@dataclass(frozen=True)
class VoxTellBodyConfig:
    input_channels: int = 1
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256, 320)
    text_embedding_dim: int = 2560
    query_dim: int = 2048
    text_projection_hidden_dim: int = 2048
    transformer_num_heads: int = 8
    transformer_layers: int = 6
    transformer_feedforward_dim: int = 2048
    transformer_dropout: float = 0.1
    decoder_layer: int = -1
    num_maskformer_stages: int = 5
    num_heads: int = 32
    deep_supervision: bool = False


class VoxTellBodyModel(nn.Module):
    """Prompt-conditioned binary mask model using VoxTell-style dynamic heads."""

    def __init__(self, config: VoxTellBodyConfig | None = None) -> None:
        super().__init__()
        self.config = config or VoxTellBodyConfig()
        if self.config.query_dim % self.config.transformer_num_heads != 0:
            raise ValueError("query_dim must be divisible by transformer_num_heads")
        if len(self.config.encoder_channels) < 2:
            raise ValueError("encoder_channels must contain at least two stages")
        if self.config.num_maskformer_stages < 1:
            raise ValueError("num_maskformer_stages must be at least 1 for the dynamic mask head")

        self.encoder = _BodyEncoder(
            input_channels=self.config.input_channels,
            channels=self.config.encoder_channels,
        )
        selected_stage = self.config.decoder_layer
        if selected_stage < 0:
            selected_stage = len(self.config.encoder_channels) + selected_stage
        if selected_stage < 0 or selected_stage >= len(self.config.encoder_channels):
            raise ValueError("decoder_layer selects no encoder stage")
        self.selected_stage = selected_stage

        selected_channels = self.config.encoder_channels[self.selected_stage]
        self.project_visual_embed = nn.Sequential(
            nn.Linear(selected_channels, self.config.query_dim),
            nn.GELU(),
            nn.Linear(self.config.query_dim, self.config.query_dim),
        )
        self.project_text_embed = nn.Sequential(
            nn.Linear(self.config.text_embedding_dim, self.config.text_projection_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.text_projection_hidden_dim, self.config.query_dim),
        )

        layer = TransformerDecoderLayer(
            d_model=self.config.query_dim,
            nhead=self.config.transformer_num_heads,
            dim_feedforward=self.config.transformer_feedforward_dim,
            dropout=self.config.transformer_dropout,
            activation="gelu",
            normalize_before=True,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=layer,
            num_layers=self.config.transformer_layers,
            norm=nn.LayerNorm(self.config.query_dim),
        )

        self.fused_stage_count = min(
            self.config.num_maskformer_stages,
            len(self.config.encoder_channels) - 1,
        )
        self.mask_projections = nn.ModuleList()
        for stage_index, channels in enumerate(self.config.encoder_channels[: self.fused_stage_count]):
            output_dim = channels if stage_index == 0 else channels * self.config.num_heads
            self.mask_projections.append(
                nn.Sequential(
                    nn.Linear(self.config.query_dim, self.config.text_projection_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.config.text_projection_hidden_dim, output_dim),
                )
            )

        self.decoder = _VoxTellBodyDecoder(
            encoder_channels=self.config.encoder_channels,
            num_heads=self.config.num_heads,
            fused_stage_count=self.fused_stage_count,
            deep_supervision=self.config.deep_supervision,
        )

    def forward(self, occupancy: Tensor, text_embeddings: Tensor) -> Tensor | List[Tensor]:
        if text_embeddings.ndim == 4 and text_embeddings.shape[2] == 1:
            text_embeddings = text_embeddings.squeeze(2)
        if text_embeddings.ndim != 3:
            raise ValueError(
                "text_embeddings must have shape (B, N, E) or (B, N, 1, E), "
                f"got {tuple(text_embeddings.shape)}"
            )

        skips = self.encoder(occupancy)
        selected_feature = skips[self.selected_stage]
        batch_size, channels, depth, height, width = selected_feature.shape

        memory = selected_feature.permute(2, 3, 4, 0, 1).reshape(-1, batch_size, channels)
        memory = self.project_visual_embed(memory)
        pos = sinusoidal_position_encoding_3d(
            (depth, height, width),
            self.config.query_dim,
            device=memory.device,
            dtype=memory.dtype,
        )

        text_queries = text_embeddings.permute(1, 0, 2)
        text_queries = self.project_text_embed(text_queries)
        mask_embedding, _ = self.transformer_decoder(
            tgt=text_queries,
            memory=memory,
            pos=pos,
        )
        mask_embedding = mask_embedding.permute(1, 0, 2)

        stage_embeddings = [projection(mask_embedding) for projection in self.mask_projections]
        num_prompts = text_embeddings.shape[1]
        per_prompt_outputs: List[List[Tensor]] = []
        for prompt_index in range(num_prompts):
            prompt_stage_embeddings = [
                embedding[:, prompt_index : prompt_index + 1, :]
                for embedding in stage_embeddings
            ]
            per_prompt_outputs.append(self.decoder(skips, prompt_stage_embeddings))

        outputs = [
            torch.cat(scale_outputs, dim=1)
            for scale_outputs in zip(*per_prompt_outputs)
        ]
        if self.config.deep_supervision:
            return outputs
        return outputs[0]


class _BodyEncoder(nn.Module):
    def __init__(self, input_channels: int, channels: Sequence[int]) -> None:
        super().__init__()
        stages = []
        current_channels = input_channels
        for stage_index, output_channels in enumerate(channels):
            stride = 1 if stage_index == 0 else 2
            stages.append(_ConvBlock(current_channels, output_channels, stride=stride))
            current_channels = output_channels
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> List[Tensor]:
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class _VoxTellBodyDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        num_heads: int,
        fused_stage_count: int,
        deep_supervision: bool,
    ) -> None:
        super().__init__()
        self.encoder_channels = tuple(int(x) for x in encoder_channels)
        self.num_heads = int(num_heads)
        self.fused_stage_count = int(fused_stage_count)
        self.deep_supervision = bool(deep_supervision)

        upconvs = []
        blocks = []
        seg_layers = []
        current_channels = self.encoder_channels[-1]
        for target_stage in reversed(range(len(self.encoder_channels) - 1)):
            target_channels = self.encoder_channels[target_stage]
            upconvs.append(
                nn.ConvTranspose3d(
                    current_channels,
                    target_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            blocks.append(_ConvBlock(target_channels * 2, target_channels))

            uses_intermediate_fusion = self._uses_fusion(target_stage) and target_stage > 0
            if uses_intermediate_fusion:
                if self.deep_supervision:
                    seg_layers.append(nn.Conv3d(target_channels + self.num_heads, 1, kernel_size=1))
                else:
                    seg_layers.append(nn.Identity())
                current_channels = target_channels + self.num_heads
            else:
                seg_layers.append(nn.Identity())
                current_channels = target_channels

        self.upconvs = nn.ModuleList(upconvs)
        self.blocks = nn.ModuleList(blocks)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: Sequence[Tensor], mask_embeddings: Sequence[Tensor]) -> List[Tensor]:
        x = skips[-1]
        outputs = []

        for decoder_index, target_stage in enumerate(reversed(range(len(skips) - 1))):
            skip = skips[target_stage]
            x = self.upconvs[decoder_index](x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat((x, skip), dim=1)
            x = self.blocks[decoder_index](x)

            if target_stage == 0:
                mask_embedding = mask_embeddings[0]
                outputs.append(torch.einsum("b c d h w, b n c -> b n d h w", x, mask_embedding))
                continue

            if self._uses_fusion(target_stage):
                mask_embedding = mask_embeddings[target_stage]
                batch_size, _, channels = mask_embedding.shape
                mask_embedding = mask_embedding.view(batch_size, self.num_heads, channels // self.num_heads)
                fusion = torch.einsum("b c d h w, b n c -> b n d h w", x, mask_embedding)
                x = torch.cat((x, fusion), dim=1)
                if self.deep_supervision:
                    outputs.append(self.seg_layers[decoder_index](x))

        outputs = outputs[::-1]
        if not self.deep_supervision:
            return outputs[:1]
        return outputs

    def _uses_fusion(self, stage_index: int) -> bool:
        return stage_index < self.fused_stage_count


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_position_encoding_3d(
    spatial_shape: Sequence[int],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 10000.0,
) -> Tensor:
    """Return ``(D*H*W, 1, dim)`` deterministic 3D sine/cosine positions."""

    depth, height, width = (int(x) for x in spatial_shape)
    z, y, x = torch.meshgrid(
        torch.linspace(0, 1, depth, device=device, dtype=dtype),
        torch.linspace(0, 1, height, device=device, dtype=dtype),
        torch.linspace(0, 1, width, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack((z, y, x), dim=-1).reshape(-1, 3)
    num_frequencies = max(1, math.ceil(dim / 6))
    frequencies = torch.exp(
        -torch.arange(num_frequencies, device=device, dtype=dtype)
        * (math.log(temperature) / num_frequencies)
    )

    parts = []
    for axis in range(3):
        values = coords[:, axis : axis + 1] * frequencies
        parts.extend((torch.sin(values), torch.cos(values)))
    encoded = torch.cat(parts, dim=1)
    if encoded.shape[1] < dim:
        padding = torch.zeros(encoded.shape[0], dim - encoded.shape[1], device=device, dtype=dtype)
        encoded = torch.cat((encoded, padding), dim=1)
    return encoded[:, :dim].unsqueeze(1)


__all__ = ["VoxTellBodyConfig", "VoxTellBodyModel", "sinusoidal_position_encoding_3d"]
