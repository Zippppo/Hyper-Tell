"""VoxTell-style prompt-conditioned model adapted for Body-Tell occupancy grids."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from .transformer import TransformerDecoder, TransformerDecoderLayer


VOXTELL_V1_1_ENCODER_CHANNELS: Tuple[int, ...] = (32, 64, 128, 256, 320, 320)
VOXTELL_V1_1_N_BLOCKS_PER_STAGE: Tuple[int, ...] = (1, 3, 4, 6, 6, 6)


@dataclass(frozen=True)
class VoxTellBodyConfig:
    input_channels: int = 1
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256, 320)
    backbone: str = "conv"
    n_blocks_per_stage: Tuple[int, ...] | None = None
    encoder_conv_bias: bool = True
    encoder_norm: str = "instance_norm_3d"
    encoder_activation: str = "leaky_relu"
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
        if self.config.num_heads < 1:
            raise ValueError("num_heads must be at least 1")

        encoder_channels = tuple(int(channel) for channel in self.config.encoder_channels)
        self.encoder = self._build_encoder(encoder_channels)
        selected_stage = self.config.decoder_layer
        if selected_stage < 0:
            selected_stage = len(encoder_channels) + selected_stage
        if selected_stage < 0 or selected_stage >= len(encoder_channels):
            raise ValueError("decoder_layer selects no encoder stage")
        self.selected_stage = selected_stage

        selected_channels = encoder_channels[self.selected_stage]
        self.project_bottleneck_embed = nn.Sequential(
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
            activation="relu",
            normalize_before=True,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=layer,
            num_layers=self.config.transformer_layers,
            norm=nn.LayerNorm(self.config.query_dim),
        )

        self.fused_stage_count = min(
            self.config.num_maskformer_stages,
            len(encoder_channels) - 1,
        )
        self.project_to_decoder_channels = nn.ModuleList()
        for stage_index, channels in enumerate(encoder_channels[: self.fused_stage_count]):
            output_dim = channels if stage_index == 0 else channels * self.config.num_heads
            self.project_to_decoder_channels.append(
                nn.Sequential(
                    nn.Linear(self.config.query_dim, self.config.text_projection_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.config.text_projection_hidden_dim, output_dim),
                )
            )

        self.decoder = VoxTellDecoder(
            encoder=self.encoder,
            num_classes=1,
            n_conv_per_stage=[1] * (len(encoder_channels) - 1),
            deep_supervision=self.config.deep_supervision,
            num_maskformer_stages=self.fused_stage_count,
            num_heads=self.config.num_heads,
        )

    def _build_encoder(self, encoder_channels: Tuple[int, ...]) -> nn.Module:
        backbone = self.config.backbone.lower().replace("-", "_")
        if backbone in {"conv", "body_conv", "legacy_conv"}:
            return _BodyEncoder(
                input_channels=self.config.input_channels,
                channels=encoder_channels,
            )
        if backbone in {"residual_encoder", "voxtell_residual_encoder"}:
            return _build_voxtell_residual_encoder(
                input_channels=self.config.input_channels,
                channels=encoder_channels,
                n_blocks_per_stage=_resolve_n_blocks_per_stage(
                    encoder_channels,
                    self.config.n_blocks_per_stage,
                ),
                conv_bias=self.config.encoder_conv_bias,
                norm=self.config.encoder_norm,
                activation=self.config.encoder_activation,
            )
        raise ValueError(
            "backbone must be one of 'conv' or 'residual_encoder', "
            f"got {self.config.backbone!r}"
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

        bottleneck_embed = selected_feature.permute(0, 3, 4, 2, 1)
        bottleneck_embed = self.project_bottleneck_embed(bottleneck_embed)
        memory = bottleneck_embed.permute(1, 2, 3, 0, 4).reshape(
            -1,
            batch_size,
            self.config.query_dim,
        )
        pos = sinusoidal_position_encoding_3d(
            (height, width, depth),
            self.config.query_dim,
            device=memory.device,
            dtype=memory.dtype,
            batch_size=batch_size,
        )
        if (
            pos.shape != memory.shape
            or pos.dtype != memory.dtype
            or pos.device != memory.device
        ):
            raise RuntimeError(
                "runtime position encoding contract violated: "
                f"memory shape={tuple(memory.shape)} dtype={memory.dtype} device={memory.device}; "
                f"pos shape={tuple(pos.shape)} dtype={pos.dtype} device={pos.device}"
            )

        text_queries = text_embeddings.permute(1, 0, 2)
        text_queries = self.project_text_embed(text_queries)
        mask_embedding, _ = self.transformer_decoder(
            tgt=text_queries,
            memory=memory,
            pos=pos,
        )
        mask_embedding = mask_embedding.permute(1, 0, 2)

        stage_embeddings = [
            projection(mask_embedding)
            for projection in self.project_to_decoder_channels
        ]
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
        self.output_channels = tuple(int(channel) for channel in channels)
        self.conv_op = nn.Conv3d
        self.conv_bias = False
        self.norm_op = _BodyDecoderNorm
        self.norm_op_kwargs = {}
        self.dropout_op = None
        self.dropout_op_kwargs = None
        self.nonlin = nn.GELU
        self.nonlin_kwargs = {}
        self.kernel_sizes = tuple((3, 3, 3) for _ in channels)
        self.strides = tuple(
            (1, 1, 1) if index == 0 else (2, 2, 2)
            for index in range(len(channels))
        )
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


class _BodyDecoderNorm(nn.GroupNorm):
    def __init__(self, num_channels: int) -> None:
        super().__init__(_num_groups(num_channels), num_channels)


def _build_voxtell_residual_encoder(
    input_channels: int,
    channels: Sequence[int],
    n_blocks_per_stage: Sequence[int],
    conv_bias: bool,
    norm: str,
    activation: str,
) -> nn.Module:
    try:
        from dynamic_network_architectures.building_blocks.residual import BasicBlockD
        from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
    except ImportError as exc:
        raise RuntimeError(
            "backbone='residual_encoder' requires dynamic_network_architectures"
        ) from exc

    output_channels = tuple(int(channel) for channel in channels)
    blocks_per_stage = tuple(int(blocks) for blocks in n_blocks_per_stage)
    kernel_sizes = tuple((3, 3, 3) for _ in output_channels)
    strides = tuple(
        (1, 1, 1) if index == 0 else (2, 2, 2)
        for index in range(len(output_channels))
    )

    encoder = ResidualEncoder(
        input_channels=input_channels,
        n_stages=len(output_channels),
        features_per_stage=output_channels,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_blocks_per_stage=blocks_per_stage,
        conv_bias=bool(conv_bias),
        norm_op=_resolve_residual_norm(norm),
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=_resolve_residual_activation(activation),
        nonlin_kwargs={"inplace": True},
        block=BasicBlockD,
        bottleneck_channels=None,
        return_skips=True,
        disable_default_stem=False,
        stem_channels=None,
    )
    encoder.output_channels = tuple(int(channel) for channel in encoder.output_channels)
    encoder.strides = tuple(
        tuple(int(value) for value in stride)
        for stride in encoder.strides
    )
    encoder.n_blocks_per_stage = blocks_per_stage
    encoder.conv_bias = bool(conv_bias)
    encoder.kernel_sizes = kernel_sizes
    encoder.norm = norm
    encoder.activation = activation
    return encoder


class VoxTellDecoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        n_conv_per_stage: int | Sequence[int],
        deep_supervision: bool,
        num_maskformer_stages: int = 5,
        nonlin_first: bool = False,
        norm_op: type[nn.Module] | None = None,
        norm_op_kwargs: dict | None = None,
        dropout_op: type[nn.Module] | None = None,
        dropout_op_kwargs: dict | None = None,
        nonlin: type[nn.Module] | None = None,
        nonlin_kwargs: dict | None = None,
        conv_bias: bool | None = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.encoder = encoder
        self.num_classes = int(num_classes)
        self.num_heads = int(num_heads)

        try:
            from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
            from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
        except ImportError as exc:
            raise RuntimeError("VoxTellDecoder requires dynamic_network_architectures") from exc

        encoder_channels = tuple(int(channel) for channel in encoder.output_channels)
        n_stages_encoder = len(encoder_channels)
        if n_stages_encoder < 2:
            raise ValueError("VoxTellDecoder requires at least two encoder stages")

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [int(n_conv_per_stage)] * (n_stages_encoder - 1)
        else:
            n_conv_per_stage = [int(value) for value in n_conv_per_stage]
        if len(n_conv_per_stage) != n_stages_encoder - 1:
            raise ValueError(
                "n_conv_per_stage must have one less entry than encoder stages: "
                f"expected {n_stages_encoder - 1}, got {len(n_conv_per_stage)}"
            )

        self.num_maskformer_stages = min(int(num_maskformer_stages), n_stages_encoder - 1)
        if self.num_maskformer_stages < 1:
            raise ValueError("num_maskformer_stages must select at least one decoder output")

        conv_op = encoder.conv_op
        transpconv_op = get_matching_convtransp(conv_op=conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else bool(conv_bias)
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = (
            encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        )
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs
        norm_op_kwargs = {} if norm_op_kwargs is None else dict(norm_op_kwargs)
        dropout_op_kwargs = {} if dropout_op_kwargs is None else dict(dropout_op_kwargs)
        nonlin_kwargs = {} if nonlin_kwargs is None else dict(nonlin_kwargs)

        stages = []
        transpconvs = []
        seg_layers = []
        for stage_idx in range(1, n_stages_encoder):
            if stage_idx <= n_stages_encoder - self.num_maskformer_stages:
                input_features_below = encoder_channels[-stage_idx]
            else:
                input_features_below = encoder_channels[-stage_idx] + self.num_heads

            input_features_skip = encoder_channels[-(stage_idx + 1)]
            stride_for_transpconv = encoder.strides[-stage_idx]

            transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    stride_for_transpconv,
                    stride_for_transpconv,
                    bias=conv_bias,
                )
            )
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[stage_idx - 1],
                    conv_op,
                    2 * input_features_skip,
                    input_features_skip,
                    encoder.kernel_sizes[-(stage_idx + 1)],
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            seg_layers.append(
                conv_op(
                    input_features_skip + self.num_heads,
                    self.num_classes,
                    1,
                    1,
                    0,
                    bias=True,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: Sequence[Tensor], mask_embeddings: Sequence[Tensor]) -> List[Tensor]:
        if len(mask_embeddings) != self.num_maskformer_stages:
            raise ValueError(
                "mask_embeddings count must match fused decoder stages: "
                f"expected {self.num_maskformer_stages}, got {len(mask_embeddings)}"
            )

        lres_input = skips[-1]
        outputs: List[Tensor] = []
        stage_mask_embeddings = list(mask_embeddings)[::-1]

        for stage_idx in range(len(self.stages)):
            x = self.transpconvs[stage_idx](lres_input)
            skip = skips[-(stage_idx + 2)]
            if x.shape[2:] != skip.shape[2:]:
                raise ValueError(
                    "VoxTellDecoder requires stride-compatible input sizes; "
                    f"stage {stage_idx} transposed-conv output spatial {tuple(x.shape[2:])} "
                    f"does not match skip spatial {tuple(skip.shape[2:])}"
                )
            x = torch.cat((x, skip), dim=1)
            x = self.stages[stage_idx](x)

            if stage_idx == len(self.stages) - 1:
                outputs.append(
                    torch.einsum("b c d h w, b n c -> b n d h w", x, stage_mask_embeddings[-1])
                )
            elif stage_idx >= len(self.stages) - len(stage_mask_embeddings):
                mask_embedding = stage_mask_embeddings.pop(0)
                batch_size, _, channels = mask_embedding.shape
                if channels % self.num_heads != 0:
                    raise ValueError(
                        "mask embedding channels must be divisible by num_heads: "
                        f"got {channels} channels and num_heads={self.num_heads}"
                    )
                mask_embedding = mask_embedding.view(batch_size, self.num_heads, -1)
                fusion = torch.einsum("b c d h w, b n c -> b n d h w", x, mask_embedding)
                x = torch.cat((x, fusion), dim=1)
                outputs.append(self.seg_layers[stage_idx](x))

            lres_input = x

        outputs = outputs[::-1]
        if not self.deep_supervision:
            return outputs[:1]
        return outputs


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


def _resolve_n_blocks_per_stage(
    encoder_channels: Tuple[int, ...],
    n_blocks_per_stage: Sequence[int] | None,
) -> Tuple[int, ...]:
    if n_blocks_per_stage is None:
        if encoder_channels == VOXTELL_V1_1_ENCODER_CHANNELS:
            return VOXTELL_V1_1_N_BLOCKS_PER_STAGE
        return tuple(1 for _ in encoder_channels)

    blocks = tuple(int(blocks) for blocks in n_blocks_per_stage)
    if len(blocks) != len(encoder_channels):
        raise ValueError(
            "n_blocks_per_stage must have one entry per encoder stage: "
            f"got {len(blocks)} blocks for {len(encoder_channels)} stages"
        )
    return blocks


def _resolve_residual_norm(norm: str) -> type[nn.Module]:
    normalized = norm.lower().replace("-", "_")
    if normalized in {"instance", "instance_norm", "instance_norm_3d", "instancenorm3d"}:
        return nn.InstanceNorm3d
    raise ValueError(
        "residual_encoder uses the VoxTell norm policy and only supports "
        f"InstanceNorm3d, got {norm!r}"
    )


def _resolve_residual_activation(activation: str) -> type[nn.Module]:
    normalized = activation.lower().replace("-", "_")
    if normalized in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU
    raise ValueError(
        "residual_encoder uses the VoxTell activation policy and only supports "
        f"LeakyReLU, got {activation!r}"
    )


def sinusoidal_position_encoding_3d(
    spatial_shape: Sequence[int],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
    temperature: float = 10000.0,
) -> Tensor:
    """Return ``(H*W*D, batch_size, dim)`` positions in VoxTell flatten order."""

    height, width, depth = (int(x) for x in spatial_shape)
    y, x, z = torch.meshgrid(
        torch.linspace(0, 1, height, device=device, dtype=dtype),
        torch.linspace(0, 1, width, device=device, dtype=dtype),
        torch.linspace(0, 1, depth, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack((y, x, z), dim=-1).reshape(-1, 3)
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
    encoded = encoded[:, :dim].to(device=device, dtype=dtype)
    return encoded.unsqueeze(1).expand(-1, int(batch_size), -1)


def load_voxtell_transformer_decoder_prefix(
    model: VoxTellBodyModel,
    voxtell_path: str | Path,
    prefix: str = "transformer_decoder.",
) -> dict[str, Any]:
    """Load only VoxTell ``transformer_decoder.*`` tensors into ``model``."""

    if prefix != "transformer_decoder.":
        raise ValueError("P6-lite only supports loading the transformer_decoder. prefix")

    checkpoint_path = _resolve_voxtell_checkpoint_path(voxtell_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VoxTell checkpoint does not exist: {checkpoint_path}")

    checkpoint = _load_checkpoint_cpu(checkpoint_path)
    if not isinstance(checkpoint, Mapping) or "network_weights" not in checkpoint:
        raise KeyError(
            "VoxTell checkpoint must contain a 'network_weights' state dict; "
            "full-checkpoint fallback loading is intentionally not supported"
        )
    network_weights = checkpoint["network_weights"]
    if not isinstance(network_weights, Mapping):
        raise TypeError("VoxTell checkpoint field 'network_weights' must be a mapping")

    source_state = {
        key[len(prefix) :]: tensor
        for key, tensor in network_weights.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    target_state = model.transformer_decoder.state_dict()

    missing_keys = sorted(set(target_state) - set(source_state))
    unexpected_keys = sorted(set(source_state) - set(target_state))
    shape_mismatches = [
        {
            "key": key,
            "checkpoint_shape": tuple(int(value) for value in source_state[key].shape),
            "model_shape": tuple(int(value) for value in target_state[key].shape),
        }
        for key in sorted(set(source_state) & set(target_state))
        if tuple(source_state[key].shape) != tuple(target_state[key].shape)
    ]

    metadata: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "prefix": prefix,
        "loaded_tensor_count": len(source_state),
        "loaded_parameter_count": sum(int(tensor.numel()) for tensor in source_state.values()),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "excluded_prefixes": [
            "encoder.",
            "decoder.",
            "project_bottleneck_embed.",
            "project_text_embed.",
            "project_to_decoder_channels.",
            "pos_embed",
        ],
    }

    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            "VoxTell transformer decoder prefix is not load-compatible: "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}, "
            f"shape_mismatches={len(shape_mismatches)}"
        )

    model.transformer_decoder.load_state_dict(source_state, strict=True)
    return metadata


def _resolve_voxtell_checkpoint_path(voxtell_path: str | Path) -> Path:
    path = Path(voxtell_path)
    if path.is_dir():
        return path / "fold_0" / "checkpoint_final.pth"
    return path


def _load_checkpoint_cpu(checkpoint_path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )


__all__ = [
    "VOXTELL_V1_1_ENCODER_CHANNELS",
    "VOXTELL_V1_1_N_BLOCKS_PER_STAGE",
    "VoxTellBodyConfig",
    "VoxTellDecoder",
    "VoxTellBodyModel",
    "load_voxtell_transformer_decoder_prefix",
    "sinusoidal_position_encoding_3d",
]
