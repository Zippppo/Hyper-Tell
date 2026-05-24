from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from body_tell.models.voxtell_body_model import (
    VOXTELL_V1_1_ENCODER_CHANNELS,
    VOXTELL_V1_1_N_BLOCKS_PER_STAGE,
    VoxTellBodyConfig,
    VoxTellBodyModel,
)


def test_voxtell_body_model_outputs_prompt_logits_shape() -> None:
    torch.manual_seed(0)
    model = VoxTellBodyModel(
        VoxTellBodyConfig(
            input_channels=1,
            encoder_channels=(4, 8, 16),
            text_embedding_dim=8,
            query_dim=16,
            transformer_num_heads=4,
            transformer_layers=1,
            num_maskformer_stages=3,
            num_heads=2,
            deep_supervision=False,
        )
    )

    occupancy = torch.randn(2, 1, 16, 16, 16)
    text_embeddings = torch.randn(2, 3, 8)

    logits = model(occupancy, text_embeddings)

    assert logits.shape == (2, 3, 16, 16, 16)
    assert logits.dtype == occupancy.dtype
    state_dict = model.state_dict()
    assert "project_bottleneck_embed.0.weight" in state_dict
    assert "project_to_decoder_channels.0.2.weight" in state_dict
    assert not any(key.startswith("project_visual_embed.") for key in state_dict)
    assert not any(key.startswith("mask_projections.") for key in state_dict)


def test_voxtell_body_model_supports_deep_supervision() -> None:
    torch.manual_seed(0)
    model = VoxTellBodyModel(
        VoxTellBodyConfig(
            input_channels=1,
            encoder_channels=(4, 8, 16),
            text_embedding_dim=8,
            query_dim=16,
            transformer_num_heads=4,
            transformer_layers=1,
            num_maskformer_stages=3,
            num_heads=2,
            deep_supervision=True,
        )
    )

    occupancy = torch.randn(1, 1, 16, 16, 16)
    text_embeddings = torch.randn(1, 2, 8)

    logits = model(occupancy, text_embeddings)

    assert isinstance(logits, list)
    assert [tuple(item.shape) for item in logits] == [
        (1, 2, 16, 16, 16),
        (1, 2, 8, 8, 8),
    ]


def test_voxtell_body_model_returns_five_aligned_decoder_outputs() -> None:
    torch.manual_seed(0)
    model = VoxTellBodyModel(
        VoxTellBodyConfig(
            input_channels=1,
            encoder_channels=(4, 8, 16, 16, 16, 16),
            text_embedding_dim=8,
            query_dim=16,
            text_projection_hidden_dim=16,
            transformer_num_heads=4,
            transformer_layers=1,
            transformer_feedforward_dim=32,
            num_maskformer_stages=5,
            num_heads=2,
            deep_supervision=True,
        )
    )

    assert len(model.decoder.stages) == 5
    assert len(model.decoder.transpconvs) == 5
    assert len(model.decoder.seg_layers) == 5
    assert len(model.project_to_decoder_channels) == 5
    assert all(isinstance(layer, nn.Conv3d) for layer in model.decoder.seg_layers)
    assert [
        projection[-1].out_features
        for projection in model.project_to_decoder_channels
    ] == [4, 16, 32, 32, 32]

    occupancy = torch.randn(1, 1, 32, 32, 32)
    text_embeddings = torch.randn(1, 2, 8)

    logits = model(occupancy, text_embeddings)

    assert isinstance(logits, list)
    assert [tuple(item.shape) for item in logits] == [
        (1, 2, 32, 32, 32),
        (1, 2, 16, 16, 16),
        (1, 2, 8, 8, 8),
        (1, 2, 4, 4, 4),
        (1, 2, 2, 2, 2),
    ]
    state_dict = model.state_dict()
    assert "decoder.transpconvs.0.weight" in state_dict
    assert "decoder.stages.0.convs.0.conv.weight" in state_dict
    assert all(f"decoder.seg_layers.{idx}.weight" in state_dict for idx in range(5))
    assert any(key.startswith("decoder.encoder.") for key in state_dict)


def test_voxtell_body_model_preserves_non_cubic_dhw_axes() -> None:
    torch.manual_seed(0)
    model = VoxTellBodyModel(
        VoxTellBodyConfig(
            input_channels=1,
            encoder_channels=(2, 4, 4, 4, 4, 4),
            text_embedding_dim=8,
            query_dim=8,
            text_projection_hidden_dim=8,
            transformer_num_heads=2,
            transformer_layers=1,
            transformer_feedforward_dim=16,
            num_maskformer_stages=5,
            num_heads=2,
            deep_supervision=True,
        )
    )
    model.eval()

    projected_bottleneck_shapes: list[tuple[int, ...]] = []

    def capture_bottleneck_input(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        projected_bottleneck_shapes.append(tuple(inputs[0].shape))

    hook = model.project_bottleneck_embed.register_forward_hook(capture_bottleneck_input)
    try:
        occupancy = torch.randn(1, 1, 32, 64, 96)
        text_embeddings = torch.randn(1, 2, 8)

        with torch.no_grad():
            logits = model(occupancy, text_embeddings)
    finally:
        hook.remove()

    assert projected_bottleneck_shapes == [(1, 2, 3, 1, 4)]
    assert isinstance(logits, list)
    assert [tuple(item.shape) for item in logits] == [
        (1, 2, 32, 64, 96),
        (1, 2, 16, 32, 48),
        (1, 2, 8, 16, 24),
        (1, 2, 4, 8, 12),
        (1, 2, 2, 4, 6),
    ]


def test_voxtell_decoder_rejects_stride_mismatched_inputs() -> None:
    torch.manual_seed(0)
    model = VoxTellBodyModel(
        VoxTellBodyConfig(
            input_channels=1,
            encoder_channels=(4, 8, 16),
            text_embedding_dim=8,
            query_dim=16,
            text_projection_hidden_dim=16,
            transformer_num_heads=4,
            transformer_layers=1,
            transformer_feedforward_dim=32,
            num_maskformer_stages=3,
            num_heads=2,
            deep_supervision=False,
        )
    )

    occupancy = torch.randn(1, 1, 15, 16, 16)
    text_embeddings = torch.randn(1, 2, 8)

    with pytest.raises(ValueError, match="stride-compatible input sizes"):
        model(occupancy, text_embeddings)


def test_voxtell_body_model_supports_residual_encoder_policy() -> None:
    torch.manual_seed(0)
    model = VoxTellBodyModel(
        VoxTellBodyConfig(
            input_channels=1,
            encoder_channels=(4, 8, 16),
            backbone="residual_encoder",
            n_blocks_per_stage=(1, 2, 3),
            text_embedding_dim=8,
            query_dim=16,
            transformer_num_heads=4,
            transformer_layers=1,
            num_maskformer_stages=3,
            num_heads=2,
            deep_supervision=False,
        )
    )

    assert model.encoder.output_channels == (4, 8, 16)
    assert model.encoder.strides == ((1, 1, 1), (2, 2, 2), (2, 2, 2))
    assert model.encoder.n_blocks_per_stage == (1, 2, 3)
    assert [len(stage.blocks) for stage in model.encoder.stages] == [1, 2, 3]

    norms = [
        module
        for module in model.encoder.modules()
        if isinstance(module, nn.InstanceNorm3d)
    ]
    assert norms
    assert all(norm.eps == 1e-5 and norm.affine for norm in norms)

    activations = [
        module
        for module in model.encoder.modules()
        if isinstance(module, nn.LeakyReLU)
    ]
    assert activations
    assert all(activation.inplace for activation in activations)

    main_convs = [
        module
        for name, module in model.encoder.named_modules()
        if isinstance(module, nn.Conv3d)
        and (name == "stem.convs.0.conv" or name.endswith(("conv1.conv", "conv2.conv")))
    ]
    assert main_convs
    assert all(conv.bias is not None for conv in main_convs)

    occupancy = torch.randn(1, 1, 16, 16, 16)
    text_embeddings = torch.randn(1, 2, 8)

    logits = model(occupancy, text_embeddings)

    assert logits.shape == (1, 2, 16, 16, 16)
    assert any(name.startswith("encoder.stem.") for name in model.state_dict())
    assert not any(name.startswith("encoder.encoder.") for name in model.state_dict())


def test_voxtell_aligned_config_declares_sp_a_encoder_plan() -> None:
    cfg = yaml.safe_load(
        Path("Body-Tell/configs/phase1_voxtell_aligned.yaml").read_text()
    )
    model_cfg = cfg["model"]

    assert model_cfg["backbone"] == "residual_encoder"
    assert tuple(model_cfg["encoder_channels"]) == VOXTELL_V1_1_ENCODER_CHANNELS
    assert tuple(model_cfg["n_blocks_per_stage"]) == VOXTELL_V1_1_N_BLOCKS_PER_STAGE
    assert model_cfg["encoder_norm"] == "instance_norm_3d"
    assert model_cfg["encoder_activation"] == "leaky_relu"
    assert model_cfg["encoder_conv_bias"] is True
    assert model_cfg["decoder_layer"] == 4
    assert model_cfg["num_maskformer_stages"] == 5
    assert model_cfg["num_heads"] == 32

    expected_path_b_projection_dims = [
        channel if stage == 0 else channel * model_cfg["num_heads"]
        for stage, channel in enumerate(model_cfg["encoder_channels"][:-1])
    ]
    assert expected_path_b_projection_dims == [32, 2048, 4096, 8192, 10240]


def test_voxtell_body_model_retains_decoder_seg_layers_without_deep_supervision() -> None:
    torch.manual_seed(0)
    model = VoxTellBodyModel(
        VoxTellBodyConfig(
            input_channels=1,
            encoder_channels=(4, 8, 16),
            text_embedding_dim=8,
            query_dim=16,
            transformer_num_heads=4,
            transformer_layers=1,
            num_maskformer_stages=3,
            num_heads=2,
            deep_supervision=False,
        )
    )

    assert len(model.project_to_decoder_channels) == 2
    assert len(model.decoder.seg_layers) == 2
    assert all(isinstance(layer, nn.Conv3d) for layer in model.decoder.seg_layers)
    assert [
        name
        for name, parameter in model.named_parameters()
        if name.startswith("decoder.seg_layers.") and parameter.requires_grad
    ]

    occupancy = torch.randn(1, 1, 16, 16, 16)
    text_embeddings = torch.randn(1, 2, 8)
    logits = model(occupancy, text_embeddings)
    logits.square().mean().backward()

    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == [
        "transformer_decoder.layers.0.self_attn.in_proj_weight",
        "transformer_decoder.layers.0.self_attn.in_proj_bias",
        "transformer_decoder.layers.0.self_attn.out_proj.weight",
        "transformer_decoder.layers.0.self_attn.out_proj.bias",
        "transformer_decoder.layers.0.norm1.weight",
        "transformer_decoder.layers.0.norm1.bias",
        "decoder.seg_layers.0.weight",
        "decoder.seg_layers.0.bias",
        "decoder.seg_layers.1.weight",
        "decoder.seg_layers.1.bias",
    ]
