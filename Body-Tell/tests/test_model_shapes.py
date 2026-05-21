from __future__ import annotations

import torch

from body_tell.models.voxtell_body_model import VoxTellBodyConfig, VoxTellBodyModel


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

