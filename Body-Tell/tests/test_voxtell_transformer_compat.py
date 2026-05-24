from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

from body_tell.models.transformer import (
    TransformerDecoder as BodyTransformerDecoder,
    TransformerDecoderLayer as BodyTransformerDecoderLayer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VOXTELL_TRANSFORMER_PATH = REPO_ROOT / "VoxTell" / "voxtell" / "model" / "transformer.py"


def _load_voxtell_transformer_module():
    spec = importlib.util.spec_from_file_location(
        "voxtell_reference_transformer",
        VOXTELL_TRANSFORMER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load VoxTell transformer from {VOXTELL_TRANSFORMER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_voxtell_reference_decoder():
    reference = _load_voxtell_transformer_module()
    layer = reference.TransformerDecoderLayer(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        activation="relu",
        normalize_before=True,
    )
    return reference.TransformerDecoder(layer, num_layers=2, norm=nn.LayerNorm(16))


def _build_body_decoder():
    layer = BodyTransformerDecoderLayer(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        activation="relu",
        normalize_before=True,
    )
    return BodyTransformerDecoder(layer, num_layers=2, norm=nn.LayerNorm(16))


def _golden_inputs():
    generator = torch.Generator().manual_seed(20260524)
    return {
        "tgt": torch.randn(3, 2, 16, generator=generator),
        "memory": torch.randn(5, 2, 16, generator=generator),
        "pos": torch.randn(5, 2, 16, generator=generator),
        "query_pos": torch.randn(3, 2, 16, generator=generator),
    }


def test_body_transformer_matches_voxtell_reference_with_shared_weights() -> None:
    torch.manual_seed(0)
    voxtell_decoder = _build_voxtell_reference_decoder()
    body_decoder = _build_body_decoder()
    body_decoder.load_state_dict(voxtell_decoder.state_dict())
    voxtell_decoder.eval()
    body_decoder.eval()

    inputs = _golden_inputs()
    with torch.no_grad():
        expected, _ = voxtell_decoder(**inputs)
        actual, _ = body_decoder(**inputs)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)
