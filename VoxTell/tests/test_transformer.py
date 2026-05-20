import torch

from voxtell.model.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    _get_activation_fn,
    _get_clones,
)


def test_transformer_decoder_layer_preserves_target_shape_in_pre_norm_mode():
    layer = TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, normalize_before=True)
    target = torch.randn(3, 2, 8)
    memory = torch.randn(5, 2, 8)
    position = torch.randn(5, 2, 8)

    output, attention = layer(target, memory, pos=position)

    assert output.shape == target.shape
    assert attention.shape == (2, 3, 5)


def test_transformer_decoder_can_return_intermediate_outputs():
    layer = TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, normalize_before=True)
    decoder = TransformerDecoder(
        decoder_layer=layer,
        num_layers=2,
        norm=torch.nn.LayerNorm(8),
        return_intermediate=True,
    )
    target = torch.randn(3, 2, 8)
    memory = torch.randn(5, 2, 8)

    output = decoder(target, memory)

    assert output.shape == (2, 3, 2, 8)


def test_get_clones_returns_independent_modules():
    module = torch.nn.Linear(2, 2)
    clones = _get_clones(module, 2)

    with torch.no_grad():
        clones[0].weight.fill_(1)
        clones[1].weight.fill_(2)

    assert not torch.equal(clones[0].weight, clones[1].weight)


def test_get_activation_fn_rejects_unknown_names():
    assert _get_activation_fn("relu") is torch.nn.functional.relu

    try:
        _get_activation_fn("unknown")
    except RuntimeError as exc:
        assert "activation should be" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unknown activation")
