import torch

from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction


def test_last_token_pool_uses_last_column_for_left_padded_batches():
    hidden_states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ]
    )

    pooled = last_token_pool(hidden_states, attention_mask)

    assert torch.equal(pooled, hidden_states[:, -1])


def test_last_token_pool_uses_sequence_lengths_for_right_padded_batches():
    hidden_states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ]
    )

    pooled = last_token_pool(hidden_states, attention_mask)

    expected = torch.stack((hidden_states[0, 2], hidden_states[1, 1]))
    assert torch.equal(pooled, expected)


def test_wrap_with_instruction_preserves_prompt_order_and_text():
    wrapped = wrap_with_instruction(["liver", "right kidney"])

    assert [item.split("Query: ", 1)[1] for item in wrapped] == ["liver", "right kidney"]
    assert all(item.startswith("Instruct: Given an anatomical term query") for item in wrapped)
