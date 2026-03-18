import torch

from prime_rl.configs.trainer import SDPOLossConfig
from prime_rl.trainer.rl.loss import LossInputs, sdpo_loss_fn


def test_sdpo_loss_matches_reverse_kl_surrogate():
    inputs = LossInputs(
        trainer_logprobs=torch.tensor([-1.0, -0.5]),
        inference_logprobs=torch.tensor([-1.2, -0.7]),
        teacher_logprobs=torch.tensor([-0.8, -0.2]),
        advantages=torch.tensor([0.0, 0.0]),
        loss_mask=torch.tensor([True, False]),
    )

    outputs = sdpo_loss_fn(inputs, SDPOLossConfig(is_clip=None))

    assert torch.isclose(outputs.loss, torch.tensor(0.2))
    assert torch.isclose(outputs.metrics["teacher_kl"], torch.tensor(0.2))
    assert torch.isclose(outputs.metrics["sdpo_is_clipped"], torch.tensor(0.0))


def test_sdpo_loss_applies_is_clip():
    inputs = LossInputs(
        trainer_logprobs=torch.tensor([-1.0]),
        inference_logprobs=torch.tensor([-2.0]),
        teacher_logprobs=torch.tensor([-0.8]),
        advantages=torch.tensor([0.0]),
        loss_mask=torch.tensor([True]),
    )

    outputs = sdpo_loss_fn(inputs, SDPOLossConfig(is_clip=1.1))

    assert torch.isclose(outputs.loss, torch.tensor(0.22), atol=1e-6)
    assert torch.isclose(outputs.metrics["sdpo_is_clipped"], torch.tensor(1.0))


def test_sdpo_loss_returns_zero_for_empty_target_mask():
    inputs = LossInputs(
        trainer_logprobs=torch.tensor([-1.0]),
        inference_logprobs=torch.tensor([-1.0]),
        teacher_logprobs=torch.tensor([-1.0]),
        advantages=torch.tensor([0.0]),
        loss_mask=torch.tensor([False]),
    )

    outputs = sdpo_loss_fn(inputs, SDPOLossConfig())

    assert torch.isclose(outputs.loss, torch.tensor(0.0))
    assert torch.isclose(outputs.metrics["sdpo_target_tokens"], torch.tensor(0.0))
