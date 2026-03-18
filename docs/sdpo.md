# SDPO

PRIME-RL supports a minimal first-class integration of Self-Distilled Policy Optimization (SDPO).

This implementation follows the PRIME-RL architecture:
- The orchestrator builds reprompted teacher prefixes from successful sibling rollouts and optional textual feedback.
- The teacher server scores the sampled completion under that alternate prefix.
- The trainer applies the token-level reverse-KL SDPO loss against those teacher logprobs.

## Quick Start

```toml
[deployment]
gpus_per_node = 3
num_train_gpus = 1
num_infer_gpus = 1
num_teacher_gpus = 1

[trainer.loss]
type = "sdpo"
is_clip = 2.0

[orchestrator]
batch_size = 128
rollouts_per_example = 8

[orchestrator.self_distillation]
success_reward_threshold = 0.5
feedback_key = "feedback"
```

Setting `num_teacher_gpus` is enough for the RL launcher to auto-start a teacher server and wire `orchestrator.teacher_model`.

## Feedback Sources

When `orchestrator.self_distillation.include_environment_feedback = true`, PRIME-RL looks for textual feedback in this order:

1. The rollout field named by `feedback_key`
2. The last trajectory step's `extras[feedback_key]`

If no feedback is available, SDPO still works by reusing successful sibling completions as demonstrations.

## Current Scope

The initial integration is intentionally narrow:

- Token-level SDPO loss only
- Single-turn text rollouts only
- Multi-turn or multimodal samples are skipped rather than erroring
- Teacher and student should use tokenizer-compatible models, since teacher scoring is token-based

This keeps the implementation small and aligned with PRIME-RL's existing teacher-prefill path.
