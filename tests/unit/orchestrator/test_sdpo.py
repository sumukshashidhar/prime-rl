from prime_rl.configs.orchestrator import SelfDistillationConfig
from prime_rl.orchestrator.sdpo import SDPOSampleContext, build_sdpo_teacher_requests, extract_feedback
from prime_rl.transport.types import TrainingSample
from transformers.tokenization_utils_base import BatchEncoding


class FakeTokenizer:
    def __init__(self, return_batch_encoding: bool = False):
        self.applied_prompts: list[str] = []
        self.return_batch_encoding = return_batch_encoding

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(64 + token_id) for token_id in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, truncation, max_length):
        rendered = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        self.applied_prompts.append(rendered)
        token_ids = [900 + idx for idx in range(min(len(rendered), max_length))]
        if self.return_batch_encoding: return BatchEncoding({"input_ids": token_ids})
        return token_ids


def _make_sample(completion_ids, completion_mask=None):
    completion_mask = [True] * len(completion_ids) if completion_mask is None else completion_mask
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=[-0.1] * len(completion_ids),
        completion_temperatures=[1.0] * len(completion_ids),
        advantage=1.0,
    )


def _make_context(example_id, reward, feedback=None, num_turns=1, is_multimodal=False):
    return SDPOSampleContext(
        example_id=example_id,
        prompt=[{"role": "system", "content": "Follow the rules."}, {"role": "user", "content": "Solve 1+1"}],
        reward=reward,
        feedback=feedback,
        num_turns=num_turns,
        is_multimodal=is_multimodal,
    )


def test_build_sdpo_teacher_requests_uses_successful_peer():
    tokenizer = FakeTokenizer()
    config = SelfDistillationConfig()
    samples = [_make_sample([3, 4]), _make_sample([5, 6])]
    contexts = [_make_context(example_id=7, reward=1.0), _make_context(example_id=7, reward=0.0, feedback="Too short")]

    requests, metrics = build_sdpo_teacher_requests(samples=samples, contexts=contexts, tokenizer=tokenizer, config=config)

    assert len(requests) == 1
    assert requests[0].sample_index == 1
    assert requests[0].tokens[-2:] == [5, 6]
    assert samples[0].completion_mask == [False, False]
    assert samples[1].completion_mask == [True, True]
    assert "Correct solution:" in tokenizer.applied_prompts[0]
    assert "The following is feedback" not in tokenizer.applied_prompts[0]
    assert metrics["self_distillation/reprompt_sample_fraction"] == 0.5


def test_build_sdpo_teacher_requests_uses_feedback_without_solution():
    tokenizer = FakeTokenizer()
    config = SelfDistillationConfig()
    samples = [_make_sample([7, 8])]
    contexts = [_make_context(example_id=11, reward=0.0, feedback="The answer format is invalid.")]

    requests, metrics = build_sdpo_teacher_requests(samples=samples, contexts=contexts, tokenizer=tokenizer, config=config)

    assert len(requests) == 1
    assert requests[0].sample_index == 0
    assert "The following is feedback" in tokenizer.applied_prompts[0]
    assert "The answer format is invalid." in tokenizer.applied_prompts[0]
    assert metrics["self_distillation/feedback_used_fraction"] == 1.0


def test_build_sdpo_teacher_requests_accepts_batch_encoding_prefix_ids():
    tokenizer = FakeTokenizer(return_batch_encoding=True)
    config = SelfDistillationConfig()
    samples = [_make_sample([7, 8])]
    contexts = [_make_context(example_id=11, reward=0.0, feedback="The answer format is invalid.")]

    requests, _ = build_sdpo_teacher_requests(samples=samples, contexts=contexts, tokenizer=tokenizer, config=config)

    assert len(requests) == 1
    assert requests[0].tokens[-2:] == [7, 8]


def test_build_sdpo_teacher_requests_disables_multiturn_samples():
    tokenizer = FakeTokenizer()
    config = SelfDistillationConfig()
    samples = [_make_sample([9, 10])]
    contexts = [_make_context(example_id=13, reward=1.0, num_turns=2)]

    requests, metrics = build_sdpo_teacher_requests(samples=samples, contexts=contexts, tokenizer=tokenizer, config=config)

    assert requests == []
    assert samples[0].completion_mask == [False, False]
    assert metrics["self_distillation/skipped_multiturn_fraction"] == 1.0


def test_extract_feedback_prefers_rollout_field_then_step_extras():
    rollout = {
        "feedback": "Top-level feedback",
        "trajectory": [{"extras": {"feedback": "Step feedback"}}],
    }
    assert extract_feedback(rollout, "feedback") == "Top-level feedback"

    rollout = {
        "trajectory": [{"extras": {}}, {"extras": {"feedback": "Step feedback"}}],
    }
    assert extract_feedback(rollout, "feedback") == "Step feedback"
