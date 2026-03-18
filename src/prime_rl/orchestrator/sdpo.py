import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import verifiers as vf
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.orchestrator import SelfDistillationConfig
from prime_rl.orchestrator.utils import TeacherPrefillRequest
from prime_rl.transport import TrainingSample

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


@dataclass
class SDPOSampleContext:
    """Rollout context needed to build SDPO teacher prompts for a training sample."""

    example_id: int
    prompt: vf.Messages | None
    reward: float
    feedback: str | None
    num_turns: int
    is_multimodal: bool


def _content_to_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for item in content:
        if item.get("type") != "text":
            return None
        text = item.get("text")
        if not isinstance(text, str):
            return None
        text_parts.append(text)
    return "".join(text_parts)


def _split_prompt(prompt: vf.Messages | None) -> tuple[list[dict[str, str]] | None, str | None]:
    if prompt is None or len(prompt) == 0:
        return None, None
    final_message = prompt[-1]
    if final_message.get("role") != "user":
        return None, None
    prompt_text = _content_to_text(final_message.get("content"))
    if prompt_text is None:
        return None, None
    system_messages: list[dict[str, str]] = []
    for message in prompt[:-1]:
        content = _content_to_text(message.get("content"))
        if content is None:
            return None, None
        system_messages.append({"role": message["role"], "content": content})
    return system_messages, prompt_text


def _strip_thinking(text: str) -> str:
    return THINK_PATTERN.sub("", text).strip()


def extract_feedback(rollout: vf.RolloutOutput, feedback_key: str) -> str | None:
    feedback = rollout.get(feedback_key)
    if isinstance(feedback, str) and feedback.strip():
        return feedback.strip()
    trajectory = rollout.get("trajectory") or []
    for step in reversed(trajectory):
        extras = step.get("extras") or {}
        feedback = extras.get(feedback_key)
        if isinstance(feedback, str) and feedback.strip():
            return feedback.strip()
    return None


def build_sdpo_context(rollout: vf.RolloutOutput, sample: TrainingSample, feedback_key: str) -> SDPOSampleContext:
    return SDPOSampleContext(
        example_id=rollout["example_id"],
        prompt=rollout.get("prompt"),
        reward=rollout["reward"],
        feedback=extract_feedback(rollout, feedback_key),
        num_turns=len(rollout.get("trajectory") or []),
        is_multimodal=sample.pixel_values is not None,
    )


def _disable_sample(sample: TrainingSample) -> None:
    sample.teacher_logprobs = [0.0] * (len(sample.prompt_ids) + len(sample.completion_ids))
    sample.completion_mask = [False] * len(sample.completion_mask)


def build_sdpo_teacher_requests(
    samples: list[TrainingSample],
    contexts: list[SDPOSampleContext],
    tokenizer: PreTrainedTokenizer,
    config: SelfDistillationConfig,
) -> tuple[list[TeacherPrefillRequest], dict[str, float]]:
    completion_texts = [tokenizer.decode(sample.completion_ids, skip_special_tokens=True) for sample in samples]
    success_by_example: dict[int, list[int]] = defaultdict(list)
    for idx, context in enumerate(contexts):
        if context.reward >= config.success_reward_threshold:
            success_by_example[context.example_id].append(idx)

    requests: list[TeacherPrefillRequest] = []
    counts = {
        "supported": 0,
        "with_solution": 0,
        "feedback_available": 0,
        "feedback_used": 0,
        "reprompted": 0,
        "skipped_multiturn": 0,
        "skipped_multimodal": 0,
        "skipped_prompt_shape": 0,
    }

    for idx, (sample, context) in enumerate(zip(samples, contexts)):
        sample.teacher_logprobs = [0.0] * (len(sample.prompt_ids) + len(sample.completion_ids))
        if context.is_multimodal:
            _disable_sample(sample)
            counts["skipped_multimodal"] += 1
            continue
        if context.num_turns != 1:
            _disable_sample(sample)
            counts["skipped_multiturn"] += 1
            continue
        system_messages, prompt_text = _split_prompt(context.prompt)
        if system_messages is None or prompt_text is None:
            _disable_sample(sample)
            counts["skipped_prompt_shape"] += 1
            continue

        counts["supported"] += 1
        solution_indices = success_by_example[context.example_id]
        if config.dont_reprompt_on_self_success:
            solution_indices = [solution_idx for solution_idx in solution_indices if solution_idx != idx]
        solution_text = completion_texts[solution_indices[0]] if len(solution_indices) > 0 else None
        if solution_text is not None and config.remove_thinking_from_demonstration:
            solution_text = _strip_thinking(solution_text)
        if solution_text is not None:
            counts["with_solution"] += 1

        feedback_text = context.feedback if config.include_environment_feedback else None
        if feedback_text is not None:
            counts["feedback_available"] += 1
        use_feedback = feedback_text is not None and (
            not config.environment_feedback_only_without_solution or solution_text is None
        )
        if use_feedback:
            counts["feedback_used"] += 1
        if solution_text is None and not use_feedback:
            _disable_sample(sample)
            continue

        solution_section = (
            config.solution_template.format(successful_previous_attempt=solution_text) if solution_text is not None else ""
        )
        feedback_section = config.feedback_template.format(feedback_raw=feedback_text) if use_feedback else ""
        reprompt_text = config.reprompt_template.format(
            prompt=prompt_text,
            solution=solution_section,
            feedback=feedback_section,
        )
        teacher_prefix_ids = tokenizer.apply_chat_template(
            system_messages + [{"role": "user", "content": reprompt_text}],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=config.max_reprompt_len,
        )
        requests.append(
            TeacherPrefillRequest(
                sample_index=idx,
                tokens=teacher_prefix_ids + sample.completion_ids,
                sample_prompt_len=len(sample.prompt_ids),
                sample_completion_len=len(sample.completion_ids),
            )
        )
        counts["reprompted"] += 1

    num_examples = len({context.example_id for context in contexts}) or 1
    metrics = {
        "self_distillation/success_group_fraction": sum(len(indices) > 0 for indices in success_by_example.values())
        / num_examples,
        "self_distillation/success_sample_fraction": counts["with_solution"] / max(len(samples), 1),
        "self_distillation/feedback_available_fraction": counts["feedback_available"] / max(len(samples), 1),
        "self_distillation/feedback_used_fraction": counts["feedback_used"] / max(len(samples), 1),
        "self_distillation/reprompt_sample_fraction": counts["reprompted"] / max(len(samples), 1),
        "self_distillation/skipped_multiturn_fraction": counts["skipped_multiturn"] / max(len(samples), 1),
        "self_distillation/skipped_multimodal_fraction": counts["skipped_multimodal"] / max(len(samples), 1),
        "self_distillation/skipped_prompt_shape_fraction": counts["skipped_prompt_shape"] / max(len(samples), 1),
    }
    return requests, metrics
