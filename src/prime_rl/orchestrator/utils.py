import asyncio
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, AsyncContextManager

import pandas as pd
import verifiers as vf
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from rich.console import Console
from rich.table import Table
from verifiers.utils.async_utils import maybe_semaphore
from verifiers.utils.client_utils import setup_openai_client

from prime_rl.configs.orchestrator import SamplingConfig, SelfDistillationConfig
from prime_rl.transport import TrainingSample
from prime_rl.utils.utils import (
    format_num,
    format_time,
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)

SEMAPHORE: AsyncContextManager | None = None


@dataclass
class SelfDistillationBatch:
    """Teacher-side reprompts for the subset of samples that have hindsight signal."""

    sample_indices: list[int]
    teacher_prompt_ids: list[list[int]]
    metrics: dict[str, float]


async def set_semaphore(limit: int):
    global SEMAPHORE
    SEMAPHORE = await maybe_semaphore(limit)


async def get_semaphore() -> AsyncContextManager:
    global SEMAPHORE
    assert SEMAPHORE is not None, "Semaphore not set"
    return SEMAPHORE


def _get_prompt_messages(rollout: vf.RolloutOutput, state_key: str) -> list[dict[str, str]] | None:
    """Load the original chat messages for a rollout from environment state."""
    prompt_messages = rollout.get(state_key)
    if not isinstance(prompt_messages, list) or not prompt_messages: return None
    if not all(isinstance(message, dict) for message in prompt_messages): return None
    return prompt_messages


def _get_feedback(rollout: vf.RolloutOutput, state_key: str) -> str | None:
    """Load textual rollout feedback if the environment provided it."""
    feedback = rollout.get(state_key)
    if not isinstance(feedback, str) or not feedback.strip(): return None
    return feedback


def _get_solution_text(rollout: vf.RolloutOutput, state_key: str) -> str | None:
    """Load a successful completion artifact that can be reused as a demonstration."""
    solution_text = rollout.get(state_key)
    if not isinstance(solution_text, str) or not solution_text.strip(): return None
    return solution_text


def _normalize_token_ids(tokenized_prompt: Any) -> list[int]:
    """Convert tokenizer outputs from chat templates into a plain list of token ids."""
    if isinstance(tokenized_prompt, list) and all(isinstance(token_id, int) for token_id in tokenized_prompt):
        return tokenized_prompt
    if hasattr(tokenized_prompt, "get") and tokenized_prompt.get("input_ids") is not None:
        input_ids = tokenized_prompt["input_ids"]
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            return list(input_ids[0])
        return list(input_ids)
    if hasattr(tokenized_prompt, "ids"):
        return list(tokenized_prompt.ids)
    if isinstance(tokenized_prompt, list) and len(tokenized_prompt) == 1 and hasattr(tokenized_prompt[0], "ids"):
        return list(tokenized_prompt[0].ids)
    if hasattr(tokenized_prompt, "tolist"):
        token_ids = tokenized_prompt.tolist()
        return token_ids[0] if token_ids and isinstance(token_ids[0], list) else token_ids
    raise TypeError(f"Unsupported tokenized prompt type for self-distillation: {type(tokenized_prompt)!r}")


def _build_teacher_messages(
    prompt_messages: list[dict[str, str]],
    solution_text: str | None,
    feedback_text: str | None,
    config: SelfDistillationConfig,
) -> list[dict[str, str]]:
    """Build the reprompted teacher context from the original prompt plus hindsight signal."""
    if not prompt_messages:
        raise ValueError("prompt_messages must not be empty")
    if prompt_messages[-1].get("role") != "user":
        raise ValueError("The final prompt message must be a user message for self-distillation")

    prompt_text = str(prompt_messages[-1].get("content", ""))
    has_solution = solution_text is not None
    has_feedback = feedback_text is not None
    use_feedback = has_feedback and (not config.environment_feedback_only_without_solution or not has_solution)
    if not has_solution and not use_feedback:
        return prompt_messages

    solution_section = (
        config.solution_template.format(successful_previous_attempt=solution_text) if has_solution else ""
    )
    feedback_section = config.feedback_template.format(feedback_raw=feedback_text) if use_feedback else ""
    reprompt_text = config.reprompt_template.format(prompt=prompt_text, solution=solution_section, feedback=feedback_section)
    return prompt_messages[:-1] + [{"role": "user", "content": reprompt_text}]


def build_self_distillation_batch(
    tokenizer,
    rollouts: list[vf.RolloutOutput],
    rollout_to_sample_indices: list[list[int]],
    config: SelfDistillationConfig,
) -> SelfDistillationBatch:
    """Build teacher reprompts for samples that have a success demo and/or environment feedback."""
    success_by_example: dict[tuple[str, int], list[int]] = {}
    for rollout_index, rollout in enumerate(rollouts):
        key = (str(rollout["task"]), int(rollout["example_id"]))
        if float(rollout["reward"]) < config.success_reward_threshold: continue
        if _get_solution_text(rollout, config.solution_state_key) is None: continue
        success_by_example.setdefault(key, []).append(rollout_index)

    sample_indices: list[int] = []
    teacher_messages_batch: list[list[dict[str, str]]] = []
    num_with_solution = 0
    num_with_feedback_available = 0
    num_with_feedback_used = 0
    example_keys = [(str(rollout["task"]), int(rollout["example_id"])) for rollout in rollouts]

    for rollout_index, rollout in enumerate(rollouts):
        prompt_messages = _get_prompt_messages(rollout, config.prompt_messages_state_key)
        if prompt_messages is None: continue

        key = example_keys[rollout_index]
        solution_indices = list(success_by_example.get(key, []))
        if config.dont_reprompt_on_self_success:
            solution_indices = [index for index in solution_indices if index != rollout_index]
        solution_text = (
            _get_solution_text(rollouts[solution_indices[0]], config.solution_state_key) if solution_indices else None
        )
        feedback_text = _get_feedback(rollout, config.feedback_state_key) if config.include_environment_feedback else None
        feedback_used = feedback_text is not None and (not config.environment_feedback_only_without_solution or solution_text is None)
        if feedback_text is not None:
            num_with_feedback_available += 1
        if solution_text is not None:
            num_with_solution += 1
        if feedback_used:
            num_with_feedback_used += 1
        if solution_text is None and not feedback_used:
            continue

        teacher_messages = _build_teacher_messages(
            prompt_messages,
            solution_text,
            feedback_text if feedback_used else None,
            config,
        )
        for sample_index in rollout_to_sample_indices[rollout_index]:
            sample_indices.append(sample_index)
            teacher_messages_batch.append(teacher_messages)

    if not teacher_messages_batch:
        return SelfDistillationBatch(sample_indices=[], teacher_prompt_ids=[], metrics={
            "self_distillation/success_group_fraction": 0.0,
            "self_distillation/success_sample_fraction": 0.0,
            "self_distillation/feedback_available_fraction": 0.0,
            "self_distillation/feedback_used_fraction": 0.0,
            "self_distillation/reprompt_sample_fraction": 0.0,
        })

    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = config.reprompt_truncation
    try:
        teacher_prompt_ids = [
            _normalize_token_ids(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    max_length=config.max_reprompt_len,
                    truncation=True,
                )
            )
            for messages in teacher_messages_batch
        ]
    finally:
        tokenizer.truncation_side = original_truncation_side

    num_examples = len(rollouts)
    successful_groups = len(success_by_example)
    metrics = {
        "self_distillation/success_group_fraction": successful_groups / max(len(set(example_keys)), 1),
        "self_distillation/success_sample_fraction": num_with_solution / max(num_examples, 1),
        "self_distillation/feedback_available_fraction": num_with_feedback_available / max(num_examples, 1),
        "self_distillation/feedback_used_fraction": num_with_feedback_used / max(num_examples, 1),
        "self_distillation/reprompt_sample_fraction": len({index for index in sample_indices}) / max(sum(len(indices) for indices in rollout_to_sample_indices), 1),
    }
    return SelfDistillationBatch(sample_indices=sample_indices, teacher_prompt_ids=teacher_prompt_ids, metrics=metrics)


def get_sampling_args(sampling_config: SamplingConfig, temperature: float) -> dict:
    # Convert SamplingConfig to vLLM OAI sampling args
    # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_2
    sampling_args = dict(sampling_config)
    sampling_args.pop("temp_scheduler", None)
    sampling_args["temperature"] = temperature
    sampling_args["top_p"] = 1.0
    sampling_args["logprobs"] = True
    sampling_args["extra_body"] = {
        **sampling_config.extra_body,
        "return_token_ids": True,  # Always return token IDs
        "top_k": -1,
        "min_p": 0.0,
    }
    sampling_args["extra_body"]["min_tokens"] = sampling_args.pop("min_tokens")
    sampling_args["extra_body"]["repetition_penalty"] = sampling_args.pop("repetition_penalty")
    return sampling_args


def parse_num_completion_tokens(responses: list[list[ChatCompletion]]) -> list[int]:
    """Parses the number of tokens from a list of chat completions returned by OAI API."""
    all_num_completion_tokens = []
    for response in responses:
        num_completion_tokens = 0
        for chat_completion in response:
            assert isinstance(chat_completion, ChatCompletion)
            assert chat_completion.usage is not None, "Usage should be present in the response"
            usage = chat_completion.usage
            assert isinstance(usage, CompletionUsage)
            num_completion_tokens += usage.completion_tokens
        all_num_completion_tokens.append(num_completion_tokens)
    assert len(all_num_completion_tokens) == len(responses), (
        "Number of completion tokens should be the same as the number of responses"
    )
    return all_num_completion_tokens


def parse_is_truncated_completions(responses: list[list[ChatCompletion]]) -> list[bool]:
    """Parses whether the completions were truncated from a list of (multi-turn) OAI chat completions"""
    all_is_truncated = []
    for response in responses:
        is_truncated = False
        for chat_completion in response:
            assert isinstance(chat_completion, ChatCompletion)
            assert len(chat_completion.choices) == 1, "Response should always have one choice"
            choice = chat_completion.choices[0]
            assert isinstance(choice, Choice)
            if choice.finish_reason == "length":
                is_truncated = True
        all_is_truncated.append(is_truncated)
    return all_is_truncated


def print_benchmark(history: dict[str, list[Any]]) -> None:
    """
    Print benchmark results as rich table. Shows formatted values for the
    inference throughput and overall step time. First first N rows show the
    per-step values, and the last row shows the mean, std, min, and max values.
    """
    history.pop("step")
    assert all(len(v) for v in history.values()), "All metrics must have logged the same number of steps"

    # Turn metric history into pd.DataFrame
    df = pd.DataFrame(dict(history.items()))
    columns = {
        "perf/throughput": "Throughput",
        "time/step": "Step Time",
    }
    df = df.rename(columns=columns)
    df = df[list(columns.values())]
    df = df.iloc[1:]  # Exclude first row

    # Setup console
    console = Console()
    table = Table(title="Benchmark")

    # Add columns
    table.add_column("Step", justify="right")
    for col in df.columns:
        table.add_column(col, justify="center", style="magenta")

    # Add formatted rows
    formatted_df = pd.DataFrame(columns=df.columns)
    formatted_df["Step Time"] = df["Step Time"].apply(format_time)
    formatted_df["Throughput"] = df["Throughput"].apply(format_num, precision=2)
    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step)] + [str(x) for x in row]))

    # Separator
    num_table_columns = 1 + len(df.columns)
    table.add_row(*([""] * num_table_columns))

    # Add row for formatted, aggregated statistics
    mean_df = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_mean_df = pd.DataFrame(columns=mean_df.columns)
    formatted_mean_df["Step Time"] = mean_df["Step Time"].apply(format_time)
    formatted_mean_df["Throughput"] = mean_df["Throughput"].apply(format_num, precision=2)
    mean_row = ["Overall"] + formatted_mean_df.T.apply(
        lambda row: f"{row['mean']} ± {row['std']} [{row['min']}, {row['max']}]", axis=1
    ).tolist()
    table.add_row(*mean_row)

    # Display table
    console.print(table)


async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
    teacher_prompt_ids: list[list[int]] | None = None,
) -> list[list[float]]:
    """Compute teacher model logprobs for a batch of training samples via prefill."""

    async def _compute_single(
        client_config: vf.ClientConfig,
        sample: TrainingSample,
        prompt_ids: list[int] | None,
    ) -> list[float]:
        client = setup_openai_client(client_config)
        token_ids = (prompt_ids if prompt_ids is not None else sample.prompt_ids) + sample.completion_ids

        async with await get_semaphore():
            response = await client.post(
                "/chat/completions/tokens",
                body={
                    "model": model_name,
                    "messages": [{"role": "user", "content": ""}],
                    "tokens": token_ids,
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "skip_special_tokens": False,
                    "prompt_logprobs": True,
                },
                cast_to=ChatCompletion,
            )
        prompt_logprobs = [
            0.0 if lp is None else float(next(iter(lp.values()))["logprob"])
            for lp in getattr(response, "prompt_logprobs", [])
        ]
        if prompt_ids is None:
            return prompt_logprobs
        completion_length = len(sample.completion_ids)
        completion_logprobs = prompt_logprobs[-completion_length:] if completion_length else []
        return ([0.0] * len(sample.prompt_ids)) + completion_logprobs

    teacher_prompt_ids = teacher_prompt_ids or [None] * len(samples)
    return await asyncio.gather(
        *[
            _compute_single(client, sample, prompt_ids)
            for client, sample, prompt_ids in zip(cycle(clients), samples, teacher_prompt_ids)
        ]
    )


def get_weight_dir(output_dir: Path, step: int, check_exists: bool = True, wait_timeout: int | None = None) -> Path:
    """Get the weight directory for a given checkpoint step.

    Args:
        output_dir: The output directory for the run.
        step: The checkpoint step.
        check_exists: If True, raises FileNotFoundError if no weight directory exists.
            If False, returns the broadcast directory path without checking existence
            (useful for NCCL mode where weights are broadcasted, not stored on disk).
        wait_timeout: Maximum time in seconds to wait for a stable directory to appear.
            If None, no waiting is performed.
    """
    ckpt_weight_dir = get_step_path(get_ckpt_dir(output_dir), step) / "weight"
    broadcast_weight_dir = get_step_path(get_broadcast_dir(output_dir), step)

    def find_stable_dir() -> Path | None:
        # For checkpoint weights, check STABLE file in parent directory (checkpoints/step_{step}/STABLE)
        ckpt_step_dir = get_step_path(get_ckpt_dir(output_dir), step)
        if (ckpt_step_dir / "STABLE").exists() and ckpt_weight_dir.exists():
            return ckpt_weight_dir

        # For broadcast weights, check STABLE file in the broadcast directory itself
        if (broadcast_weight_dir / "STABLE").exists() and broadcast_weight_dir.exists():
            return broadcast_weight_dir

        return None

    # Check immediately, then wait if needed
    result = find_stable_dir()
    if result is None and wait_timeout:
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            time.sleep(1)
            result = find_stable_dir()
            if result:
                break

    if result:
        return result
    if not check_exists:
        return broadcast_weight_dir

    raise FileNotFoundError(f"No weight directory found for checkpoint step {step}")
