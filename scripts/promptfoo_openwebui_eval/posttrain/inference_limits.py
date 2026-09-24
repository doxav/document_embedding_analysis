"""Explicit inference budgets; byte bounds are conservative estimates, not token counts."""
from __future__ import annotations

import os
from typing import Any

from .core import require

ABSOLUTE_OUTPUT_TOKENS = 50_000
EPISODE_FIELDS = {'max_completion_tokens', 'max_llm_calls', 'max_tool_calls', 'max_seconds'}
HISTORICAL_EPISODE_LIMITS = {'max_completion_tokens': 16384, 'max_llm_calls': 8,
                             'max_tool_calls': 8, 'max_seconds': 600}


def positive_int(value: Any, name: str, maximum: int = 2**31 - 1) -> int:
    """Reject booleans, coercions and nonpositive or excessive configured integers."""
    require(type(value) is int and 0 < value <= maximum, 'Invalid inference limit: ' + name)
    return value


def setting(name: str, default: int, maximum: int = 2**31 - 1,
            spec: dict[str, Any] | None = None, field: str | None = None) -> int:
    """Resolve an explicit spec value before the environment and a documented default."""
    if spec is not None and field in spec:
        return positive_int(spec[field], str(field), maximum)
    raw = os.environ.get(name)
    if raw is None:
        return positive_int(default, name, maximum)
    require(raw.isascii() and raw.isdigit(), 'Invalid inference limit: ' + name)
    return positive_int(int(raw), name, maximum)


def output_ceiling() -> int:
    """Apply the operator ceiling without permitting more than 50,000 output tokens."""
    return setting('SA_OUTPUT_TOKEN_CEILING', ABSOLUTE_OUTPUT_TOKENS, ABSOLUTE_OUTPUT_TOKENS)


def episode_limits(overrides: dict[str, Any] | None = None) -> dict[str, int]:
    """Resolve the cumulative episode budget, distinct from each provider call."""
    if overrides is not None:
        require(isinstance(overrides, dict) and set(overrides) == EPISODE_FIELDS,
                'Explicit evaluation-limit keys required')
    definitions = {
        'max_completion_tokens': ('SA_MAX_EPISODE_COMPLETION_TOKENS', 50_000, ABSOLUTE_OUTPUT_TOKENS),
        'max_llm_calls': ('SA_MAX_LLM_CALLS', 8, 2**31 - 1),
        'max_tool_calls': ('SA_MAX_TOOL_CALLS', 8, 2**31 - 1),
        'max_seconds': ('SA_MAX_EPISODE_SECONDS', 600, 36_000),
    }
    return {field: setting(env, default, maximum, overrides, field)
            for field, (env, default, maximum) in definitions.items()}


def validate_teacher_limits(value: Any) -> dict[str, Any]:
    """Validate the complete limits pinned by an authenticated recorder session."""
    fields = {'max_output_tokens', 'max_input_bound_bytes', 'request_timeout_seconds',
              'collection_timeout_seconds', 'evaluation_limits'}
    require(isinstance(value, dict) and set(value) == fields, 'Invalid recorder inference-limit fields')
    limits = episode_limits(value['evaluation_limits'])
    output = positive_int(value['max_output_tokens'], 'max_output_tokens', output_ceiling())
    require(output <= limits['max_completion_tokens'], 'Per-call output exceeds the episode token budget')
    return {'max_output_tokens': output,
            'max_input_bound_bytes': positive_int(value['max_input_bound_bytes'], 'max_input_bound_bytes'),
            'request_timeout_seconds': positive_int(value['request_timeout_seconds'], 'request_timeout_seconds', 36_000),
            'collection_timeout_seconds': positive_int(value['collection_timeout_seconds'], 'collection_timeout_seconds', 36_000),
            'evaluation_limits': limits}


def teacher_limits(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve teacher and transport settings without altering explicit existing specs."""
    spec = {} if spec is None else spec
    require(isinstance(spec, dict), 'Invalid inference specification')
    return validate_teacher_limits({
        'max_output_tokens': setting('SA_MAX_OUTPUT_TOKENS', 8192, output_ceiling(), spec, 'max_output_tokens'),
        'max_input_bound_bytes': setting('SA_MAX_INPUT_BOUND', 50_000, spec=spec, field='max_input_bound_bytes'),
        'request_timeout_seconds': setting('SA_TEACHER_TIMEOUT_SECONDS', 600, 36_000, spec, 'request_timeout_seconds'),
        'collection_timeout_seconds': setting('SA_COLLECTION_TIMEOUT_SECONDS', 900, 36_000, spec, 'collection_timeout_seconds'),
        'evaluation_limits': episode_limits(spec.get('evaluation_limits'))})


def judge_limits(openrouter: bool) -> dict[str, Any]:
    """Resolve the bounded judge protocol; reasoning is sent only to verified OpenRouter."""
    effort = os.environ.get('SA_JUDGE_REASONING_EFFORT', 'low') if openrouter else None
    require(effort is None or effort in {'low', 'medium', 'high'}, 'Unsupported judge reasoning effort')
    return {'max_output_tokens': setting('SA_JUDGE_MAX_OUTPUT_TOKENS', 4096, output_ceiling()),
            'max_input_bound_bytes': setting('SA_JUDGE_MAX_INPUT_BOUND', 131072),
            'request_timeout_seconds': setting('SA_JUDGE_TIMEOUT_SECONDS', 300, 36_000),
            'reasoning_effort': effort}


def native_episode_seconds() -> int:
    """Keep the former native timeout override within the common episode deadline."""
    common = episode_limits()['max_seconds']
    return min(common, setting('SA_NATIVE_MAX_SECONDS', common, 36_000),
               setting('OPENWEBUI_TIMEOUT_SECONDS', 600, 36_000))
