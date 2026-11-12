"""JSON-serializable design contracts; not generators or validators."""
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Metadata:
    dataset_version: str
    source: str
    validation_status: str = "pending"


@dataclass
class GroundTruth:
    scenario_type: str
    candidate_tools: list[str]
    target_tools: list[str]
    metadata: Metadata


@dataclass
class BackwardQuery:
    target_tools: list[str]
    user_query: str
    allowed_context: dict[str, Any]
    metadata: Metadata


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ForwardCalls:
    user_query: str
    tool_calls: list[ToolCall]
    metadata: Metadata


@dataclass
class VerificationFeedback:
    stage: int
    valid: bool
    status: str
    checks: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    retry_count: int
    max_retries: int
    failure_reasons: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TrajectoryStep:
    call: ToolCall
    observation: dict[str, Any] | None
    argument_sources: dict[str, dict[str, Any]]  # query, context, observation, or schema default


@dataclass
class Trajectory:
    user_query: str
    steps: list[TrajectoryStep]
    final_state: dict[str, Any] | None
    metadata: Metadata


def to_dict(contract):
    return asdict(contract)
