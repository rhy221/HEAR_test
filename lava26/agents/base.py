from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentResult:
    claim: str
    evidence_pages: List[int]   # 0-indexed
    status: str = "Uncertain"   # Uncertain | Consistent | Contradictory
    raw_response: str = ""
    token_usage: Dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0


class Agent(ABC):
    name: str = "agent"

    def __init__(self, llm_client: Any, cfg: Any):
        self.llm_client = llm_client
        self.cfg = cfg

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> AgentResult:
        ...
