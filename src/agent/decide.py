"""Decision agent — composes forecast + sentiment + risk into a TradeSignal.

P4.x — implement after Phase 1, 2, 3 evals are green.

When implementing:
  - Use LangChain agent with three tools: get_forecast, get_sentiment, get_risk
  - AGENTS.md rule #6: ≤ 3 tools per agent loop
  - AGENTS.md rule #7: self-critique pass before emitting signal
  - AGENTS.md rule #10: human-in-the-loop, no autonomous execution
  - The TradeSignal Pydantic model in src.agent.schema is the contract
"""

from datetime import date

from src.agent.schema import TradeSignal


def decide(country: str, target_date: date) -> TradeSignal:
    raise NotImplementedError(
        "P4.x not yet implemented. Phases 1–3 evals must pass first."
    )
