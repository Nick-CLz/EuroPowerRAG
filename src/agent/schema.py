"""Pydantic schemas — every cross-module value is a typed object.

Per AGENTS.md rule #5: structured output, always. Free-text outputs are
unparseable. JSON / Pydantic outputs are programs.

These types form the contract between layers (sentiment → forecast → risk →
agent → backtest). If you change one, downstream consumers must adapt.
"""

from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Sentiment layer ──────────────────────────────────────────────────────────


class SentimentScore(BaseModel):
    """One LLM scoring of one news article."""

    score: float = Field(
        ge=-1.0,
        le=1.0,
        description="-1 strongly bearish for next-day price; +1 strongly bullish.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="LLM's self-rated confidence in this score."
    )
    reasoning: str = Field(
        max_length=300, description="One-sentence justification, used for audit only."
    )


# ── Forecast layer ───────────────────────────────────────────────────────────


class ForecastResult(BaseModel):
    """Single-step price forecast output."""

    country: str
    target_date: date
    point_forecast_eur_mwh: float
    std_eur_mwh: float = Field(ge=0.0, description="Forecast uncertainty (1-sigma).")
    model_name: str = Field(
        description="One of: naive, seasonal_naive, rolling_mean, arima, xgb, xgb_sentiment."
    )
    features_used: list[str] = Field(default_factory=list)


# ── Risk layer ───────────────────────────────────────────────────────────────


class RiskParameters(BaseModel):
    """Risk-adjusted sizing for one trade idea."""

    position_size_mwh: float = Field(
        description="Signed: positive = long, negative = short, 0 = no trade."
    )
    stop_price_eur_mwh: float
    target_price_eur_mwh: float
    max_loss_eur: float = Field(ge=0.0)
    realized_vol_annualized: float = Field(ge=0.0)
    kelly_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of full Kelly used (we cap at 0.25 by default).",
    )


# ── Decision agent ───────────────────────────────────────────────────────────


class TradeDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeSignal(BaseModel):
    """Final output from the decision agent. The thing a human consumes."""

    country: str
    target_date: date
    direction: TradeDirection
    size_mwh: float = Field(description="Always >= 0; direction encodes sign.")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(
        description="The agent's explanation. Must reference forecast, sentiment, and risk."
    )

    forecast_price_eur_mwh: float
    forecast_std_eur_mwh: float
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    realized_vol_annualized: float

    sources: list[str] = Field(
        default_factory=list,
        description="Document IDs / URLs used in the rationale.",
    )

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("rationale")
    @classmethod
    def rationale_must_be_substantive(cls, v: str) -> str:
        if len(v) < 50:
            raise ValueError("Rationale too short — must reference forecast, sentiment, and risk.")
        return v


# ── Backtest ─────────────────────────────────────────────────────────────────


class BacktestTrade(BaseModel):
    """One row in the backtest trade log."""

    entry_date: date
    exit_date: date
    country: str
    direction: TradeDirection
    size_mwh: float
    entry_price: float
    exit_price: float
    pnl_eur: float
    confidence: float


class BacktestSummary(BaseModel):
    """Aggregate metrics from a walk-forward backtest run."""

    start_date: date
    end_date: date
    n_trades: int
    sharpe_annualized: float
    sortino_annualized: float
    max_drawdown_pct: float = Field(ge=0.0, le=1.0)
    win_rate: float = Field(ge=0.0, le=1.0)
    total_pnl_eur: float
    passed_thresholds: bool = Field(
        description="True iff sharpe >= 0.8 AND max_drawdown_pct <= 0.08."
    )
