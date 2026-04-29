"""Position sizing & stop/target logic.

P3.2 / P3.3 — fractional Kelly with a hard cap. Conservative on purpose.
"""

from src.agent.schema import RiskParameters, TradeDirection

KELLY_CAP = 0.25  # Never use more than 1/4 of full Kelly — full Kelly is too aggressive for noise
DEFAULT_STOP_SIGMAS = 1.5
DEFAULT_TARGET_SIGMAS = 2.5


def kelly_fraction(edge: float, vol_annualized: float) -> float:
    """Fractional Kelly. edge = expected excess return (annualized)."""
    if vol_annualized <= 0:
        return 0.0
    raw = edge / (vol_annualized ** 2)
    return max(0.0, min(KELLY_CAP, raw))


def size_position(
    *,
    direction: TradeDirection,
    confidence: float,
    expected_return_pct: float,
    realized_vol_annualized: float,
    account_size_eur: float,
    current_price_eur_mwh: float,
    stop_sigmas: float = DEFAULT_STOP_SIGMAS,
    target_sigmas: float = DEFAULT_TARGET_SIGMAS,
) -> RiskParameters:
    """Compute risk parameters for one trade idea.

    Args:
        direction: BUY/SELL/HOLD
        confidence: 0..1
        expected_return_pct: annualized expected excess return, e.g. 0.05 = 5%
        realized_vol_annualized: same scale as expected_return_pct
        account_size_eur: total notional available
        current_price_eur_mwh: entry price
    """
    if direction == TradeDirection.HOLD or confidence <= 0:
        return RiskParameters(
            position_size_mwh=0.0,
            stop_price_eur_mwh=current_price_eur_mwh,
            target_price_eur_mwh=current_price_eur_mwh,
            max_loss_eur=0.0,
            realized_vol_annualized=realized_vol_annualized,
            kelly_fraction=0.0,
        )

    edge = expected_return_pct * confidence
    f = kelly_fraction(edge, realized_vol_annualized)
    notional_eur = account_size_eur * f
    size_mwh = notional_eur / max(current_price_eur_mwh, 1.0)

    daily_vol = realized_vol_annualized / (252 ** 0.5)
    move = current_price_eur_mwh * daily_vol

    if direction == TradeDirection.BUY:
        stop = current_price_eur_mwh - stop_sigmas * move
        target = current_price_eur_mwh + target_sigmas * move
    else:  # SELL
        stop = current_price_eur_mwh + stop_sigmas * move
        target = current_price_eur_mwh - target_sigmas * move

    max_loss = abs(current_price_eur_mwh - stop) * size_mwh

    return RiskParameters(
        position_size_mwh=round(size_mwh, 3),
        stop_price_eur_mwh=round(stop, 2),
        target_price_eur_mwh=round(target, 2),
        max_loss_eur=round(max_loss, 2),
        realized_vol_annualized=realized_vol_annualized,
        kelly_fraction=f,
    )
