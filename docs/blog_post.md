# EuroPowerRAG v2: From Context to Conviction

*Draft Case Study*

When we built the first iteration of EuroPowerRAG, the goal was simple: make European power market data queryable in plain English. By hooking ENTSO-E transparency data and energy news RSS feeds into a local ChromaDB index, we allowed analysts to ask, *"What drove German baseload prices last week?"* and get a grounded, cited answer.

But in trading, context is only half the battle. The other half is conviction. 

In **v2**, we transformed a passive retrieval system into an active decision engine. We didn't just ask the LLM to summarize; we built an architecture that forecasts, critiques, and sizes trades.

## The Architecture
We started by layering traditional quantitative models—ARIMA and XGBoost—to form a baseline price forecast. We then wired a prompt-cached Claude agent (`claude-haiku-4-5`) to act as the primary decision-maker.

The agent loop is constrained and deliberate. It doesn't have infinite freedom. Instead, it is given:
1. **The Forecast**: Next-day price predictions and uncertainty.
2. **The Sentiment**: A structured score (-1 to +1) derived from our news RAG pipeline.
3. **The Risk Profile**: Current realized volatility and dynamic Kelly-fraction sizing.

## The Self-Critique Loop
The most critical addition in v2 was the **self-critique pass**. Before any `TradeSignal` is emitted, the agent must evaluate its own confidence. If the confidence falls below 60%, or if the rationale fails to synthesize all three inputs (forecast, sentiment, risk), the signal is autonomously overridden to a `HOLD`. 

## Results & Walk-Forward
*(Metrics to be populated by the eval log after a 6-month walk-forward backtest...)*

## Conclusion
EuroPowerRAG v2 demonstrates how to bridge the gap between generative AI and quantitative finance. By treating the LLM not as a calculator, but as a synthesizer of distinct quantitative inputs, we created a system that reasons about risk just as much as it reasons about price.
