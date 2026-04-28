# AI Agent Playbook — How to Prompt Before You Code

These are the working rules for using Claude / Gemini / any LLM agent on this project. They're heavily influenced by Andrej Karpathy's writing on LLMs as a new computing primitive. Read this before kicking off a new task.

The premise: **prompting is engineering, not chatting.** Treat the prompt as code — versioned, reviewed, evaluated. Treat the context window as RAM — finite, expensive, allocate it deliberately.

---

## The 10 Rules

### 1. The eval is the product

Before you write the prompt, write the test. Before you write the test, write what "good" looks like as a number.

If you cannot say *"this works when metric X ≥ Y on dataset Z"*, you're not ready to start. Vague goals produce vague code.

For this project: never accept a Phase as "done" until its eval passes its DoD threshold.

---

### 2. Context engineering > prompt engineering

The prompt is small. The context window is large. **What you put into the context matters more than how you phrase the instructions.**

- Strip irrelevant tokens. If the model doesn't need it, don't paste it.
- Order matters: most-relevant info closest to the question.
- Cite, don't dump: pass 5 ranked snippets, not 50 unranked ones.
- For RAG: rerank before generation when retrieval is noisy.

Karpathy's framing: *the LLM is a CPU, the context window is RAM, your job is the OS.*

---

### 3. Ship the dumbest version first

The first version of any feature should be embarrassingly simple. Karpathy: *"the most common cause of a stuck project is over-engineering early."*

Order of operations:
1. Hardcoded constants
2. Naive baseline (last value, mean, simple lookup)
3. Heuristic
4. Classical ML
5. LLM-only
6. LLM + tools
7. Multi-agent

You only earn the right to step N+1 by demonstrating that step N is insufficient. **Most projects die at step 6 because they tried to start there.**

---

### 4. Treat LLMs as probabilistic

Same prompt + same input ≠ same output. Plan for it.

- Use `temperature=0` for deterministic ops (extraction, classification, scoring)
- Use higher temperature only when you want diversity (brainstorming, expansion)
- For critical paths, run N samples and take majority vote (self-consistency)
- Set `seed` if your model exposes it
- Always log prompt + response for debugging

---

### 5. Structured output, always

Free-text outputs are unparseable. JSON / Pydantic outputs are programs.

- Define a Pydantic schema for every LLM call that feeds another step
- Use `response_schema` (Gemini) or `tool_use` (Claude) — don't beg for JSON in the prompt
- Validate. If validation fails, retry once with the error in the prompt, then escalate to human

For this project: every signal, sentiment score, and forecast comes back as a typed object.

---

### 6. Tools are expensive, choose them deliberately

Each tool call adds latency, cost, failure modes, and surface area for prompt injection.

- Default to no tools. Add the first tool only when the LLM provably can't do the task without it.
- One agent loop should call ≤ 3 tools end-to-end. If you find yourself wanting 8, you're building the wrong abstraction.
- Tools should be idempotent and stateless where possible.
- Every tool needs a docstring that explicitly says when *not* to use it.

---

### 7. Verification loops, not blind trust

For anything that goes downstream, the LLM checks its own work before returning.

```
generate → critique → fix
```

- Critique step should run a different prompt (and ideally a smaller / faster model)
- If critique disagrees beyond a threshold, output `UNCERTAIN` with reasoning, not a wrong answer
- For this project: the trading decision agent **must** run self-critique before emitting a signal

---

### 8. Prompts are code

- Store prompts in version control as `.md` files or constants in source
- Never inline a multi-line prompt in a function call
- Diff every prompt change. Re-run evals after every diff.
- Tag prompt versions in commit messages: `prompt(sentiment): v3 — add severity calibration`

---

### 9. Latency and cost budgets

Every agent has a budget. Set it before you start.

- **Latency:** what's the p95 user can tolerate? Build to half that, leaving slack for retries.
- **Cost:** what's the daily $ ceiling? Track it per-call.
- Cache aggressively. Embeddings are deterministic for the same input — cache forever.
- Retry with exponential backoff, but cap retries at 3.

For this project: `MAX_DAILY_LLM_USD=5`, `MAX_QUERY_LATENCY_S=10`. Both enforced in `src/utils/budget.py`.

---

### 10. Human-in-the-loop where it matters

Agents are great at structured suggestions. They are bad at *commitment to actions with consequences*.

- For research / signals: agent decides, human consumes
- For execution / money / sends: agent recommends, human approves
- The user should never be surprised by what the agent did

For this project: the trading agent emits signals. **No autonomous order placement, ever.** The Streamlit dashboard is the human review surface.

---

## The Pre-Flight Checklist (use before starting any new task)

Before you write a single line of LLM-touching code, fill this out in a comment block at the top of your PR / commit:

```
TASK:           <one sentence>
WHY:            <one sentence — what does this enable downstream?>
SUCCESS METRIC: <number on a dataset>
DATASET:        <where it lives, how it was built, who labeled it>
BASELINE:       <the dumbest version that does this>
COST BUDGET:    <$ per call, $ per day>
LATENCY BUDGET: <p50, p95>
CONTEXT INPUTS: <what goes into the context window, in order>
OUTPUT SCHEMA:  <Pydantic class>
EVAL CADENCE:   <when do we re-run the eval>
ROLLBACK PLAN:  <what's the previous-known-good prompt version>
```

If you can't fill all 11 lines, **don't start.** That's not a delay — that's the work.

---

## Anti-Patterns (don't do these)

- **"I'll add a smarter prompt later."** No, you'll forget. Prompts get worse over time without active maintenance.
- **"Let's just chain three agents."** Each handoff is a place where context degrades. Inline what you can.
- **"This output looks reasonable, ship it."** Reasonable ≠ correct. Ship against the eval.
- **"It works on the example."** The example is in your prompt. The eval set is not.
- **"We can fine-tune later."** Most projects don't need fine-tuning. They need better evals and better context.
- **"Just bump the model size."** A bigger model fixes some bugs and creates new ones. Eval first, then upgrade.

---

## Karpathy's Reading List

If you have a free hour, read these in order:

1. *Software 2.0* (2017) — neural networks as a new programming paradigm
2. *Software 3.0* talks (2024+) — English as the new programming language
3. *The unreasonable effectiveness of recurrent neural networks* — for intuition on why LLMs work
4. His 2024 keynote at YC — *LLM OS, context windows, and the new dev loop*

The recurring theme: **the LLM is a new kind of computer. Treat it like one. Engineering discipline still applies — it just applies to different abstractions.**
