# Local LLM Benchmark

This project is for running your own repeatable benchmark on your local hardware and comparing different language models under the same conditions. It measures both generation performance and a small, fixed intelligence/task suite using an OpenAI-compatible local server.

The goal is practical comparison on your machine—not a universal leaderboard. Results depend on your hardware, memory pressure, model quantization, server implementation, context length, and runtime settings.

## What it measures

### Speed benchmark

The speed benchmark:

- warms up the model without saving the warmup result;
- runs the short apple-pie prompt twice per requested run;
- runs a longer prompt to exercise prefill and decode performance;
- records time to first token (TTFT), total time, generation throughput, and prefill throughput when authoritative token usage is returned by the server.

Use the same server configuration, prompt settings, and number of runs when comparing models. The benchmark assigns a `run_id` to each invocation so the dashboard can identify the latest run reliably.

### IQ/task benchmark

The `--iq` suite contains fixed, machine-scored tasks across math, logic, knowledge, coding, retrieval, and instruction-following categories. It includes an older v1 set and a harder v2 set intended to expose differences between models and quantizations.

The suite is useful as a quick regression signal, but it is not a complete measure of intelligence, factual reliability, coding ability, or general usefulness.

## Requirements

- Python 3.10 or newer
- A local or reachable OpenAI-compatible server exposing `/v1/models` and `/v1/chat/completions`
- No third-party Python packages are required by the benchmark scripts

The default server URL is `http://localhost:11234/v1`. Start your preferred local inference server separately and make sure it supports streaming chat completions. For accurate token-based speed metrics, the server should return streaming `usage` data when requested; otherwise token counts and token rates are shown as unavailable rather than estimated.

## Quick start

```bash
git clone https://github.com/gorikfr/benchmark.git
cd benchmark

# Verify the built-in answer key before running the IQ suite.
python3 verify_iq.py

# See the model IDs exposed by the local server.
python3 llm_bench.py --list

# Run a speed benchmark.
python3 llm_bench.py --model your-model-id --runs 3

# Run the IQ/task benchmark.
python3 llm_bench.py --model your-model-id --iq
```

Replace `your-model-id` with the exact ID returned by `--list`.

## Common commands

```bash
# Use a different OpenAI-compatible server.
python3 llm_bench.py \
  --url http://127.0.0.1:8000/v1 \
  --model your-model-id

# Put runs with the same hardware/server configuration in one experiment.
python3 llm_bench.py \
  --experiment-id macbook-mlx-default \
  --model your-model-id \
  --runs 3

# Compare multiple models in one invocation.
python3 llm_bench.py \
  --model model-a \
  --model model-b \
  --runs 3

# Repeat the IQ suite to reduce run-to-run noise.
python3 llm_bench.py --model your-model-id --iq --iq-runs 2

# Run only selected IQ categories.
python3 llm_bench.py \
  --model your-model-id \
  --iq \
  --iq-categories math-hard,retrieval

# Increase the response budget for reasoning-heavy models.
python3 llm_bench.py --model your-model-id --iq --iq-tokens 4096
```

The numeric options must be positive. The speed benchmark uses two short-prompt samples for each `--runs` iteration and one long-prompt sample per invocation.

## Viewing results

Results are written locally as JSON Lines files under `results/`, with one file per model. They are intentionally ignored by Git so personal hardware measurements are not published with the source code.

After running a benchmark, start a simple local web server from the repository root:

```bash
python3 -m http.server 8001
```

Open [http://localhost:8001/dashboard.html](http://localhost:8001/dashboard.html) in a browser. The benchmark automatically refreshes the local `results.json` index after a run. If result files were added or changed manually, rebuild that index with:

```bash
python3 llm_bench.py --index
```

The dashboard lets you select an experiment, shows the latest benchmark run and historical runs within it, and displays IQ category scores. The combined ranking averages valid run-level summaries across the selected experiment. It gives IQ accuracy 50% of the score and the available speed metrics the other 50%; a model needs both IQ and speed data to receive a combined score. Use a different `--experiment-id` whenever hardware, model settings, server settings, or benchmark code changes materially.

## Reproducibility tips

For a fair comparison:

1. Use the same machine, server runtime, model settings, and endpoint for every model.
2. Close unrelated workloads and allow the hardware to reach a stable temperature.
3. Run a few repetitions and compare averages rather than a single measurement.
4. Record model quantization, context settings, batch/concurrency settings, and software versions alongside the results.
5. Treat missing token usage as missing data, not as evidence that one model is faster.

## Safety and limitations

The IQ coding tasks execute Python code returned by the model to test its functions. Run this benchmark only against models and endpoints you trust, and use an isolated environment if model output is not fully trusted. The evaluator is not a security sandbox.

The task bank and scoring rules are deliberately simple and deterministic. They can contain blind spots, and passing these tasks should not be interpreted as a broad safety, quality, or capability guarantee.
