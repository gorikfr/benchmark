"""Regression checks for benchmark parsing and result handling."""
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm_bench import _extract_number, _lm_eval_rows, run_iq, run_lm_eval, score_task

CODE = {"kind": "code", "tests": [("add(2, 3)", repr(5)), ("add(-4, 1)", repr(-3))]}


def main():
    fails = []

    def check(name, cond):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    # number: take the value after the "Answer:" label, not the last number in
    # the tail (models restate other facts after answering)
    check("number: fact restated after answer",
          _extract_number("Answer: 62 bar. Note there are 512 crates.") == 62)
    check("number: parenthetical restatement",
          _extract_number("Answer: 62 bar (the limit), not 512 crates.") == 62)
    check("number: 'Final answer is' variant",
          _extract_number("Final answer is 2.4 hours") == 2.4)
    check("number: fallback without a label",
          _extract_number("The maintenance override PIN is 7741.") == 7741)

    # code: the @@ result marker must not be forgeable by the model
    check("code: honest solution passes",
          score_task(CODE, "```python\ndef add(a,b):\n    return a+b\n```"))
    check("code: forged marker in truncated fence rejected",
          not score_task(CODE, "```python\ndef add(a,b):\n    return 0\n"
                               "print('@@[true, true]@@ 1')"))
    check("code: marker built at runtime rejected",
          not score_task(CODE, "def add(a,b):\n    return 0\n"
                               "print(chr(64)*2+'[true, true]'+chr(64)*2)"))

    # request failures must be tracked separately from evaluated wrong answers
    def fake_ask(_base_url, _model, prompt, _max_tokens):
        if prompt.startswith("A bakery bakes muffins"):
            return "Answer: 378", 4
        raise RuntimeError("offline")

    with TemporaryDirectory() as tmp, patch("llm_bench.bench", return_value={}), \
            patch("llm_bench.ask", side_effect=fake_ask):
        rows = run_iq("http://unused", "test-model", Path(tmp), 32, {"math"})
    math_row = next(r for r in rows if r["label"] == "iq")
    check("request failures: excluded from evaluated total",
          math_row["correct"] == 1 and math_row["total"] == 1 and
          math_row["failed"] == 10 and math_row["attempted"] == 11)
    check("results: run and experiment IDs are recorded",
          all(r["run_id"] for r in rows) and
          all(r["experiment_id"] == "default" for r in rows))

    lm_payload = {
        "results": {
            "hellaswag": {
                "acc,none": 0.625,
                "acc_stderr,none": 0.041,
                "alias": "hellaswag",
            },
            "gsm8k": {"exact_match,flexible-extract": 0.5},
        },
        "higher_is_better": {
            "hellaswag": {"acc,none": True},
            "gsm8k": {"exact_match,flexible-extract": True},
        },
    }
    lm_rows = _lm_eval_rows(lm_payload, "test-model", "run-1", "exp-1", "now")
    hellaswag = next(r for r in lm_rows if r["task"] == "hellaswag")
    check("lm-eval: numeric metrics are normalized",
          len(lm_rows) == 2 and hellaswag["value"] == 0.625 and
          hellaswag["stderr"] == 0.041 and hellaswag["suite"] == "lm-evaluation-harness" and
          all(r["run_id"] == "run-1" and r["experiment_id"] == "exp-1" for r in lm_rows))

    with TemporaryDirectory() as tmp, patch("llm_bench.shutil.which", return_value="/bin/lm-eval"), \
            patch("llm_bench.subprocess.run", return_value=CompletedProcess([], 0, "done\n", "")) as run, \
            patch("llm_bench._lm_eval_result_payload", return_value=lm_payload):
        rows = run_lm_eval("http://localhost:11234/v1", "test-model", Path(tmp),
                           "hellaswag, gsm8k", limit=20, num_fewshot=0, experiment_id="exp-1")
        command = run.call_args.args[0]
        output = Path(tmp) / "test-model.jsonl"
        wrote = output.exists()
    check("lm-eval: invokes local chat backend with requested options",
          len(rows) == 2 and command[:6] == [
              "/bin/lm-eval", "run", "--model", "local-completions",
              "--model_args", "base_url=http://localhost:11234/v1/completions,model=test-model,tokenized_requests=false",
          ] and "--tasks" in command and command[command.index("--tasks") + 1] == "hellaswag,gsm8k" and
          command[command.index("--limit") + 1] == "20" and
          command[command.index("--num_fewshot") + 1] == "0" and wrote)

    with TemporaryDirectory() as tmp, patch("llm_bench.bench", return_value={}), \
            patch("llm_bench.ask", side_effect=RuntimeError("offline")):
        rows = run_iq("http://unused", "test-model", Path(tmp), 32, {"math"})
    math_row = next(r for r in rows if r["label"] == "iq")
    check("request failures: all-failed category has no score",
          math_row["correct"] == 0 and math_row["total"] == 0 and
          math_row["failed"] == 11 and math_row["accuracy"] is None and
          math_row["tokens"] is None)

    print("\n%d failure(s)" % len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
