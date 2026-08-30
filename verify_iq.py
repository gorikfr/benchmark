#!/usr/bin/env python3
"""Mechanically verify every IQ task's answer key before benchmarking.

Checks per kind:
  number    - optional "verify" expression recomputes the answer
  code      - reference solution passes its own test harness
  choice    - keyed letter exists exactly once among the prompt options
  retrieval - needle facts embedded in the haystack; numeric needles occur
              exactly once (no filler collision)
  all kinds - the scorer accepts a canonical reply built from the key

Run:  python verify_iq.py
"""

import json
import math
import re
import sys

from llm_bench import IQ_TASKS, _norm, score_task


def canon_reply(task) -> str:
    """Minimal reply a correct model could give, built from the key itself."""
    k = task["kind"]
    if k == "number":
        return f"Reasoning... Answer: {task['answer']}"
    if k == "choice":
        return task["answer"]
    if k in ("text", "raw_text"):
        return task["accept"][0]
    if k == "json":
        return json.dumps(task["value"])
    if k == "code":
        return "```python\n" + (task.get("ref") or "").strip() + "\n```"
    if k == "line_count":
        return "\n".join(["a line"] * task["count"])
    if k == "bullets":
        return "\n".join(["- thing"] * task["count"])
    if k == "line_rule":
        base = task.get("require") or ("zz" if task.get("forbid") else "word")
        return "\n".join([base] * task["count"])
    if k == "words_exact":
        return " ".join(["word"] * task["count"])
    raise ValueError(k)


def main() -> int:
    failures = []
    names = [t["name"] for t in IQ_TASKS]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        failures.append(f"duplicate task names: {sorted(dupes)}")

    for t in IQ_TASKS:
        name = t["name"]
        fail = lambda msg: failures.append(f"{name}: {msg}")

        # 1. explicit verification expression recomputes the answer
        if "verify" in t:
            try:
                got = eval(t["verify"], {"math": math})  # noqa: S307 - our own bank
            except Exception as e:
                fail(f"verify expr error: {e}")
                continue
            want = float(t["answer"])
            if abs(float(got) - want) > max(1e-9, abs(want) * 1e-9):
                fail(f"verify={t['verify']} -> {got}, key says {t['answer']}")

        # 2. code: reference solution must pass its own tests
        if t["kind"] == "code":
            from llm_bench import _check_code
            ref = t.get("ref")
            if not ref:
                fail("missing reference solution")
            elif not _check_code(ref.strip(), t["tests"]):
                fail("reference solution fails its own tests")

        # 3. choice: keyed letter present exactly once as an option
        if t["kind"] == "choice":
            letters = re.findall(r"^([A-D])\)", t["prompt"], re.M)
            if len(letters) != len(set(letters)):
                fail(f"duplicate option letters: {letters}")
            if len(letters) < 2:
                fail("fewer than two options parsed from prompt")
            if t["answer"] not in letters:
                fail(f"key {t['answer']} not among parsed options {letters}")

        # 4. retrieval: needles embedded; numeric needles unique in prompt
        if t["cat"] == "retrieval":
            for fact in t.get("facts", []):
                if fact not in t["prompt"]:
                    fail(f"needle missing from generated haystack: {fact!r}")
            if t["kind"] == "number":
                hits = t["prompt"].count(str(t["answer"]))
                if hits != 1:
                    fail(f"answer {t['answer']} occurs {hits}x in prompt "
                         "(filler collision or missing needle)")

        # 5. closed loop: scorer must accept the canonical reply for this key
        try:
            if not score_task(t, canon_reply(t)):
                fail(f"scorer rejects its own key (kind={t['kind']})")
        except Exception as e:
            fail(f"scorer raised on canonical reply: {e}")

    cats = {}
    for t in IQ_TASKS:
        c = t["cat"]
        cats[c] = cats.get(c, 0) + 1
    print(f"{len(IQ_TASKS)} tasks across {len(cats)} categories:")
    for c in sorted(cats):
        print(f"  {c:<14}{cats[c]}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  x {f}")
        return 1
    print("\nall keys verified OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
