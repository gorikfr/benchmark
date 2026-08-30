#!/usr/bin/env python3
"""Benchmark local LLM servers (OpenAI-compatible, e.g. mlx_lm.server).

Usage:
  python llm_bench.py --model mlx-community/Qwen3.8-27B-8bit
  python llm_bench.py --model a/b --model c/d --runs 3
  python llm_bench.py --model a/b --iq          # intelligence suite (accuracy)
  python llm_bench.py --model a/b --iq --iq-categories math-hard,retrieval
  python llm_bench.py --model a/b --iq --iq-runs 2   # repeat suite, averages out noise
  python llm_bench.py --suite lm-eval --tasks hellaswag --model a/b
  python llm_bench.py --suite lm-eval --tasks hellaswag --model a/b --new-session
  python llm_bench.py --model a/b --experiment-id macbook-mlx-default
  python llm_bench.py --list                  # list models served by --url
  python llm_bench.py --url http://host:port/v1 ...

The IQ suite has two generations of tasks:
  v1 categories (saturate on modern >=4B models): math, choice, logic, code,
      instruction — kept for historical comparability.
  v2 categories (harder, discriminate quantization damage): math-hard,
      code-hard, knowledge-hard, logic-hard, retrieval, instr-hard.

Verify all answers before benchmarking:  python verify_iq.py

The optional lm-evaluation-harness suite requires its separate installation;
see README.md. It runs standard tasks through the local server's completion API.

Results are appended to results/<model>.jsonl (one JSON line per measurement or
task metric) so you
can compare across models and over time. Speed token rates require the server
to return streaming usage; otherwise token-based rates are left unavailable.
Each benchmark invocation is tagged with a run_id and experiment_id so
dashboard.html can group comparable runs reliably.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
SESSION_STATE_PATH = RESULTS_DIR / ".sessions.json"

SHORT_PROMPT = "Write a detailed step-by-step recipe for apple pie."
LONG_PROMPT = ("Summarize the history of computing. " * 400) + "\n\nNow count from 1 to 300."


def _load_sessions() -> dict:
    try:
        state = json.loads(SESSION_STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"sessions": []}
    return state if isinstance(state, dict) and isinstance(state.get("sessions"), list) else {"sessions": []}


def _save_sessions(state: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    temporary = SESSION_STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(SESSION_STATE_PATH)


def _session_fingerprint(base_url: str, model: str, suite: str,
                         backend: str | None = None,
                         num_fewshot: int | None = None) -> str:
    context = {
        "base_url": base_url.rstrip("/"),
        "model": model,
        "suite": suite,
        "backend": backend,
        "num_fewshot": num_fewshot,
    }
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def begin_experiment(base_url: str, model: str, suite: str,
                     explicit_id: str | None = None,
                     new_session: bool = False,
                     backend: str | None = None,
                     num_fewshot: int | None = None) -> tuple[str, bool]:
    """Select or create a per-model session, returning (id, resumed)."""
    if explicit_id and new_session:
        raise ValueError("use either --experiment-id or --new-session, not both")
    state = _load_sessions()
    sessions = state["sessions"]
    fingerprint = _session_fingerprint(base_url, model, suite, backend, num_fewshot)
    matching = [
        session for session in sessions
        if session.get("model") == model
        and session.get("suite") == suite
        and session.get("fingerprint") == fingerprint
        and session.get("status") == "running"
    ]
    resumed = bool(matching) and explicit_id is None and not new_session
    if resumed:
        experiment_id = max(matching, key=lambda session: session.get("updated_at", ""))["experiment_id"]
    else:
        if new_session:
            for session in sessions:
                if session.get("model") == model and session.get("suite") == suite and session.get("status") == "running":
                    session["status"] = "superseded"
        experiment_id = explicit_id or (
            "auto-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
    sessions[:] = [
        session for session in sessions
        if not (
            session.get("model") == model
            and session.get("suite") == suite
            and session.get("experiment_id") == experiment_id
        )
    ]
    sessions.append({
        "model": model,
        "suite": suite,
        "fingerprint": fingerprint,
        "experiment_id": experiment_id,
        "status": "running",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    _save_sessions(state)
    return experiment_id, resumed


def finish_experiment(model: str, suite: str, experiment_id: str) -> None:
    state = _load_sessions()
    for session in state["sessions"]:
        if (session.get("model") == model and session.get("suite") == suite and
                session.get("experiment_id") == experiment_id):
            session["status"] = "finished"
            session["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_sessions(state)


def get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def list_models(base_url: str):
    data = get(base_url.rstrip("/") + "/models")
    for m in data.get("data", []):
        state = m.get("state", "?")
        print(f"  {m['id']}  [{state}]")


def bench(base_url: str, model: str, prompt: str, max_tokens: int, label: str,
          run_id: str | None = None, experiment_id: str | None = None) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    content_chunks = 0
    completion_tokens = None
    prompt_tokens = None

    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            delta = json.loads(payload)
            if delta.get("error"):
                raise RuntimeError(f"streaming API error: {delta['error']}")
            usage = delta.get("usage")
            if usage is not None:
                if usage.get("completion_tokens") is not None:
                    completion_tokens = usage["completion_tokens"]
                if usage.get("prompt_tokens") is not None:
                    prompt_tokens = usage["prompt_tokens"]
                continue
            choices = delta.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                content_chunks += 1

    total = time.perf_counter() - t0
    gen_tps = (
        (completion_tokens - 1) / (total - ttft)
        if completion_tokens is not None and ttft and total > ttft
        else float("nan")
    )
    prefill_tps = (
        prompt_tokens / ttft
        if prompt_tokens is not None and prompt_tokens > 0 and ttft
        else float("nan")
    )
    result = {
        "model": model,
        "label": label,
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "max_tokens": max_tokens,
        "prompt_tokens": prompt_tokens,
        "tokens": completion_tokens,
        "content_chunks": content_chunks,
        "token_source": "usage" if completion_tokens is not None else "unavailable",
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_s": round(total, 4),
        "gen_tps": round(gen_tps, 2) if not math.isnan(gen_tps) else None,
        "prefill_tps": round(prefill_tps, 2) if not math.isnan(prefill_tps) else None,
    }
    if run_id is not None:
        result["run_id"] = run_id
    if experiment_id is not None:
        result["experiment_id"] = experiment_id
    print(f"[{model} | {label}] tokens={completion_tokens} ttft={result['ttft_s']}s "
          f"total={result['total_s']}s gen={result['gen_tps']} tok/s")
    return result


# --------------------------------------------------------------------------
# intelligence suite (--iq): fixed task bank, temperature 0, machine-scored


def ask(base_url: str, model: str, prompt: str, max_tokens: int):
    """Single completion at temperature 0 -> (text, completion_tokens)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    if data.get("error"):
        raise RuntimeError(f"API error: {data['error']}")
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("API response did not contain choices")
    text = (choices[0].get("message") or {}).get("content")
    if text is None:
        raise RuntimeError("API response did not contain message content")
    tokens = (data.get("usage") or {}).get("completion_tokens")
    return text, tokens


def _final_answer(text: str) -> str:
    i = text.lower().rfind("answer")
    return text[i:] if i != -1 else text


def _norm(text: str) -> str:
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


_ANSWER_LABEL = re.compile(r"(?:final\s+)?answer\s*(?:is|:)?\s*", re.I)


def _extract_number(text: str):
    t = text.replace(",", "")
    # Prefer the number right after the last "Answer:" label: reasoning models
    # often restate other facts after it, and a tail scan picks those up.
    labels = list(_ANSWER_LABEL.finditer(t))
    if labels:
        m = re.search(r"-?\d+(?:\.\d+)?", t[labels[-1].end():])
        if m:
            return float(m.group())
    nums = re.findall(r"-?\d+(?:\.\d+)?", _final_answer(t))
    return float(nums[-1]) if nums else None


def _extract_choice(text: str):
    t = _final_answer(text).strip()
    pats = [
        (r"(?:answer|option)s?\s*(?:is|:)\s*\**\(?([A-D])\b", re.I),
        (r"^[^\w]{0,4}([A-D])[\).\]:\s]", re.M),
        (r"\b([A-D])\b", 0),
    ]
    for pat, flags in pats:
        m = re.findall(pat, t, flags)
        if m:
            return m[0].upper()
    return None


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    i = text.find("def ")
    return text[i:].strip() if i != -1 else text.strip()


def _check_code(code: str, tests) -> bool:
    if "@@" in code:  # model must not be able to forge the result marker
        return False
    harness = (
        code.rstrip()
        + "\n\nimport json\n_tests = "
        + repr([[expr, expected] for expr, expected in tests])
        + "\n_res = []\nfor expr, expected in _tests:\n"
        + "    try:\n"
        + "        _res.append(repr(eval(expr)) == expected)\n"
        + "    except Exception:\n"
        + "        _res.append(False)\n"
        # leading newline keeps the marker on a line of its own (models often
        # end their code without a trailing newline)
        + "print('\\n@@' + json.dumps(_res) + '@@')\n"
    )
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(harness)
        try:
            proc = subprocess.run([sys.executable, path], capture_output=True,
                                  text=True, timeout=15)
        except subprocess.TimeoutExpired:
            return False
    finally:
        os.unlink(path)
    # last marker line wins; exact line shape so model stdout can't forge it
    markers = re.findall(r"^@@(\[.*\])@@$", proc.stdout, re.M)
    if not markers:
        return False
    try:
        flags = json.loads(markers[-1])
    except ValueError:
        return False
    return len(flags) == len(tests) and all(flags)


def score_task(task: dict, text: str) -> bool:
    kind = task["kind"]
    if kind == "number":
        got = _extract_number(text)
        want = task["answer"]
        return got is not None and abs(got - want) <= max(1e-6, abs(want) * 0.001)
    if kind == "choice":
        got = _extract_choice(text)
        return got is not None and got == task["answer"].upper()
    if kind == "code":
        return _check_code(_extract_code(text), task["tests"])
    if kind == "text":
        t = f" {_norm(text)} "
        return any(f" {_norm(v)} " in t for v in task["accept"])
    if kind == "raw_text":
        return text.strip() in task["accept"]
    if kind == "json":
        t = re.sub(r"```(?:json)?", "", text)
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return False
        try:
            return json.loads(m.group()) == task["value"]
        except ValueError:
            return False
    if kind == "line_count":
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        return len(lines) == task["count"]
    if kind == "bullets":
        items = re.findall(r"^-\s+\S+\s*$", text.strip(), re.M)
        return len(items) == task["count"]
    if kind == "line_rule":
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if len(lines) != task["count"]:
            return False
        req = task.get("require", "").lower()
        forbid = task.get("forbid", "").lower()
        return all((not req or req in ln.lower()) and (not forbid or forbid not in ln.lower())
                   for ln in lines)
    if kind == "words_exact":
        return len(text.split()) == task["count"]
    raise ValueError(kind)


MATH_SUFFIX = ' Reason step by step, then end your reply with "Answer: <number>".'
CHOICE_SUFFIX = "\nRespond with ONLY the letter of the correct option."


# --------------------------------------------------------------------------
# v2 task bank helpers


_HAY_TEMPLATES = [
    "A {adj} {noun} was logged at {place} during the morning shift.",
    "Staff flagged one {adj} {noun} form for supervisor review.",
    "The {noun} schedule for next week was posted outside {place}.",
    "Two crates from the {adj} {noun} batch were moved to {place}.",
    "An intern catalogued the {adj} {noun} paperwork without errors.",
    "Weather delays pushed the {adj} {noun} route back by several hours.",
    "The night crew finished the {noun} count ahead of schedule.",
    "A replacement part for the {adj} {noun} system arrived by courier.",
    "Shift {n1} closed with {n2} open tickets at {place}.",
    "The vendor portal listed the {adj} {noun} as pending sign-off.",
]

_HAY_WORDS = {
    "adj": ["routine", "coastal", "internal", "scheduled", "archived",
            "preliminary", "seasonal", "unresolved"],
    "noun": ["shipment", "inspection", "invoice", "inventory",
             "maintenance", "audit", "manifest", "delivery"],
    "place": ["the north depot", "warehouse row C", "the loading bay",
              "the annex office", "dock nine", "the records room"],
}


def _haystack(facts: list[str], approx_words: int, seed: int) -> str:
    """Deterministic filler text with needle `facts` inserted at spread depths."""
    rng = random.Random(seed)
    sents: list[str] = []
    n_words = 0
    while n_words < approx_words:
        tpl = rng.choice(_HAY_TEMPLATES)
        sents.append(tpl.format(adj=rng.choice(_HAY_WORDS["adj"]),
                                noun=rng.choice(_HAY_WORDS["noun"]),
                                place=rng.choice(_HAY_WORDS["place"]),
                                n1=rng.randint(10, 99), n2=rng.randint(2, 40)))
        n_words += len(sents[-1].split())
    rng.shuffle(sents)
    for i, fact in enumerate(facts):
        pos = int(len(sents) * (0.15 + 0.7 * i / max(1, len(facts))))
        sents.insert(pos, fact)
    return "Below are archived shift-log notes.\n\n" + "\n".join(sents)


def _ret(name: str, facts: list[str], words: int, seed: int,
         question: str, **score) -> dict:
    """Retrieval task: needle facts buried in deterministic filler."""
    return {"name": name, "cat": "retrieval", "facts": facts,
            "prompt": _haystack(facts, words, seed) + "\n\n" + question, **score}

IQ_TASKS = [
    {"name": "muffins", "cat": "math", "kind": "number",
     "prompt": "A bakery bakes muffins in trays of 24. On Friday they baked 17 trays and sold all but 30 muffins. How many muffins were sold?" + MATH_SUFFIX,
     "answer": 378},
    {"name": "train", "cat": "math", "kind": "number",
     "prompt": "A train travels 240 km in 3 hours. At the same speed, how many km would it travel in 5 hours?" + MATH_SUFFIX,
     "answer": 400},
    {"name": "books", "cat": "math", "kind": "number",
     "prompt": "Sara has 3 boxes with 14 books in each box. She gives away 15 books. How many books does she have left?" + MATH_SUFFIX,
     "answer": 27},
    {"name": "jacket", "cat": "math", "kind": "number",
     "prompt": "A jacket costs $80. It is marked down by 25%, and then an extra 10% is taken off the reduced price. What is the final price in dollars?" + MATH_SUFFIX,
     "answer": 54},
    {"name": "pages", "cat": "math", "kind": "number",
     "prompt": "Tom reads 20 pages every day for 12 days. The book has 350 pages. How many pages does he have left?" + MATH_SUFFIX,
     "answer": 110},
    {"name": "painters", "cat": "math", "kind": "number",
     "prompt": "If 3 painters can paint a house in 8 hours, how many hours would 6 painters need, working at the same rate?" + MATH_SUFFIX,
     "answer": 4},
    {"name": "sum-50", "cat": "math", "kind": "number",
     "prompt": "What is the sum of all whole numbers from 1 to 50?" + MATH_SUFFIX,
     "answer": 1275},
    {"name": "change", "cat": "math", "kind": "number",
     "prompt": "Apples cost $3 per kg and bananas cost $2 per kg. You buy 4 kg of apples and 3 kg of bananas and pay with a $20 bill. How much change do you get in dollars?" + MATH_SUFFIX,
     "answer": 2},
    {"name": "bacteria", "cat": "math", "kind": "number",
     "prompt": "A bacteria culture starts with 5 cells and doubles every hour. How many cells are there after 6 hours?" + MATH_SUFFIX,
     "answer": 320},
    {"name": "strawberry", "cat": "math", "kind": "number",
     "prompt": "How many times does the letter 'r' appear in the word 'strawberry'? Reply with just the number.",
     "answer": 3},
    {"name": "rect-area", "cat": "math", "kind": "number",
     "prompt": "A rectangle's length is twice its width. Its perimeter is 36 cm. What is its area in square centimeters?" + MATH_SUFFIX,
     "answer": 72},
    {"name": "capital-australia", "cat": "choice", "kind": "choice",
     "prompt": "What is the capital city of Australia?\nA) Sydney\nB) Melbourne\nC) Canberra\nD) Perth" + CHOICE_SUFFIX,
     "answer": "C"},
    {"name": "symbol-fe", "cat": "choice", "kind": "choice",
     "prompt": "Which element has the chemical symbol Fe?\nA) Fluorine\nB) Iron\nC) Francium\nD) Fermium" + CHOICE_SUFFIX,
     "answer": "B"},
    {"name": "sequence", "cat": "choice", "kind": "choice",
     "prompt": "What number comes next in the sequence: 2, 6, 12, 20, 30, ...?\nA) 40\nB) 42\nC) 44\nD) 46" + CHOICE_SUFFIX,
     "answer": "B"},
    {"name": "not-prime", "cat": "choice", "kind": "choice",
     "prompt": "Which of these numbers is NOT prime?\nA) 21\nB) 17\nC) 29\nD) 31" + CHOICE_SUFFIX,
     "answer": "A"},
    {"name": "binary-search", "cat": "choice", "kind": "choice",
     "prompt": "What is the time complexity of binary search on a sorted array?\nA) O(1)\nB) O(n)\nC) O(log n)\nD) O(n log n)" + CHOICE_SUFFIX,
     "answer": "C"},
    {"name": "river-egypt", "cat": "choice", "kind": "choice",
     "prompt": "Which river flows through Egypt?\nA) Amazon\nB) Yangtze\nC) Danube\nD) Nile" + CHOICE_SUFFIX,
     "answer": "D"},
    {"name": "syllogism", "cat": "choice", "kind": "choice",
     "prompt": "All roses are flowers. Some flowers fade quickly. Which conclusion follows?\nA) All roses fade quickly\nB) No roses fade quickly\nC) No conclusion can be drawn about whether roses fade quickly\nD) All flowers are roses" + CHOICE_SUFFIX,
     "answer": "C"},
    {"name": "kibibyte", "cat": "choice", "kind": "choice",
     "prompt": "How many bytes are in one kibibyte?\nA) 1000\nB) 1024\nC) 2048\nD) 512" + CHOICE_SUFFIX,
     "answer": "B"},
    {"name": "riddle-map", "cat": "logic", "kind": "text",
     "prompt": "I have cities but no houses, mountains but no trees, and water but no fish. What am I?",
     "accept": ["map"]},
    {"name": "months-28", "cat": "logic", "kind": "text",
     "prompt": "How many months have 28 days?",
     "accept": ["12", "all of them", "all", "every month", "twelve"]},
    {"name": "fib-next", "cat": "logic", "kind": "text",
     "prompt": "What number comes next in the sequence: 1, 1, 2, 3, 5, 8, ...? Reply with just the number.",
     "accept": ["13", "thirteen"]},
    {"name": "sheep", "cat": "logic", "kind": "text",
     "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
     "accept": ["9", "nine"]},
    {"name": "riddle-ton", "cat": "logic", "kind": "text",
     "prompt": "Forward I am heavy, but backward I am not. What am I?",
     "accept": ["ton"]},
    {"name": "triangle-angles", "cat": "logic", "kind": "text",
     "prompt": "What is the total number of degrees in the interior angles of any triangle? Reply with just the number.",
     "accept": ["180", "one hundred eighty"]},
    {"name": "code-add", "cat": "code", "kind": "code",
     "prompt": "Write a Python function add(a, b) that returns the sum of a and b.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("add(2, 3)", repr(5)), ("add(-4, 1)", repr(-3))],
     "ref": """
def add(a, b):
    return a + b
"""},
    {"name": "code-palindrome", "cat": "code", "kind": "code",
     "prompt": "Write a Python function is_palindrome(s) that returns True if the string s reads the same forwards and backwards when case, spaces and punctuation are ignored, otherwise False.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("is_palindrome('A man, a plan, a canal: Panama')", repr(True)),
               ("is_palindrome('Hello')", repr(False)),
               ("is_palindrome('No lemon, no melon')", repr(True))],
     "ref": """
def is_palindrome(s):
    t = ''.join(c.lower() for c in s if c.isalnum())
    return t == t[::-1]
"""},
    {"name": "code-vowels", "cat": "code", "kind": "code",
     "prompt": "Write a Python function count_vowels(s) that returns how many vowels (a, e, i, o, u) the string s contains, case-insensitively.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("count_vowels('hello world')", repr(3)),
               ("count_vowels('xyz')", repr(0)),
               ("count_vowels('AEIOU')", repr(5))],
     "ref": """
def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')
"""},
    {"name": "code-fizzbuzz", "cat": "code", "kind": "code",
     "prompt": "Write a Python function fizzbuzz(n) that returns a list of strings for the numbers 1..n, replacing multiples of 3 with 'Fizz', multiples of 5 with 'Buzz', and multiples of both with 'FizzBuzz'.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("fizzbuzz(15)", repr(["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz",
                                      "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]))],
     "ref": """
def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
"""},
    {"name": "code-flatten", "cat": "code", "kind": "code",
     "prompt": "Write a Python function flatten(x) that takes an arbitrarily nested list and returns a flat list with the same elements in order.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("flatten([1,[2,[3,[4]]],5])", repr([1, 2, 3, 4, 5])),
               ("flatten([])", repr([])),
               ("flatten([[1],[2,3]])", repr([1, 2, 3]))],
     "ref": """
def flatten(x):
    out = []
    for item in x:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
"""},
    {"name": "code-fib", "cat": "code", "kind": "code",
     "prompt": "Write a Python function fib(n) that returns the n-th Fibonacci number, defined by fib(0) = 0, fib(1) = 1.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("fib(10)", repr(55)), ("fib(1)", repr(1)), ("fib(0)", repr(0))],
     "ref": """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
"""},
    {"name": "instr-blue", "cat": "instruction", "kind": "text",
     "prompt": "What color is the sky on a clear day? Answer with exactly one word and nothing else.",
     "accept": ["blue"]},
    {"name": "instr-json", "cat": "instruction", "kind": "json",
     "prompt": 'Reply with ONLY a JSON object having keys "name" set to "Alice" and "age" set to 30.',
     "value": {"name": "Alice", "age": 30}},
    {"name": "instr-nanab", "cat": "instruction", "kind": "raw_text",
     "prompt": "Spell the word banana backwards. Reply with only your answer, in capital letters.",
     "accept": ["NANAB"]},
    {"name": "instr-haiku", "cat": "instruction", "kind": "line_count",
     "prompt": "Write a haiku about rain. Reply with only the three lines of the haiku.",
     "count": 3},
    {"name": "instr-bullets", "cat": "instruction", "kind": "bullets",
     "prompt": "List five different fruits as bullet points, each line formatted as '- word' with exactly one fruit word. Reply with nothing else.",
     "count": 5},

    # ---- v2 bank: harder categories (answers verified by verify_iq.py) ----

    # math-hard
    {"name": "pool-pipes", "cat": "math-hard", "kind": "number",
     "prompt": "Pipe A can fill a swimming pool in 6 hours. Pipe B can fill it in 4 hours. If both pipes are opened together, how many hours will it take to fill the pool?" + MATH_SUFFIX,
     "answer": 2.4, "verify": "1/(1/6+1/4)"},
    {"name": "buses-lcm", "cat": "math-hard", "kind": "number",
     "prompt": "Two shuttle buses leave the depot at 9:00. One returns to the depot every 12 minutes, the other every 18 minutes. In how many minutes will they next be at the depot at the same time?" + MATH_SUFFIX,
     "answer": 36, "verify": "math.lcm(12,18)"},
    {"name": "phone-rise-fall", "cat": "math-hard", "kind": "number",
     "prompt": "A phone costs $600. Its price rises by 20%, and later falls by 25%. What is the final price in dollars?" + MATH_SUFFIX,
     "answer": 540, "verify": "600*1.2*0.75"},
    {"name": "mod-pow", "cat": "math-hard", "kind": "number",
     "prompt": "What is the remainder when 7^100 is divided by 13?" + MATH_SUFFIX,
     "answer": 9, "verify": "pow(7,100,13)"},
    {"name": "seating-factorial", "cat": "math-hard", "kind": "number",
     "prompt": "In how many different orders can 4 people sit in a row of 4 chairs?" + MATH_SUFFIX,
     "answer": 24, "verify": "math.factorial(4)"},
    {"name": "teams-comb", "cat": "math-hard", "kind": "number",
     "prompt": "From a group of 8 colleagues, how many distinct teams of exactly 3 people can be formed?" + MATH_SUFFIX,
     "answer": 56, "verify": "math.comb(8,3)"},
    {"name": "wall-workers", "cat": "math-hard", "kind": "number",
     "prompt": "Five workers can build a shed in 12 days, all working at the same rate. They work for 4 days, then 3 more workers join them at the same rate. How many more days are needed to finish the shed?" + MATH_SUFFIX,
     "answer": 5, "verify": "(5*12 - 5*4)/(5+3)"},
    {"name": "age-gap", "cat": "math-hard", "kind": "number",
     "prompt": "Anna is 24 years older than her brother Ben. In exactly 6 years, Anna will be twice as old as Ben. How old is Ben now?" + MATH_SUFFIX,
     "answer": 18, "verify": "24 + 6 - 12"},
    {"name": "even-sum", "cat": "math-hard", "kind": "number",
     "prompt": "What is the sum of all even numbers from 2 to 100 inclusive?" + MATH_SUFFIX,
     "answer": 2550, "verify": "sum(range(2,101,2))"},
    {"name": "cube-surface", "cat": "math-hard", "kind": "number",
     "prompt": "A cube has a volume of 64 cubic centimeters. What is its total surface area in square centimeters?" + MATH_SUFFIX,
     "answer": 96, "verify": "6*(64**(1/3))**2"},

    # code-hard
    {"name": "roman-numerals", "cat": "code-hard", "kind": "code",
     "prompt": "Write a Python function roman_to_int(s) that converts a Roman numeral string (I, V, X, L, C, D, M) to an integer.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("roman_to_int('III')", repr(3)),
               ("roman_to_int('LVIII')", repr(58)),
               ("roman_to_int('MCMXCIV')", repr(1994))],
     "ref": """
def roman_to_int(s):
    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    for i, ch in enumerate(s):
        if i + 1 < len(s) and vals[s[i + 1]] > vals[ch]:
            total -= vals[ch]
        else:
            total += vals[ch]
    return total
"""},
    {"name": "group-anagrams", "cat": "code-hard", "kind": "code",
     "prompt": "Write a Python function group_anagrams(words) that takes a list of lowercase strings and returns a list of groups (lists), where each group contains all words that are anagrams of each other.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("sorted([sorted(g) for g in group_anagrams(['eat','tea','tan','ate','nat','bat'])])",
                repr([['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']])),
               ("group_anagrams([''])", repr([['']])),
               ("group_anagrams([])", repr([]))],
     "ref": """
def group_anagrams(words):
    groups = {}
    for w in words:
        key = ''.join(sorted(w))
        groups.setdefault(key, []).append(w)
    return list(groups.values())
"""},
    {"name": "run-length-encode", "cat": "code-hard", "kind": "code",
     "prompt": "Write a Python function rle_encode(s) that performs run-length encoding on string s: each run of identical consecutive characters becomes the character followed by the run length.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("rle_encode('aaabbc')", repr('a3b2c1')),
               ("rle_encode('abc')", repr('a1b1c1')),
               ("rle_encode('')", repr(''))],
     "ref": """
def rle_encode(s):
    parts = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        parts.append(s[i] + str(j - i))
        i = j
    return ''.join(parts)
"""},
    {"name": "flatten-dict", "cat": "code-hard", "kind": "code",
     "prompt": "Write a Python function flatten_dict(d) that takes a dict whose values may themselves be nested dicts, and returns a flat dict where nested keys are joined with dots (e.g. {'a': {'b': 1}} -> {'a.b': 1}).\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("flatten_dict({'a': {'b': {'c': 1}}, 'd': 2})", repr({'a.b.c': 1, 'd': 2})),
               ("flatten_dict({})", repr({})),
               ("flatten_dict({'x': 5})", repr({'x': 5}))],
     "ref": """
def flatten_dict(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = prefix + '.' + k if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out
"""},
    {"name": "csv-split", "cat": "code-hard", "kind": "code",
     "prompt": "Write a Python function csv_split(line) that splits one CSV line into a list of fields on commas, treating double quotes as field delimiters and doubled quotes \"\" inside a quoted field as a literal quote character.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [('csv_split(\'a,"b,c",d\')', repr(['a', 'b,c', 'd'])),
               ('csv_split(\'"say ""hi""",x\')', repr(['say "hi"', 'x'])),
               ("csv_split('')", repr(['']))],
     "ref": """
def csv_split(line):
    fields, cur, in_q = [], [], False
    i = 0
    while i < len(line):
        c = line[i]
        if in_q:
            if c == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    cur.append('"')
                    i += 1
                else:
                    in_q = False
            else:
                cur.append(c)
        elif c == '"':
            in_q = True
        elif c == ',':
            fields.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    fields.append(''.join(cur))
    return fields
"""},
    {"name": "matrix-multiply", "cat": "code-hard", "kind": "code",
     "prompt": "Write a Python function matmul(a, b) that multiplies two matrices given as lists of lists of numbers and returns the result as a list of lists.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("matmul([[1,0],[0,1]], [[5,6],[7,8]])", repr([[5, 6], [7, 8]])),
               ("matmul([[1,2],[3,4]], [[5,6],[7,8]])", repr([[19, 22], [43, 50]])),
               ("matmul([[1,2,3]], [[1],[2],[3]])", repr([[14]]))],
     "ref": """
def matmul(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
            for i in range(rows)]
"""},
    {"name": "bracket-balance", "cat": "code-hard", "kind": "code",
     "prompt": "Write a Python function is_balanced(s) that returns True if the brackets (), [] and {} in string s are correctly matched and nested, otherwise False. Non-bracket characters are ignored.\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("is_balanced('({[]})')", repr(True)),
               ("is_balanced('([)]')", repr(False)),
               ("is_balanced('')", repr(True)),
               ("is_balanced('(')", repr(False))],
     "ref": """
def is_balanced(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
"""},
    {"name": "binary-add", "cat": "code-hard", "kind": "code",
     "prompt": "Write a Python function add_binary(a, b) that adds two non-negative binary number strings and returns their sum as a binary string, without converting via int().\nWrap the code in a triple-backtick python block and reply with nothing else.",
     "tests": [("add_binary('1010','111')", repr('10001')),
               ("add_binary('0','0')", repr('0')),
               ("add_binary('111','1')", repr('1000'))],
     "ref": """
def add_binary(a, b):
    i, j, carry = len(a) - 1, len(b) - 1, 0
    out = []
    while i >= 0 or j >= 0 or carry:
        s = carry
        if i >= 0:
            s += int(a[i])
            i -= 1
        if j >= 0:
            s += int(b[j])
            j -= 1
        out.append(str(s % 2))
        carry = s // 2
    return ''.join(reversed(out)) or '0'
"""},

    # knowledge-hard
    {"name": "shortest-day", "cat": "knowledge-hard", "kind": "choice",
     "prompt": "Which planet in our solar system has the shortest day?\nA) Venus\nB) Jupiter\nC) Mercury\nD) Mars" + CHOICE_SUFFIX,
     "answer": "B"},
    {"name": "conductance-unit", "cat": "knowledge-hard", "kind": "choice",
     "prompt": "What is the SI unit of electrical conductance?\nA) Ohm\nB) Farad\nC) Siemens\nD) Tesla" + CHOICE_SUFFIX,
     "answer": "C"},
    {"name": "kuhn-book", "cat": "knowledge-hard", "kind": "choice",
     "prompt": "Who wrote the 1962 book \"The Structure of Scientific Revolutions\"?\nA) Karl Popper\nB) Thomas Kuhn\nC) Imre Lakatos\nD) Paul Feyerabend" + CHOICE_SUFFIX,
     "answer": "B"},
    {"name": "lowest-boiling", "cat": "knowledge-hard", "kind": "choice",
     "prompt": "Which element has the lowest boiling point?\nA) Hydrogen\nB) Nitrogen\nC) Helium\nD) Oxygen" + CHOICE_SUFFIX,
     "answer": "C"},
    {"name": "bosch-painting", "cat": "knowledge-hard", "kind": "choice",
     "prompt": "Who painted \"The Garden of Earthly Delights\"?\nA) Jan van Eyck\nB) Pieter Bruegel the Elder\nC) Albrecht Dürer\nD) Hieronymus Bosch" + CHOICE_SUFFIX,
     "answer": "D"},
    {"name": "baking-soda", "cat": "knowledge-hard", "kind": "choice",
     "prompt": "What is the chemical formula of baking soda?\nA) NaCl\nB) NaHCO3\nC) Na2CO3\nD) NaOH" + CHOICE_SUFFIX,
     "answer": "B"},
    {"name": "guilder-currency", "cat": "knowledge-hard", "kind": "choice",
     "prompt": "Before adopting the euro, what was the currency of the Netherlands?\nA) Franc\nB) Mark\nC) Guilder\nD) Escudo" + CHOICE_SUFFIX,
     "answer": "C"},
    {"name": "largest-moon", "cat": "knowledge-hard", "kind": "choice",
     "prompt": "Which is the largest moon in our solar system?\nA) Titan\nB) Ganymede\nC) Callisto\nD) Europa" + CHOICE_SUFFIX,
     "answer": "B"},
    {"name": "sound-speed", "cat": "knowledge-hard", "kind": "choice",
     "prompt": "Approximately how fast does sound travel through air at 20 degrees Celsius, at sea level?\nA) 343 m/s\nB) 299 m/s\nC) 1230 m/s\nD) 150 m/s" + CHOICE_SUFFIX,
     "answer": "A"},
    {"name": "first-bug-moth", "cat": "knowledge-hard", "kind": "choice",
     "prompt": "In 1947, engineers led by Grace Hopper taped an insect into a logbook as the famous first \"computer bug\". What kind of insect was it?\nA) Moth\nB) Cockroach\nC) Beetle\nD) Fly" + CHOICE_SUFFIX,
     "answer": "A"},

    # logic-hard
    {"name": "knight-or-knave", "cat": "logic-hard", "kind": "choice",
     "prompt": "On an island, knights always tell the truth and knaves always lie. You meet islander A, who says: \"I am a knight.\" What can you conclude?\nA) A is definitely a knight\nB) A is definitely a knave\nC) A could be either\nD) This situation is impossible" + CHOICE_SUFFIX,
     "answer": "C"},
    {"name": "both-knaves", "cat": "logic-hard", "kind": "choice",
     "prompt": "On an island, knights always tell the truth and knaves always lie. Islanders A and B stand together. A says: \"We are both knaves.\" What follows?\nA) B is definitely a knight\nB) B is definitely a knave\nC) B could be either\nD) This situation is impossible" + CHOICE_SUFFIX,
     "answer": "A"},
    {"name": "seat-three", "cat": "logic-hard", "kind": "choice",
     "prompt": "Five friends sit in a row of five seats numbered 1 to 5 from left to right. Ana sits in seat 2. Eve sits in seat 5. Bo sits immediately to the left of Cy. Who sits in seat 3?\nA) Ana\nB) Bo\nC) Cy\nD) Dee" + CHOICE_SUFFIX,
     "answer": "B"},
    {"name": "one-true-label", "cat": "logic-hard", "kind": "choice",
     "prompt": "One box contains gold. Box 1 is labeled \"The gold is here\". Box 2 is labeled \"The gold is not in box 2\". Box 3 is labeled \"The gold is not in box 1\". Exactly one of the three labels is true. Which box holds the gold?\nA) Box 1\nB) Box 2\nC) Box 3\nD) Cannot be determined" + CHOICE_SUFFIX,
     "answer": "B"},
    {"name": "handshakes-four", "cat": "logic-hard", "kind": "number",
     "prompt": "Four colleagues meet and each pair of them shakes hands exactly once. How many handshakes take place in total?" + MATH_SUFFIX,
     "answer": 6, "verify": "math.comb(4,2)"},
    {"name": "clock-angle", "cat": "logic-hard", "kind": "number",
     "prompt": "What is the angle, in degrees, between the hour hand and the minute hand of an analog clock at exactly 3:30?" + MATH_SUFFIX,
     "answer": 75, "verify": "abs((3*30 + 30*0.5) - 180)"},
    {"name": "bloops-lazzies", "cat": "logic-hard", "kind": "choice",
     "prompt": "All bloops are razzies. All razzies are lazzies. Which statement must be true?\nA) All bloops are lazzies\nB) All lazzies are bloops\nC) Some razzies are not bloops\nD) No bloops are lazzies" + CHOICE_SUFFIX,
     "answer": "A"},
    {"name": "birthday-threshold", "cat": "logic-hard", "kind": "number",
     "prompt": "Ignoring leap years and assuming all 365 birthdays are equally likely, what is the smallest number of people needed in a room so that the probability that at least two of them share a birthday exceeds 50%?" + MATH_SUFFIX,
     "answer": 23},

    # retrieval
    _ret("ret-pin-deep", ["The maintenance override PIN is 7741."], 1200, 41,
         "What is the maintenance override PIN mentioned in the notes? Reply with just the number.",
         kind="number", answer=7741),
    _ret("ret-color-shallow", ["The prototype's paint color is teal."], 600, 42,
         "What is the prototype's paint color? Reply with just the color word.",
         kind="text", accept=["teal"]),
    _ret("ret-ledger-code", ["The vendor ledger code is VX-208."], 900, 43,
         "What is the vendor ledger code mentioned in the notes?",
         kind="text", accept=["VX-208"]),
    _ret("ret-auditor-name", ["Dr. Elena Vasquez will audit the reactor logs."], 1500, 44,
         "Who will audit the reactor logs?",
         kind="text", accept=["Elena Vasquez"]),
    _ret("ret-pressure-limit",
         ["Warehouse B currently holds 512 crates.",
          "Route 47 closes for resurfacing in October.",
          "The pump pressure limit is 62 bar."], 800, 45,
         "What is the pump pressure limit mentioned in the notes? Reply with just the number.",
         kind="number", answer=62),

    # instr-hard
    {"name": "harbor-lines", "cat": "instr-hard", "kind": "line_rule",
     "prompt": "Write exactly four sentences about sailing. Put each sentence on its own line. Every sentence must contain the word 'harbor'. Reply with only the four lines.",
     "count": 4, "require": "harbor"},
    {"name": "winter-no-e", "cat": "instr-hard", "kind": "line_rule",
     "prompt": "Think of five words related to winter, none of which contain the letter 'e'. Write them one per line. Reply with only the five words.",
     "count": 5, "forbid": "e"},
    {"name": "nested-json", "cat": "instr-hard", "kind": "json",
     "prompt": 'Reply with ONLY a JSON object of exactly this shape: key "order" mapping to an object with keys "id" set to "A-1029" and "items" set to a list containing "bolt" then "nut"; plus top-level key "paid" set to true.',
     "value": {"order": {"id": "A-1029", "items": ["bolt", "nut"]}, "paid": True}},
    {"name": "alpha-sort-line", "cat": "instr-hard", "kind": "text",
     "prompt": "Sort these words alphabetically: kettle, apple, mango, zebra, bread. Reply with the sorted list on a single line, comma-separated.",
     "accept": ["apple, bread, kettle, mango, zebra"]},
    {"name": "six-word-rain", "cat": "instr-hard", "kind": "words_exact",
     "prompt": "Describe rain in exactly six words. Reply with only those six words on one line.",
     "count": 6},
    {"name": "reverse-colors", "cat": "instr-hard", "kind": "text",
     "prompt": "Reply with this exact list reversed and comma-separated: red, amber, lime, teal",
     "accept": ["teal, lime, amber, red"]},
]


def run_iq(base_url: str, model: str, results_dir: Path, iq_tokens: int,
           categories: set[str] | None = None,
           experiment_id: str = "default") -> list[dict]:
    tasks = IQ_TASKS if not categories else [t for t in IQ_TASKS if t["cat"] in categories]
    run_id = uuid.uuid4().hex
    print(f"\n=== {model} ===")
    print("--- warmup ---")
    bench(base_url, model, SHORT_PROMPT, 32, "warmup", run_id, experiment_id)

    # Each category tracks correct answers, evaluated answers, and request
    # failures separately. A transport/API failure is not a model mistake.
    tally = {}
    tokens_cat = {}
    token_known_cat = {}
    for i, task in enumerate(tasks, 1):
        try:
            text, n_tok = ask(base_url, model, task["prompt"], iq_tokens)
            ok = score_task(task, text)
        except Exception as e:
            text, n_tok, ok = f"<request failed: {e}>", None, None
        cat = task["cat"]
        correct, evaluated, failed = tally.get(cat, (0, 0, 0))
        if ok is True:
            correct += 1
            evaluated += 1
        elif ok is False:
            evaluated += 1
        else:
            failed += 1
        tally[cat] = (correct, evaluated, failed)
        token_known_cat.setdefault(cat, None)
        if ok is not None and n_tok is None:
            token_known_cat[cat] = False
        elif ok is not None and token_known_cat[cat] is not False:
            token_known_cat[cat] = True
        if n_tok is not None:
            tokens_cat[cat] = tokens_cat.get(cat, 0) + n_tok
        status = "ok  " if ok is True else "FAIL" if ok is False else "ERR "
        print(f"[{i:>2}/{len(tasks)}] {status} {cat:<12}{task['name']}")
        if ok is not True:
            snippet = " ".join(str(text).split())[:120]
            print(f"       got: {snippet!r}")

    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    rows = []
    tot = [0, 0, 0, 0]
    all_tokens_known = True
    for cat in sorted(tally):
        correct, evaluated, failed = tally[cat]
        rows.append({"model": model, "label": "iq", "category": cat,
                     "ts": ts, "run_id": run_id, "experiment_id": experiment_id,
                     "correct": correct,
                     "total": evaluated, "attempted": evaluated + failed,
                     "failed": failed,
                     "accuracy": round(correct / evaluated, 3) if evaluated else None,
                     "tokens": tokens_cat[cat] if token_known_cat.get(cat) is True else None})
        tot[0] += correct
        tot[1] += evaluated
        tot[2] += failed
        if token_known_cat.get(cat) is True:
            tot[3] += tokens_cat[cat]
        else:
            all_tokens_known = False
    rows.append({"model": model, "label": "iq-total", "ts": ts,
                 "run_id": run_id, "experiment_id": experiment_id,
                 "correct": tot[0], "total": tot[1],
                 "attempted": tot[1] + tot[2], "failed": tot[2],
                 "accuracy": round(tot[0] / tot[1], 3) if tot[1] else None,
                 "tokens": tot[3] if tot[1] and all_tokens_known else None})

    print("\niq score:")
    for r in rows:
        accuracy = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.0%}"
        ceiling = " <- ceiling" if r["accuracy"] == 1.0 and r["total"] >= 5 and not r["failed"] else ""
        print(f"  {r.get('category', 'TOTAL'):<14}{r['correct']:>2}/{r['total']:<2}"
              f" ({accuracy}, {r['failed']} failed){ceiling}")

    results_dir.mkdir(exist_ok=True)
    out = results_dir / (model.replace("/", "__") + ".jsonl")
    with out.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"saved {len(rows)} rows -> {out}")
    return rows


def run_model(base_url: str, model: str, runs: int, results_dir: Path,
              experiment_id: str = "default") -> list[dict]:
    results = []
    run_id = uuid.uuid4().hex
    print(f"\n=== {model} ===")
    print("--- warmup ---")
    bench(base_url, model, SHORT_PROMPT, 32, "warmup", run_id, experiment_id)
    for i in range(1, runs + 1):
        print(f"--- generation speed (short prompt, run {i}/{runs}) ---")
        results.append(bench(base_url, model, SHORT_PROMPT, 512, "gen", run_id, experiment_id))
        results.append(bench(base_url, model, SHORT_PROMPT, 512, "gen", run_id, experiment_id))
    print("--- prefill + decode (long ~2800-token prompt) ---")
    results.append(bench(base_url, model, LONG_PROMPT, 256, "long-prompt", run_id, experiment_id))

    results_dir.mkdir(exist_ok=True)
    out = results_dir / (model.replace("/", "__") + ".jsonl")
    with out.open("a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"saved {len(results)} runs -> {out}")
    return results


def _lm_eval_result_payload(output_dir: Path) -> dict:
    """Find the JSON result written by lm-evaluation-harness."""
    candidates = []
    for path in output_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
            candidates.append((path, payload))
    if not candidates:
        raise RuntimeError(f"lm-eval produced no result JSON in {output_dir}")
    return max(candidates, key=lambda item: item[0].stat().st_mtime_ns)[1]


def _lm_eval_cache_path(results_dir: Path, base_url: str, model: str,
                        tasks: str, backend: str, num_fewshot: int | None,
                        experiment_id: str) -> Path:
    """Return a stable response-cache path for one comparable evaluation."""
    context = json.dumps({
        "base_url": base_url.rstrip("/"),
        "model": model,
        "tasks": tasks,
        "backend": backend,
        "num_fewshot": num_fewshot,
        "experiment_id": experiment_id,
    }, sort_keys=True).encode()
    digest = hashlib.sha256(context).hexdigest()[:16]
    model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("._") or "model"
    return results_dir / ".lm-eval-cache" / f"{model_name}-{digest}"


def _lm_eval_interpreter(executable: str) -> str:
    """Use the interpreter selected by the installed lm-eval launcher."""
    try:
        first_line = Path(executable).read_text().splitlines()[0]
    except (OSError, IndexError):
        return sys.executable
    if first_line.startswith("#!"):
        interpreter = first_line[2:].strip().split()[0]
        if Path(interpreter).exists():
            return interpreter
    return sys.executable


def _lm_eval_stderr_key(metric: str, metrics: dict) -> str | None:
    base, separator, suffix = metric.partition(",")
    candidates = [
        f"{base}_stderr{separator}{suffix}" if separator else f"{base}_stderr",
        f"{base}_stderr",
    ]
    return next((key for key in candidates if key in metrics), None)


def _lm_eval_rows(payload: dict, model: str, run_id: str,
                  experiment_id: str, ts: str) -> list[dict]:
    """Normalize lm-eval's task/metric map into dashboard-friendly JSONL rows."""
    rows = []
    higher_is_better = payload.get("higher_is_better") or {}
    for task, metrics in payload.get("results", {}).items():
        if not isinstance(metrics, dict):
            continue
        task_higher = higher_is_better.get(task, {})
        if not isinstance(task_higher, dict):
            task_higher = {}
        for metric, value in metrics.items():
            if metric.endswith("_stderr") or "_stderr," in metric:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue
            stderr_key = _lm_eval_stderr_key(metric, metrics)
            stderr = metrics.get(stderr_key) if stderr_key else None
            if isinstance(stderr, bool) or not isinstance(stderr, (int, float)):
                stderr = None
            rows.append({
                "model": model,
                "label": "lm-eval",
                "suite": "lm-evaluation-harness",
                "task": task,
                "metric": metric,
                "value": value,
                "stderr": stderr,
                "higher_is_better": task_higher.get(metric),
                "ts": ts,
                "run_id": run_id,
                "experiment_id": experiment_id,
            })
    if not rows:
        raise RuntimeError("lm-eval result JSON contained no numeric task metrics")
    return rows


def run_lm_eval(base_url: str, model: str, results_dir: Path, tasks: str,
                limit: int | None = None, num_fewshot: int | None = None,
                experiment_id: str = "default",
                backend: str = "completions") -> list[dict]:
    """Run the optional lm-evaluation-harness API backend and save its scores."""
    executable = shutil.which("lm-eval")
    if executable is None:
        raise RuntimeError(
            "lm-eval is not installed; install it with: "
            "python3 -m pip install 'lm-eval[api]'"
        )

    run_id = uuid.uuid4().hex
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    harness_model = "local-chat-completions"
    if backend == "completions":
        endpoint = base_url.rstrip("/") + "/completions"
        harness_model = "local-completions"
    elif backend != "chat":
        raise ValueError(f"unknown lm-eval backend: {backend}")
    tasks = ",".join(task.strip() for task in tasks.split(",") if task.strip())
    cache_path = _lm_eval_cache_path(
        results_dir, base_url, model, tasks, backend, num_fewshot, experiment_id
    )
    cache_db = Path(str(cache_path) + "_rank0.db")
    if cache_db.exists():
        print(f"resuming from cached lm-eval responses: {cache_db}")
    else:
        print(f"lm-eval response checkpoint: {cache_db}")
    harness_args = [
        "run",
        "--model", harness_model,
        # The local server accepts text prompts, not the token-ID arrays that
        # lm-eval otherwise sends for log-likelihood requests. lm-eval still
        # uses its tokenizer locally to locate the continuation tokens.
        "--model_args",
        f"base_url={endpoint},model={model},tokenized_requests=false",
        "--tasks", tasks,
        "--use_cache", str(cache_path),
    ]
    if limit is not None:
        harness_args.extend(["--limit", str(limit)])
    if num_fewshot is not None:
        harness_args.extend(["--num_fewshot", str(num_fewshot)])

    if backend == "completions":
        # Import the compatibility shim before lm-eval lazily loads its model
        # class. The installed executable may belong to a different Python
        # environment than the interpreter running this benchmark script.
        interpreter = _lm_eval_interpreter(executable)
        runner = (
            "import sys; "
            "import lm_eval_compat; lm_eval_compat.install(); "
            "from lm_eval.__main__ import cli_evaluate; "
            "sys.argv = sys.argv[1:]; sys.exit(cli_evaluate())"
        )
        command = [interpreter, "-c", runner, executable, *harness_args]
        environment = os.environ.copy()
        project_dir = str(Path(__file__).parent)
        environment["PYTHONPATH"] = project_dir + os.pathsep + environment.get("PYTHONPATH", "")
    else:
        command = [executable, *harness_args]
        environment = None

    print(f"\n=== {model} | lm-evaluation-harness ===")
    print("command:", " ".join(command))
    with tempfile.TemporaryDirectory(prefix="lm-eval-") as temp_dir:
        command.extend(["--output_path", temp_dir])
        # Let tqdm/logging output reach the terminal while the harness runs.
        # Capturing stdout/stderr here makes a full suite appear hung for hours.
        completed = subprocess.run(command, check=False, env=environment)
        if completed.returncode:
            raise RuntimeError(
                f"lm-eval failed with exit code {completed.returncode}; "
                "see the harness output above"
            )
        payload = _lm_eval_result_payload(Path(temp_dir))

    rows = _lm_eval_rows(payload, model, run_id, experiment_id, ts)
    results_dir.mkdir(exist_ok=True)
    out = results_dir / (model.replace("/", "__") + ".jsonl")
    with out.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"saved {len(rows)} lm-eval metric rows -> {out}")
    return rows


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("must be an integer") from e
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("must be an integer") from e
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], help="model id(s); repeatable; omit with --list")
    ap.add_argument("--url", default="http://localhost:11234/v1", help="OpenAI-compatible base URL")
    ap.add_argument("--runs", type=positive_int, default=1, help="repeat short-prompt round (default 1)")
    ap.add_argument("--iq", action="store_true", help="run the intelligence suite (accuracy) instead of speed tests")
    ap.add_argument("--suite", choices=("speed", "iq", "lm-eval"),
                    help="benchmark suite; --iq is retained as a shortcut for --suite iq")
    ap.add_argument("--tasks", default="hellaswag",
                    help="comma-separated lm-evaluation-harness tasks (default: hellaswag)")
    ap.add_argument("--limit", type=positive_int,
                    help="limit lm-evaluation-harness examples per task")
    ap.add_argument("--num-fewshot", type=nonnegative_int,
                    help="number of few-shot examples for lm-evaluation-harness")
    ap.add_argument("--lm-eval-backend", choices=("completions", "chat"), default="completions",
                    help="lm-eval API backend (default: completions; chat cannot run log-likelihood tasks)")
    ap.add_argument("--iq-tokens", type=positive_int, default=2048, help="max_tokens per IQ question (room for reasoning models)")
    ap.add_argument("--iq-categories", default="", help="comma-separated category filter, e.g. math-hard,retrieval (default: all)")
    ap.add_argument("--iq-runs", type=positive_int, default=1, help="repeat the IQ suite N times")
    ap.add_argument("--experiment-id",
                    help="explicit comparable-run group; otherwise sessions are created or resumed automatically")
    ap.add_argument("--new-session", action="store_true",
                    help="start a fresh auto-generated session instead of resuming an unfinished one")
    ap.add_argument("--list", action="store_true", help="list models served by --url and exit")
    ap.add_argument("--index", action="store_true", help="rebuild results.json index for dashboard.html")
    args = ap.parse_args()

    if args.index:
        RESULTS_DIR.mkdir(exist_ok=True)
        files = sorted(p.name for p in RESULTS_DIR.glob("*.jsonl"))
        (RESULTS_DIR.parent / "results.json").write_text(json.dumps(files))
        print(f"wrote results.json ({len(files)} files) for dashboard.html")
        return
    if args.list:
        print(f"models at {args.url}:")
        list_models(args.url)
        return
    if not args.model:
        ap.error("give --model (repeatable) or --list")
    if args.iq and args.suite:
        ap.error("use either --iq or --suite, not both")
    if args.experiment_id and args.new_session:
        ap.error("use either --experiment-id or --new-session, not both")
    suite = args.suite or ("iq" if args.iq else "speed")
    if suite == "lm-eval" and not {task.strip() for task in args.tasks.split(",") if task.strip()}:
        ap.error("--tasks must contain at least one task for --suite lm-eval")
    if suite != "iq" and args.iq_categories:
        ap.error("--iq-categories requires --iq or --suite iq")
    if suite == "iq":
        cats = {c.strip() for c in args.iq_categories.split(",") if c.strip()}
        if cats:
            known = {t["cat"] for t in IQ_TASKS}
            unknown = sorted(cats - known)
            if unknown:
                ap.error(f"unknown --iq-categories {unknown}; known: {sorted(known)}")
    if suite == "lm-eval" and shutil.which("lm-eval") is None:
        ap.error("lm-eval is not installed; install it with: python3 -m pip install 'lm-eval[api]'")

    for model in args.model:
        try:
            experiment_id, resumed = begin_experiment(
                args.url, model, suite, args.experiment_id, args.new_session,
                args.lm_eval_backend if suite == "lm-eval" else None,
                args.num_fewshot if suite == "lm-eval" else None,
            )
            if resumed:
                print(f"resuming unfinished {suite} experiment for {model}: {experiment_id}")
            else:
                print(f"starting {suite} experiment for {model}: {experiment_id}")
            if suite == "iq":
                cats = {c.strip() for c in args.iq_categories.split(",") if c.strip()} or None
                for _ in range(args.iq_runs):
                    run_iq(args.url, model, RESULTS_DIR, args.iq_tokens, cats, experiment_id)
            elif suite == "lm-eval":
                run_lm_eval(args.url, model, RESULTS_DIR, args.tasks, args.limit,
                            args.num_fewshot, experiment_id, args.lm_eval_backend)
            else:
                run_model(args.url, model, args.runs, RESULTS_DIR, experiment_id)
        except RuntimeError as e:
            ap.error(str(e))
        else:
            finish_experiment(model, suite, experiment_id)
    RESULTS_DIR.mkdir(exist_ok=True)
    files = sorted(p.name for p in RESULTS_DIR.glob("*.jsonl"))
    (RESULTS_DIR.parent / "results.json").write_text(json.dumps(files))
    print("\ndone. open the dashboard: python -m http.server 8001 && open http://localhost:8001/dashboard.html")


if __name__ == "__main__":
    main()
