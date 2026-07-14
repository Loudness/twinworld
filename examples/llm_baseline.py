"""Head-to-head: a local LLM vs dowhat-with-gate, on ARC and CausalARC.

Part A (always): the known-fitting ARC tasks plus a few more corpus tasks.
The LLM always answers; dowhat answers only when its confidence gate is high
— the calibration contrast is the point, not just the solve counts.

Part B (--causalarc): CausalARC counting tasks with and without labelled
counterfactual demonstration pairs in the prompt — the benchmark's own
"counterfactual reasoning with in-context learning" setting, run locally.

Usage: python examples/llm_baseline.py [N_EXTRA] [--causalarc]
Endpoint: DOWHAT_LLM_URL (default http://localhost:11434/v1/chat/completions)
"""

import sys
import time

import dowhat
from dowhat import UnsolvedTaskError, as_grid
from dowhat.baselines.llm import llm_predict, reachable
from dowhat.domains.arc import iter_tasks, load_task

KNOWN_IDS = ["25ff71a9", "42a50994", "5582e5ca", "9565186b", "a79310a0", "e0fb7511", "b230c067"]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
n_extra = int(args[0]) if args else 8
run_causalarc = "--causalarc" in sys.argv

if not reachable():
    sys.exit("LLM endpoint not reachable — set DOWHAT_LLM_URL or start the server")

tasks = [load_task(tid) for tid in KNOWN_IDS]
for task in iter_tasks():
    if len(tasks) >= len(KNOWN_IDS) + n_extra:
        break
    if task.task_id not in KNOWN_IDS and len(task.colours()) <= 6:
        tasks.append(task)

print(f"ARC head-to-head over {len(tasks)} tasks "
      f"({len(KNOWN_IDS)} known-fitting + {len(tasks) - len(KNOWN_IDS)} extra)\n")
llm_right = dowhat_right = dowhat_wrong = abstained = unsolved = 0
for task in tasks:
    expected = as_grid(task.test[0][1])
    t0 = time.perf_counter()
    guess = llm_predict(task)
    llm_ms = (time.perf_counter() - t0) * 1000
    llm_ok = guess == expected
    llm_right += llm_ok

    t0 = time.perf_counter()
    try:
        rep = dowhat.model(task)
        prediction, report = dowhat.predict(rep)
        if prediction is None:
            ours = "ABSTAINED"
            abstained += 1
        elif prediction[0] == expected:
            ours = "correct"
            dowhat_right += 1
        else:
            ours = "WRONG"
            dowhat_wrong += 1
    except UnsolvedTaskError:
        ours = "no fit"
        unsolved += 1
    our_ms = (time.perf_counter() - t0) * 1000
    print(f"  {task.task_id}: LLM {'correct' if llm_ok else 'wrong  '} ({llm_ms:6.0f}ms) "
          f"| dowhat {ours:9s} ({our_ms:5.0f}ms)", flush=True)

print(f"\n  LLM    : {llm_right}/{len(tasks)} correct (always answers)")
print(f"  dowhat : {dowhat_right} correct, {dowhat_wrong} wrong, "
      f"{abstained} abstained, {unsolved} no fit")
print("  calibration: dowhat's wrong-when-confident count is the number to watch")

if run_causalarc:
    from dowhat.domains.causalarc import load_causalarc

    causal_tasks = load_causalarc("counting")
    print(f"\nCausalARC counting ({len(causal_tasks)} tasks): does counterfactual "
          f"feedback help the LLM?\n")
    plain_right = cf_right = 0
    for ct in causal_tasks:
        expected = as_grid(ct.task.test[0][1])
        plain = llm_predict(ct.task) == expected
        with_cf = llm_predict(ct.task, cf_demos=[d[:2] for d in ct.cf_demos]) == expected
        plain_right += plain
        cf_right += with_cf
        print(f"  {ct.task.task_id}: plain {'correct' if plain else 'wrong'} | "
              f"with counterfactuals {'correct' if with_cf else 'wrong'}", flush=True)
    print(f"\n  plain prompts        : {plain_right}/{len(causal_tasks)}")
    print(f"  + counterfactual demos: {cf_right}/{len(causal_tasks)}")