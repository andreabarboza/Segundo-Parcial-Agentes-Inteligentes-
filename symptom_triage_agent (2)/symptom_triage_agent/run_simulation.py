"""Ejecuta el agente en el simulador: fase de entrenamiento (con aprendizaje) y de prueba.

Ejemplos:
    python run_simulation.py                       # agente de utilidad, 500 + 300 casos
    python run_simulation.py --policy fixed        # línea base: cuestionario rígido
    python run_simulation.py --no-learning         # sin aprendizaje
"""
import argparse
import csv
import os
import time

from triage.agent import SymptomTriageAgent
from triage.environment import PatientSimulator
from triage.logger import RunLogger


def metrics(rows):
    n = len(rows)
    if n == 0:
        return {}
    exact = sum(r["predicted"] == r["clinician"] for r in rows)
    within1 = sum(abs(r["predicted"] - r["clinician"]) <= 1 for r in rows)
    under = sum(r["predicted"] > r["clinician"] for r in rows)
    over = sum(r["predicted"] < r["clinician"] for r in rows)
    truth = sum(r["predicted"] == r["true_level"] for r in rows)
    return {"n": n, "agreement": exact / n, "agreement_truth": truth / n, "within_1": within1 / n,
            "under_triage": under / n, "over_triage": over / n,
            "avg_questions": sum(r["questions"] for r in rows) / n,
            "escalated": sum(r["escalated"] for r in rows) / n}


def show(title, m):
    print(f"{title:<24} n={m['n']:<4} coincidencia={m['agreement']:.1%} (vs real {m['agreement_truth']:.1%})  ±1 nivel={m['within_1']:.1%}  "
          f"sub-triage={m['under_triage']:.1%}  sobre-triage={m['over_triage']:.1%}  "
          f"preguntas={m['avg_questions']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy", choices=["utility", "fixed"], default="utility")
    ap.add_argument("--no-learning", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.policy}{'_nolearn' if args.no_learning else ''}_seed{args.seed}"
    logger = RunLogger(os.path.join(args.out, f"log_{tag}.jsonl"))
    env = PatientSimulator(seed=args.seed)
    agent = SymptomTriageAgent(policy=args.policy, learning=not args.no_learning, logger=logger)

    rows, t0 = [], time.time()
    for i in range(args.train):                       # entrenamiento (aprende)
        rows.append({**agent.run_episode(env, i), "phase": "train"})
    agent.learning = False                            # prueba: modelo congelado
    for i in range(args.test):
        rows.append({**agent.run_episode(env, args.train + i), "phase": "test"})
    logger.close()

    with open(os.path.join(args.out, f"results_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    train = [r for r in rows if r["phase"] == "train"]
    test = [r for r in rows if r["phase"] == "test"]
    print(f"\n== {tag}  ({time.time() - t0:.1f} s) ==")
    block = max(1, len(train) // 5)
    for b in range(0, len(train), block):
        show(f"entrenamiento {b}-{b + block}", metrics(train[b:b + block]))
    if test:
        show("PRUEBA (congelado)", metrics(test))
    print(f"\nLog: {args.out}/log_{tag}.jsonl | Resultados: {args.out}/results_{tag}.csv")


if __name__ == "__main__":
    main()
