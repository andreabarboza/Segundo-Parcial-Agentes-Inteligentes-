"""Evalúa el agente con casos REALES del dataset KTAS (Kaggle).

    python run_ktas.py --data data.csv                 # 5 particiones aleatorias 70/30
    python run_ktas.py --data data.csv --seeds 10 --train-frac 0.8

Por cada semilla: se parte el dataset en entrenamiento y prueba (estratificado por nivel
experto); el agente aprende con la corrección del experto en entrenamiento y se evalúa con el
modelo congelado en prueba. Referencias de comparación: la enfermera (KTAS_RN) y la clase
mayoritaria.

Limitación: el dataset no trae identificador de paciente, así que no se puede garantizar que
una misma persona no aparezca en entrenamiento y prueba.
"""
import argparse
import csv
import os
import random
import statistics as st

from run_simulation import metrics
from triage.agent import SymptomTriageAgent
from triage.ktas import (KTAS_FEATURES, KTAS_RED_FLAGS, KTASReplayEnvironment,
                         describe_ranges, load_ktas)
from triage.logger import RunLogger

CONFIGS = {
    "Utilidad + aprendizaje": dict(policy="utility", learning=True),
    "Utilidad sin aprendizaje": dict(policy="utility", learning=False),
    "Cuestionario fijo + aprendizaje": dict(policy="fixed", learning=True),
}


def stratified_split(levels, frac, seed):
    rng, train, test = random.Random(seed), [], []
    for lv in sorted(set(levels)):
        idx = [i for i, x in enumerate(levels) if x == lv]
        rng.shuffle(idx)
        cut = round(len(idx) * frac)
        train += idx[:cut]
        test += idx[cut:]
    rng.shuffle(train)
    return train, test


def extra_metrics(rows):
    m = metrics(rows)
    urg = [r for r in rows if r["clinician"] <= 2]          # casos realmente urgentes (1-2)
    m["urgent_recall"] = (sum(r["predicted"] <= 2 for r in urg) / len(urg)) if urg else float("nan")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", default="runs_ktas")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases, info = load_ktas(args.data)
    print(f"Casos válidos: {info['filas_validas']}/{info['filas_leidas']}   SHA-256: {info['sha256']}")
    print("(la guía consultada reporta 0e2c088e358fd4cdfd0dcc2fd4c2f085aa303856e3a6df8ceb44a69cef8ed2de;"
          " si difiere, es otra versión del archivo)")
    if info["no_numericos_inesperados"]:
        print("AVISO: valores no numéricos inesperados (se tratan como faltantes):",
              info["no_numericos_inesperados"])
    print("\nDatos faltantes por síntoma:",
          {k: f"{v:.0%}" for k, v in info["faltantes_por_sintoma"].items()})
    print("\nRangos (revisa que las unidades tengan sentido: BT en °C, Saturation en %, etc.):")
    print(describe_ranges(args.data).round(1).to_string())
    print(f"\nAcuerdo enfermera vs. experto en TODO el dataset: {info['acuerdo_enfermera_vs_experto']:.1%}")
    print("(esta es la referencia humana: el agente debería compararse contra este número)\n")

    levels = cases.ktas_expert.tolist()
    per_cfg = {name: [] for name in list(CONFIGS) + ["Enfermera (KTAS_RN)", "Clase mayoritaria"]}
    rows_out = []
    for seed in range(args.seeds):
        train, test = stratified_split(levels, args.train_frac, seed)
        for name, cfg in CONFIGS.items():
            logger = RunLogger(os.path.join(args.out, f"log_ktas_seed{seed}.jsonl")
                               if (seed == 0 and name.startswith("Utilidad +")) else None)
            env = KTASReplayEnvironment(cases, seed=seed)
            agent = SymptomTriageAgent(features=KTAS_FEATURES, red_flags=KTAS_RED_FLAGS,
                                       logger=logger, **cfg)
            n = 0
            if cfg["learning"]:
                # Entrenamiento: el agente de utilidad explora (pregunta todo) para aprender.
                agent.explore = cfg["policy"] == "utility"
                env.set_cases(train)
                for _ in train:
                    agent.run_episode(env, n); n += 1
            agent.learning, agent.explore = False, False      # prueba: modelo congelado
            env.set_cases(test)
            rows = [agent.run_episode(env, n + i) for i in range(len(test))]
            logger.close()
            per_cfg[name].append(extra_metrics(rows))
            rows_out += [{"seed": seed, "config": name, **r} for r in rows]
        # Referencias
        sub = cases.iloc[test]
        nurse = [{"predicted": int(r.ktas_rn), "clinician": int(r.ktas_expert),
                  "true_level": int(r.ktas_expert), "questions": 0, "escalated": False}
                 for r in sub.itertuples() if r.ktas_rn is not None and r.ktas_rn == r.ktas_rn]
        per_cfg["Enfermera (KTAS_RN)"].append(extra_metrics(nurse))
        moda = st.mode([levels[i] for i in train])
        maj = [{"predicted": moda, "clinician": levels[i], "true_level": levels[i],
                "questions": 0, "escalated": False} for i in test]
        per_cfg["Clase mayoritaria"].append(extra_metrics(maj))

    def ms(name, key):
        v = [m[key] for m in per_cfg[name]]
        return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

    print(f"== Prueba congelada · {args.seeds} particiones {args.train_frac:.0%}/{1 - args.train_frac:.0%} ==")
    print(f"{'Configuración':<34}{'Coincidencia':>16}{'±1 nivel':>10}{'Sub-triage':>12}"
          f"{'Sobre-triage':>14}{'Urgentes (1-2)':>16}{'Preguntas':>11}")
    for name in per_cfg:
        a, s = ms(name, "agreement")
        print(f"{name:<34}{a:>10.1%} ±{s:.1%}{ms(name, 'within_1')[0]:>10.1%}"
              f"{ms(name, 'under_triage')[0]:>12.1%}{ms(name, 'over_triage')[0]:>14.1%}"
              f"{ms(name, 'urgent_recall')[0]:>16.1%}{ms(name, 'avg_questions')[0]:>11.2f}")
    print("\n'Urgentes (1-2)' = de los casos que el experto marcó nivel 1 o 2, % que el agente también"
          " clasificó como 1 o 2 (sensibilidad de los casos graves).")

    path = os.path.join(args.out, "resultados_prueba.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f"\nGuardado: {path} y {args.out}/log_ktas_seed0.jsonl")


if __name__ == "__main__":
    main()
