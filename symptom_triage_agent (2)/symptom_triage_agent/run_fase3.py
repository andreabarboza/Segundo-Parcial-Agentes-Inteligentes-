"""Fase 3: pruebas basadas en las medidas de desempeño.

Lee los umbrales de fase3/umbrales.json (fijados ANTES de ejecutar), entrena el agente,
lo congela y lo prueba en 6 escenarios (normal, límite y 3 de falla esperada) con
10 repeticiones cada uno. Genera la tabla de resultados, un CSV con los conteos y un log
por ejecución.

    python run_fase3.py
    python run_fase3.py --out fase3_resultados --keep-logs all
"""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics as st
from datetime import datetime

from triage.agent import SymptomTriageAgent
from triage.environment import PatientSimulator
from triage.features import LEVELS
from triage.logger import RunLogger

CICLO = {"perception", "decision", "action", "feedback", "episode_end"}


def train_agent(rep, n_train):
    env, agent = PatientSimulator(seed=1000 + rep), SymptomTriageAgent(learning=True)
    for i in range(n_train):
        agent.run_episode(env, i)
    agent.learning = False                      # modelo congelado para las pruebas
    return agent


def run_scenario(agent, spec, seed, n, logger):
    """Atiende n pacientes capturando excepciones. Devuelve conteos y los resultados por caso."""
    env = PatientSimulator(seed=seed, **spec["sim"])
    agent.log = logger                          # el log del agente escribe en el archivo de esta ejecución
    c = dict(n=n, correct=0, under=0, urgent_total=0, urgent_hit=0, rf_total=0, rf_hit=0,
             questions=0, valid=0, errors=0)
    per_case = []
    for i in range(n):
        before = len(env.tools.outbox)
        try:
            res = agent.run_episode(env, i)
            summ = env.tools.outbox[-1] if len(env.tools.outbox) == before + 1 else None
            ok = (res["predicted"] in LEVELS and summ is not None
                  and summ.get("requires_human_validation") is True
                  and summ.get("suggested_level") in LEVELS)
            pred, esc, q = res["predicted"], res["escalated"], res["questions"]
            clin = res["clinician"]
        except Exception as e:                  # una caída cuenta como fallo, no detiene la prueba
            logger.log(i, "error", tipo=type(e).__name__, mensaje=str(e))
            ok, pred, esc, q, clin = False, None, False, 0, env.clinician_label()
            c["errors"] += 1
        c["valid"] += ok
        c["questions"] += q
        c["correct"] += pred == clin
        c["under"] += (pred is not None and pred > clin)
        if clin <= 2:
            c["urgent_total"] += 1
            c["urgent_hit"] += (pred is not None and pred <= 2)
        if env.true_red_flag:
            c["rf_total"] += 1
            c["rf_hit"] += bool(esc)
        per_case.append((pred, q, esc, clin))
    agent.log = RunLogger(None)
    return c, per_case


def traceable_fraction(path, n):
    """Fracción de episodios con el ciclo completo y marca de tiempo en el log."""
    eps = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "ts" in r:
                eps.setdefault(r["episode"], set()).add(r["phase"])
    return sum(CICLO <= eps.get(i, set()) for i in range(n)) / n


def metric(counts, name):
    n = sum(c["n"] for c in counts)
    g = lambda k: sum(c[k] for c in counts)
    if name == "agreement": return g("correct") / n
    if name == "under_triage": return g("under") / n
    if name == "avg_questions": return g("questions") / n
    if name == "valid_output": return g("valid") / n
    if name == "urgent_recall": return g("urgent_hit") / g("urgent_total") if g("urgent_total") else math.nan
    if name == "escalation_rate": return g("rf_hit") / g("rf_total") if g("rf_total") else math.nan
    if name == "traceable": return sum(c["trace"] * c["n"] for c in counts) / n
    if name == "reproducible": return sum(c["repro"] for c in counts) / len(counts)
    raise KeyError(name)


def fmt(name, x):
    if x != x: return "—"
    return f"{x:.2f}" if name == "avg_questions" else f"{x:.1%}"


def ms(name, vals):
    vals = [v for v in vals if v == v]
    if not vals: return "—"
    m, s = st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)
    return f"{fmt(name, m)} ± {s:.2f}" if name == "avg_questions" else f"{fmt(name, m)} ± {s * 100:.1f} pp"


def cumple(op, thr, x):
    return x == x and (x >= thr - 1e-12 if op == ">=" else x <= thr + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbrales", default="fase3/umbrales.json")
    ap.add_argument("--out", default="fase3_resultados")
    ap.add_argument("--keep-logs", choices=["first", "all"], default="first",
                    help="first: conserva los logs de la repetición 1 (todos se verifican igual)")
    args = ap.parse_args()

    raw = open(args.umbrales, "rb").read()
    cfg = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    R, N, NT = cfg["repeticiones"], cfg["pacientes_por_escenario"], cfg["pacientes_entrenamiento_previo"]
    esc = cfg["escenarios"]
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    print(f"Umbrales: {args.umbrales}\nSHA-256 de los umbrales: {sha}\n"
          f"{R} repeticiones x {len(esc)} escenarios x {N} pacientes (entrenamiento previo: {NT})\n")

    data = {s: [] for s in esc}                    # escenario -> [conteos por repetición]
    for rep in range(1, R + 1):
        agent = train_agent(rep, NT)
        for idx, (sid, spec) in enumerate(esc.items()):
            path = os.path.join(args.out, "logs", f"{sid}_rep{rep:02d}.jsonl")
            logger = RunLogger(path)
            c, cases = run_scenario(agent, spec, seed=rep * 100 + idx, n=N, logger=logger)
            logger.close()
            c["trace"], c["rep"] = traceable_fraction(path, N), rep
            if sid == "E1":                         # reproducibilidad: repetir con la misma semilla
                agent2 = train_agent(rep, NT)
                _, cases2 = run_scenario(agent2, spec, seed=rep * 100 + idx, n=N, logger=RunLogger(None))
                c["repro"] = 1.0 if cases == cases2 else 0.0
            else:
                c["repro"] = math.nan
            if args.keep_logs == "first" and rep > 1:
                os.remove(path)
            data[sid].append(c)
        print(f"  repetición {rep}/{R} lista")

    # ---- CSV con los conteos por repetición y escenario
    with open(os.path.join(args.out, "resultados_por_corrida.csv"), "w", newline="", encoding="utf-8") as fh:
        keys = ["escenario", "tipo", "rep"] + [k for k in data["E1"][0] if k != "rep"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for sid, lst in data.items():
            for c in lst:
                w.writerow({"escenario": sid, "tipo": esc[sid]["tipo"], **c})

    # ---- tabla principal
    total_all = sum(c["n"] for lst in data.values() for c in lst)
    rows, resumen = [], []
    for m in cfg["medidas"]:
        accept = m["escenario_aceptacion"] if isinstance(m["escenario_aceptacion"], list) else [m["escenario_aceptacion"]]
        per_rep = [metric([data[s][r] for s in accept], m["metrica"]) for r in range(R)]
        mean = st.mean([v for v in per_rep if v == v]) if any(v == v for v in per_rep) else math.nan
        ok = cumple(m["operador"], m["umbral"], mean)
        if m["metrica"] == "urgent_recall":
            casos = sum(data[s][r]["urgent_total"] for s in accept for r in range(R))
            nota = f"{casos} pacientes graves (de {len(accept) * R * N})"
        elif m["metrica"] == "escalation_rate":
            casos = sum(data[s][r]["rf_total"] for s in accept for r in range(R))
            nota = f"{casos} pacientes con riesgo vital real (de {len(accept) * R * N})"
        elif m["metrica"] == "reproducible":
            nota = f"{R} repeticiones"
        else:
            casos = sum(data[s][r]["n"] for s in accept for r in range(R))
            nota = f"{casos} pacientes"
        thr = m["umbral"]
        umbral_txt = f"{m['operador']} {thr:.1f}" if m["metrica"] == "avg_questions" else f"{m['operador']} {thr:.0%}"
        evid = f"logs/{accept[0]}_rep01.jsonl · resultados_por_corrida.csv (escenario {', '.join(accept)})"
        rows.append((m["id"], m["nombre"], m["formula"], umbral_txt, nota, ms(m["metrica"], per_rep),
                     "Sí" if ok else "No", evid))
        resumen.append({"id": m["id"], "medida": m["nombre"], "media": mean, "umbral": thr,
                        "operador": m["operador"], "cumple": bool(ok)})

    # ---- cuadrícula medida x escenario (informativa: casos límite y de falla)
    grid_metrics = [("M1", "agreement"), ("M2", "under_triage"), ("M3", "urgent_recall"),
                    ("M4", "escalation_rate"), ("M5", "avg_questions"), ("M6", "valid_output"), ("M7", "traceable")]
    names = {m["id"]: m["nombre"] for m in cfg["medidas"]}
    L = []
    L.append("# Fase 3 — Resultados de las pruebas\n")
    L.append(f"- Fecha de ejecución: {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"- SHA-256 de `umbrales.json` (fijado antes de ejecutar): `{sha}`")
    L.append(f"- {R} repeticiones por escenario, {N} pacientes por escenario y repetición, entrenamiento previo de {NT} pacientes; "
             f"total de pacientes ejecutados: {total_all}")
    L.append("- Resultado = media ± desviación estándar entre repeticiones (pp = puntos porcentuales)\n")
    L.append("## Tabla de resultados\n")
    L.append("| ID | Medida de desempeño | Fórmula | Umbral | Casos ejecutados | Resultado obtenido | ¿Cumple? | Evidencia |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    L.append("\n## Escenarios\n")
    L.append("| ID | Tipo | Descripción |\n|---|---|---|")
    for sid, spec in esc.items():
        L.append(f"| {sid} | {spec['tipo']} | {spec['descripcion']} |")
    L.append("\n## Resultados por escenario (casos límite y de falla esperada; informativos)\n")
    L.append("| Medida | " + " | ".join(esc) + " |\n|---|" + "---|" * len(esc))
    for mid, met in grid_metrics:
        cells = [ms(met, [metric([data[s][r]], met) for r in range(R)]) for s in esc]
        L.append(f"| {mid} {names[mid]} | " + " | ".join(cells) + " |")
    open(os.path.join(args.out, "tabla_resultados.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(resumen, open(os.path.join(args.out, "resumen.json"), "w"), indent=1, ensure_ascii=False)

    print("\n" + "\n".join(L))
    print(f"\nGuardado en {args.out}/ (tabla_resultados.md, resultados_por_corrida.csv, resumen.json, logs/)")


if __name__ == "__main__":
    main()
