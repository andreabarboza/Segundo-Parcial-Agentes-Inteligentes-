"""Diagnóstico de las medidas incumplidas (M1 y M4). Se ejecuta desde la raíz del proyecto:
    python fase3/diagnostico.py
No cambia los umbrales ni los resultados oficiales: solo ayuda a explicar POR QUÉ fallaron.
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triage.agent as A
from run_fase3 import train_agent
from triage.environment import PatientSimulator
from triage.features import FEATURES, RED_FLAGS

R, N, NT = 10, 100, 500


def m1_variantes():
    """¿Cuánto del incumplimiento de M1 se explica por cada factor?"""
    variantes = {
        "Referencia (escenario E1 oficial)": dict(w=3.0, sim={}),
        "Sin desacuerdo entre clínicos (etiqueta 100 % fiable)": dict(w=3.0, sim=dict(clinician_agreement=1.0)),
        "Costo simétrico de errores (W_UNDER=1)": dict(w=1.0, sim={}),
        "Sin ruido ni deterioro en las respuestas": dict(w=3.0, sim=dict(noise=0.0, p_unknown=0.0, p_deteriorate=0.0, p_abandon=0.0)),
        "Las tres cosas anteriores a la vez": dict(w=1.0, sim=dict(clinician_agreement=1.0, noise=0.0, p_unknown=0.0, p_deteriorate=0.0, p_abandon=0.0)),
    }
    print("== M1: ¿por qué solo ~60 % de coincidencia? (10 repeticiones x 100 pacientes) ==")
    print(f"{'Variante':<56}{'Coincidencia':>14}{'Sub-triage':>12}{'Sobre-triage':>14}{'Preguntas':>11}")
    for nombre, v in variantes.items():
        A.W_UNDER = v["w"]
        ag_, ub_, ov_, qs_ = [], [], [], []
        for rep in range(1, R + 1):
            agent = train_agent(rep, NT)
            env = PatientSimulator(seed=rep * 100, **v["sim"])
            rows = [agent.run_episode(env, i) for i in range(N)]
            ag_.append(sum(r["predicted"] == r["clinician"] for r in rows) / N)
            ub_.append(sum(r["predicted"] > r["clinician"] for r in rows) / N)
            ov_.append(sum(r["predicted"] < r["clinician"] for r in rows) / N)
            qs_.append(sum(r["questions"] for r in rows) / N)
        print(f"{nombre:<56}{st.mean(ag_):>14.1%}{st.mean(ub_):>12.1%}{st.mean(ov_):>14.1%}{st.mean(qs_):>11.2f}")
    A.W_UNDER = 3.0


class _Espia(PatientSimulator):
    """Registra qué respuestas dio el paciente a cada pregunta, para auditar M4."""
    def reset(self):
        self.rastro = []
        return super().reset()

    def step(self, feature):
        p = super().step(feature)
        self.rastro.append((feature, p.get("answer")))
        return p


def m4_causas():
    """De los pacientes con riesgo vital real que NO se escalaron, ¿qué pasó?"""
    flagged = {(f, a) for (f, a) in RED_FLAGS}
    c = dict(total=0, escalados=0, nunca_pregunto=0, no_se_vio=0, no_confirmo=0)
    for rep in range(1, R + 1):
        agent = train_agent(rep, NT)
        env = _Espia(seed=rep * 100 + 2, level_prior=[0.5, 0.5, 0, 0, 0])
        for i in range(N):
            res = agent.run_episode(env, i)
            if not env.true_red_flag:
                continue
            c["total"] += 1
            if res["escalated"]:
                c["escalados"] += 1
                continue
            verdad = [f for (f, a) in RED_FLAGS
                      if FEATURES[f].answers[env.true_values[f]] == a]
            preguntadas = [(f, a) for f, a in env.rastro if f in verdad]
            if not preguntadas:
                c["nunca_pregunto"] += 1
            elif not any((f, a) in flagged for f, a in preguntadas):
                c["no_se_vio"] += 1        # preguntó, pero la respuesta fue errónea o "no sé"
            else:
                c["no_confirmo"] += 1      # vio la bandera, pero la confirmación la contradijo
    t, ne = c["total"], c["total"] - c["escalados"]
    print("\n== M4: pacientes con riesgo vital real en el escenario E3 ==")
    print(f"Total: {t}   escalados: {c['escalados']} ({c['escalados'] / t:.1%})   no escalados: {ne}")
    for k, txt in [("nunca_pregunto", "El agente nunca preguntó por el signo de riesgo vital (clasificó antes)"),
                   ("no_se_vio", "Preguntó, pero la respuesta fue errónea o 'no sé' (la bandera no se vio)"),
                   ("no_confirmo", "Vio la bandera, pero la segunda respuesta la contradijo")]:
        print(f"  - {txt}: {c[k]} ({c[k] / ne:.0%} de los no escalados)" if ne else "")


def palancas():
    """¿Qué cambios en el agente mejorarían M1 y M4? (hipótesis; NO son los resultados oficiales)"""
    variantes = {
        "Agente actual (oficial)": dict(nt=NT, q=0.03, scr=()),
        "+ tamizaje obligatorio (respiración y conciencia)": dict(nt=NT, q=0.03, scr=("breathing", "consciousness")),
        "+ más entrenamiento (3000 pacientes)": dict(nt=3000, q=0.03, scr=()),
        "+ costo de pregunta bajo (0,005)": dict(nt=NT, q=0.005, scr=()),
        "Las tres mejoras juntas": dict(nt=3000, q=0.005, scr=("breathing", "consciousness")),
    }
    print("\n== Qué cambiaría en el agente (hipótesis, 10 repeticiones) ==")
    print(f"{'Variante':<52}{'M1 coincid.':>12}{'M2 sub-tr.':>11}{'M3 graves':>11}{'M4 escal.':>11}{'M5 preg.':>10}")
    for nombre, v in variantes.items():
        m1, m2, m5, m3, m4 = [], [], [], [], []
        for rep in range(1, R + 1):
            agent = train_agent(rep, v["nt"])
            agent.q_cost, agent.screening = v["q"], v["scr"]
            env = PatientSimulator(seed=rep * 100)                                   # E1
            rows = [agent.run_episode(env, i) for i in range(N)]
            m1.append(sum(r["predicted"] == r["clinician"] for r in rows) / N)
            m2.append(sum(r["predicted"] > r["clinician"] for r in rows) / N)
            m5.append(sum(r["questions"] for r in rows) / N)
            env = PatientSimulator(seed=rep * 100 + 2, level_prior=[0.5, 0.5, 0, 0, 0])   # E3
            rf = gr = hit = hit3 = 0
            for i in range(N):
                r = agent.run_episode(env, i)
                if r["clinician"] <= 2:
                    gr += 1; hit3 += r["predicted"] <= 2
                if env.true_red_flag:
                    rf += 1; hit += r["escalated"]
            m3.append(hit3 / gr); m4.append(hit / rf if rf else float("nan"))
        print(f"{nombre:<52}{st.mean(m1):>12.1%}{st.mean(m2):>11.1%}{st.mean(m3):>11.1%}"
              f"{st.mean(m4):>11.1%}{st.mean(m5):>10.2f}")


if __name__ == "__main__":
    m1_variantes()
    m4_causas()
    palancas()
