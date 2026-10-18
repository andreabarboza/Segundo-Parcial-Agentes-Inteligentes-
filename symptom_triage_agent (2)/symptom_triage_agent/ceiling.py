"""Techo de exactitud del simulador: qué tan bien podría clasificar un agente PERFECTO
que conoce las tablas reales y hace las 7 preguntas + historia clínica (con el mismo
10 % de ruido en las respuestas). Sirve para interpretar el % de coincidencia obtenido.
"""
import random

from triage.features import FEATURES, LEVELS, SIM_PRIOR, likelihood_table

rng, NOISE, N = random.Random(1), 0.10, 20000
tabs = {L: {f: likelihood_table(F, L) for f, F in FEATURES.items()} for L in LEVELS}
ok = 0
for _ in range(N):
    L = rng.choices(LEVELS, SIM_PRIOR)[0]
    post = dict(zip(LEVELS, SIM_PRIOR))
    for f, F in FEATURES.items():
        k = len(F.answers)
        a = rng.choices(range(k), tabs[L][f])[0]
        if rng.random() < NOISE:
            a = rng.choice([j for j in range(k) if j != a])
        for l in LEVELS:
            post[l] *= (1 - NOISE) * tabs[l][f][a] + NOISE / k
    ok += max(post, key=post.get) == L
print(f"Techo de exactitud vs nivel real: {ok / N:.1%}")
print("Contra el clínico (88 % de acuerdo con el nivel real) el techo práctico es ~75 %.")
