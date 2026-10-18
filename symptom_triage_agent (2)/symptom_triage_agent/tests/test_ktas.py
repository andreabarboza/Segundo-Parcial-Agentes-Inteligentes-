"""Pruebas del modo con datos reales usando un CSV mínimo con el formato del dataset KTAS."""
from triage.agent import SymptomTriageAgent
from triage.ktas import (EXPECTED_COLUMNS, KTAS_FEATURES, KTAS_RED_FLAGS,
                         KTASReplayEnvironment, load_ktas)


def _write_csv(path, rows):
    """Escribe un CSV con separador ';' y codificación cp1254, como el original."""
    lines = [";".join(EXPECTED_COLUMNS)]
    for r in rows:
        base = {c: "" for c in EXPECTED_COLUMNS}
        base.update(r)
        lines.append(";".join(base[c] for c in EXPECTED_COLUMNS))
    path.write_bytes("\n".join(lines).encode("cp1254"))


ROWS = [
    # sano: decimal con coma, todo medido
    dict(Group="1", Age="45", **{"Arrival mode": "3"}, Injury="2", Chief_complain="leg pain",
         Mental="1", Pain="1", NRS_pain="3", SBP="120", DBP="80", HR="80", RR="18", BT="36,5",
         Saturation="98", KTAS_RN="4", KTAS_expert="4"),
    # inconsciente, con faltantes "??" y "#BOŞ!"
    dict(Group="2", Age="70", **{"Arrival mode": "2"}, Injury="1", Chief_complain="unresponsive",
         Mental="4", Pain="1", NRS_pain="#BOŞ!", SBP="85", DBP="50", HR="??", RR="30", BT="38,2",
         Saturation="??", KTAS_RN="2", KTAS_expert="1"),
    # hipoxia, sin dolor
    dict(Group="2", Age="30", **{"Arrival mode": "1"}, Injury="1", Chief_complain="dyspnea",
         Mental="1", Pain="0", NRS_pain="#BOŞ!", SBP="130", DBP="85", HR="110", RR="32", BT="37",
         Saturation="88", KTAS_RN="2", KTAS_expert="2"),
]


def test_loader_handles_missing_tokens_and_decimal_comma(tmp_path):
    p = tmp_path / "data.csv"
    _write_csv(p, ROWS)
    cases, info = load_ktas(p)
    assert len(cases) == 3 and info["filas_leidas"] == 3
    a, b, c = (cases.iloc[i] for i in range(3))
    assert (a.bt, a.spo2, a.pain, a.consciousness) == ("normal", "normal", "leve", "alerta")
    assert b.consciousness == "no_responde" and b.spo2 is None and b.hr is None   # '??' -> faltante
    assert b.pain is None                       # dolor sí, pero sin puntaje: NO se asume 0
    assert c.spo2 == "muy_baja" and c.rr == "grave" and c.pain == "leve"          # Pain=0 -> leve
    assert (a.arrival, b.arrival, c.arrival) == ("vehiculo_privado", "ambulancia", "caminando")
    assert list(cases.ktas_expert) == [4, 1, 2]


def test_loader_rejects_wrong_format(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c\n1,2,3\n")
    try:
        load_ktas(p)
        assert False, "debía fallar"
    except ValueError as e:
        assert "faltan columnas" in str(e)


def test_agent_runs_on_real_format_and_never_sees_forbidden_columns(tmp_path):
    p = tmp_path / "data.csv"
    _write_csv(p, ROWS)
    cases, _ = load_ktas(p)
    prohibidas = {"Diagnosis in ED", "Disposition", "Length of stay_min", "KTAS duration_min",
                  "Error_group", "mistriage", "KTAS_RN", "KTAS_expert", "ktas_rn", "ktas_expert"}
    assert not prohibidas & set(KTAS_FEATURES)          # el agente no tiene esas variables
    env = KTASReplayEnvironment(cases, seed=0)
    env.set_cases([0, 1, 2])
    agent = SymptomTriageAgent(features=KTAS_FEATURES, red_flags=KTAS_RED_FLAGS)
    res = [agent.run_episode(env, i) for i in range(3)]
    assert res[1]["escalated"] and res[1]["predicted"] == 1     # no responde -> bandera roja
    assert all(r["clinician"] == e for r, e in zip(res, [4, 1, 2]))
    assert len(env.tools.outbox) == 3
