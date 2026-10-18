"""Pruebas de robustez (Fase 3): entradas inválidas y sensores sin dato no deben tumbar al agente."""
import json

from triage.agent import SymptomTriageAgent
from triage.logger import RunLogger
from tests.test_agent import ScriptedEnv


class QueueDownEnv(ScriptedEnv):
    """Igual que ScriptedEnv pero la API de la cola no responde (queue=None)."""

    def reset(self):
        p = super().reset(); p["queue"] = None; return p

    def step(self, feature):
        p = super().step(feature); p["queue"] = None; return p


def test_invalid_answers_are_treated_as_unknown_and_logged(tmp_path):
    path = tmp_path / "log.jsonl"
    log = RunLogger(str(path))
    env = ScriptedEnv(["???", 42, True, "asdf", ""] + [None] * 10)
    res = SymptomTriageAgent(policy="fixed", logger=log).run_episode(env, 0)
    log.close()
    assert res["predicted"] in (1, 2, 3, 4, 5)
    assert env.tools.outbox[0]["requires_human_validation"] is True
    avisos = [json.loads(l) for l in path.read_text().splitlines() if '"warning"' in l]
    assert len(avisos) >= 5 and all(a["tipo"] == "entrada_invalida" for a in avisos)


def test_queue_sensor_down_does_not_crash(tmp_path):
    path = tmp_path / "log.jsonl"
    log = RunLogger(str(path))
    env = QueueDownEnv([])
    res = SymptomTriageAgent(policy="fixed", logger=log).run_episode(env, 0)
    log.close()
    assert res["predicted"] in (1, 2, 3, 4, 5) and len(env.tools.outbox) == 1
    assert any('"sensor_sin_dato"' in l for l in path.read_text().splitlines())


def test_unknown_percept_type_is_ignored():
    ag = SymptomTriageAgent()
    ag.start_episode(0)
    ag.perceive({"type": "algo_raro", "queue": {"length": 3}})
    assert ag.queue_len == 3 and ag.evidence == []
