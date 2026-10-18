import json
import random

from triage.agent import SymptomTriageAgent
from triage.features import FEATURES
from triage.environment import PatientSimulator, QueueSystem, ToolKit
from triage.logger import RunLogger


class ScriptedEnv:
    """Entorno de juguete con respuestas fijas, para probar reglas concretas."""

    def __init__(self, answers, ehr=None):
        self.answers, self.ehr, self.true_level = list(answers), ehr, 3
        self.tools = ToolKit(QueueSystem(random.Random(0)))
        self.asked = []

    def reset(self):
        return {"type": "initial", "patient_id": "T1", "age": 40, "chief_complaint": "x",
                "ehr": self.ehr, "queue": {"length": 5, "urgent_waiting": 1}}

    def step(self, feature):
        self.asked.append(feature)
        # respuestas guiadas; cuando se acaban, la más benigna de cada síntoma
        ans = self.answers.pop(0) if self.answers else FEATURES[feature].answers[0]
        return {"type": "answer", "feature": feature, "answer": ans,
                "queue": {"length": 5, "urgent_waiting": 1}}

    def clinician_label(self):
        return 3


def test_confirmed_red_flag_escalates():
    env = ScriptedEnv(["dificultad_severa", "dificultad_severa"])
    res = SymptomTriageAgent(policy="fixed").run_episode(env, 0)
    assert res["escalated"] and res["predicted"] == 1
    assert env.asked == ["breathing", "breathing"]          # se confirmó una vez
    assert env.tools.alerts and env.tools.outbox[0]["requires_human_validation"]


def test_unconfirmed_red_flag_is_discarded():
    env = ScriptedEnv(["dificultad_severa", "normal"])
    res = SymptomTriageAgent(policy="fixed").run_episode(env, 0)
    assert not res["escalated"]


def test_dont_know_on_red_flag_confirmation_escalates_for_safety():
    env = ScriptedEnv(["dificultad_severa", None])
    res = SymptomTriageAgent(policy="fixed").run_episode(env, 0)
    assert res["escalated"]


def test_no_repeated_questions_without_red_flag():
    env = PatientSimulator(seed=3, p_abandon=0, p_deteriorate=0, noise=0, p_unknown=0)
    ag = SymptomTriageAgent(policy="fixed")
    for i in range(30):
        ag.run_episode(env, i)
        assert ag.n_questions <= 7 + 3        # 7 preguntas + confirmaciones


def test_learning_updates_only_when_enabled():
    env = PatientSimulator(seed=1)
    ag = SymptomTriageAgent(learning=True)
    before = sum(ag.level_counts.values())
    ag.run_episode(env, 0)
    assert sum(ag.level_counts.values()) == before + 1
    ag.learning = False
    ag.run_episode(env, 1)
    assert sum(ag.level_counts.values()) == before + 1


def test_reproducible_with_seed():
    def run():
        env, ag = PatientSimulator(seed=7), SymptomTriageAgent()
        return [ag.run_episode(env, i) for i in range(50)]
    assert run() == run()


def test_log_has_timestamped_cycle(tmp_path):
    path = tmp_path / "log.jsonl"
    log = RunLogger(str(path))
    SymptomTriageAgent(logger=log).run_episode(PatientSimulator(seed=2), 0)
    log.close()
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    phases = {r["phase"] for r in recs}
    assert {"perception", "decision", "action", "feedback", "episode_end"} <= phases
    assert all("ts" in r for r in recs)
