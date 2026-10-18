"""Symptom Triage Agent: agente basado en utilidad con aprendizaje."""

from dataclasses import dataclass, field

from .features import (FEATURES as DEFAULT_FEATURES, LEVELS, LEVEL_NAMES,
                       PROTOCOL_PRIOR, RED_FLAGS as DEFAULT_RED_FLAGS,
                       protocol_likelihood)
from .logger import RunLogger

W_UNDER = 3.0   # costo por nivel de sub-triage (pred. menos urgente que el real)
W_OVER = 1.0    # costo por nivel de sobre-triage
N0_LIK, N0_PRIOR = 10.0, 20.0   # fuerza del conocimiento inicial (pseudo-conteos)
NOISE_EPS = 0.10                # el agente asume que ~10% de las respuestas son erróneas


def triage_cost(true_level, pred_level):
    d = pred_level - true_level
    return W_UNDER * d if d > 0 else W_OVER * (-d)


@dataclass
class Action:
    kind: str                 # ASK | CLASSIFY | ESCALATE
    feature: str = None
    level: int = None
    reason: str = ""
    info: dict = field(default_factory=dict)


class SymptomTriageAgent:
    def __init__(self, policy="utility", learning=True, logger=None,
                 q_cost=0.03, max_questions=8, p_answer=0.9,
                 features=None, red_flags=None, explore=False, screening=()):
        assert policy in ("utility", "fixed")
        # Conjunto de síntomas y banderas rojas configurables: el simulador usa los de
        # features.py; el modo con datos reales (triage/ktas.py) usa los suyos.
        # explore=True: durante el entrenamiento pregunta TODO, para recoger evidencia sin
        # sesgo y poder aprender. Evita la trampa "no pregunto porque creo que no aporta, y
        # por eso nunca aprendo que sí aportaba". En la prueba se pone en False.
        self.explore = explore
        # screening: síntomas que se preguntan SIEMPRE primero (tamizaje de riesgo vital, como
        # el ABCDE de un triage real). Por defecto vacío: el agente decide solo por utilidad.
        self.screening = tuple(screening)
        self.features = DEFAULT_FEATURES if features is None else features
        self.red_flags = DEFAULT_RED_FLAGS if red_flags is None else red_flags
        self.policy, self.learning = policy, learning
        self.log = logger or RunLogger(None)
        self.q_cost, self.max_questions, self.p_answer = q_cost, max_questions, p_answer
        # Conocimiento (largo plazo): pseudo-conteos que el aprendizaje va actualizando
        self.level_counts = {L: N0_PRIOR * p for L, p in zip(LEVELS, PROTOCOL_PRIOR)}
        self.counts = {L: {n: [N0_LIK * p for p in protocol_likelihood(f, L)]
                           for n, f in self.features.items()} for L in LEVELS}
        self.episode = -1

    # ------------------------------------------------------------------ modelo --
    def _lik(self, feature, level, ans_idx):
        row = self.counts[level][feature]
        p = row[ans_idx] / sum(row)
        return (1 - NOISE_EPS) * p + NOISE_EPS / len(row)

    def _posterior(self, evidence):
        tot = sum(self.level_counts.values())
        post = {L: self.level_counts[L] / tot for L in LEVELS}
        for f, a in evidence:
            for L in LEVELS:
                post[L] *= self._lik(f, L, a)
        z = sum(post.values())
        return {L: p / z for L, p in post.items()}

    @staticmethod
    def _best_prediction(post):
        """Nivel que minimiza el costo esperado (empates -> el más urgente)."""
        best, best_cost = None, float("inf")
        for pred in LEVELS:
            c = sum(post[L] * triage_cost(L, pred) for L in LEVELS)
            if c < best_cost - 1e-12:
                best, best_cost = pred, c
        return best, best_cost

    def _expected_gain(self, feature, post):
        """Valor esperado de la información: cuánto baja el costo esperado al preguntar."""
        _, base = self._best_prediction(post)
        k = len(self.features[feature].answers)
        after = 0.0
        for a in range(k):
            p_a = sum(post[L] * self._lik(feature, L, a) for L in LEVELS)
            post_a = {L: post[L] * self._lik(feature, L, a) / p_a for L in LEVELS}
            after += p_a * self._best_prediction(post_a)[1]
        return self.p_answer * (base - after)

    # ------------------------------------------------------------------ episodio --
    def start_episode(self, episode_id):
        self.episode = episode_id
        self.evidence, self.asked = [], set()       # memoria de corto plazo
        self.queue_len, self.patient_left, self.red_flag = 0, False, None
        self.pending_flag = None    # bandera roja vista una vez, aún sin confirmar
        self.patient, self.n_questions, self.ehr_used = {}, 0, False
        self.patient = {"id": f"sin_id_{episode_id}", "age": None, "chief_complaint": None}

    # 1) PERCEPCIÓN ---------------------------------------------------------------
    def _valid(self, feature, ans):
        """True si `ans` es una respuesta permitida para ese síntoma."""
        return feature in self.features and ans in self.features[feature].answers

    def _warn(self, tipo, **data):
        """Registra una anomalía de entrada sin interrumpir la atención del paciente."""
        self.log.log(self.episode, "warning", tipo=tipo, **data)

    def perceive(self, percept):
        """Recibe una percepción del entorno y actualiza la memoria/creencia.

        Es tolerante a fallas: una entrada fuera de dominio se trata como "no sé" y un
        sensor sin dato (p. ej. la API de la cola caída) mantiene el último valor conocido.
        En ambos casos se deja una advertencia en el log. Nunca lanza excepción por datos
        malos: un paciente no debe quedar sin clasificar por un fallo de entrada.
        """
        self.log.log(self.episode, "perception", percept=percept)
        queue = percept.get("queue")
        if queue is None:                                     # sensor: cola (sin dato)
            self._warn("sensor_sin_dato", sensor="cola", usa="ultimo_valor_conocido",
                       valor=self.queue_len)
        else:
            self.queue_len = queue["length"]
        kind = percept.get("type")
        if kind == "initial":
            self.patient = {"id": percept.get("patient_id"), "age": percept.get("age"),
                            "chief_complaint": percept.get("chief_complaint")}
            for f, ans in (percept.get("ehr") or {}).items():  # sensor: historia clínica
                if self._valid(f, ans):
                    self.evidence.append((f, self.features[f].answers.index(ans)))
                    self.asked.add(f)
                    self.ehr_used = True
                else:
                    self._warn("entrada_invalida", fuente="historia_clinica", feature=f, valor=repr(ans))
        elif kind == "answer":                                # sensor: chat
            f, ans = percept.get("feature"), percept.get("answer")
            if ans is not None and not self._valid(f, ans):
                self._warn("entrada_invalida", fuente="chat", feature=f, valor=repr(ans))
                ans = None                                    # se trata como "no sé"
            if ans is not None:                               # "no sé" no aporta evidencia
                self.evidence.append((f, self.features[f].answers.index(ans)))
            self._update_red_flag(f, ans)
        elif kind == "patient_left":
            self.patient_left = True
            if self.pending_flag:           # no se pudo descartar: se trata como real
                self.red_flag, self.pending_flag = self.pending_flag, None
        else:
            self._warn("percepcion_desconocida", valor=repr(kind))

    def _update_red_flag(self, f, ans):
        """Una respuesta de riesgo vital se confirma una vez antes de escalar, para que
        un error de respuesta no dispare una falsa alarma. Si no se puede confirmar
        ("no sé"), se escala por seguridad."""
        if self.pending_flag and self.pending_flag[0] == f:       # respuesta de confirmación
            flag, self.pending_flag = self.pending_flag, None
            if ans is None or (f, ans) in self.red_flags:
                self.red_flag = flag if ans is None else (f, ans, self.red_flags[(f, ans)])
        elif (f, ans) in self.red_flags:
            self.pending_flag = (f, ans, self.red_flags[(f, ans)])

    # 2) DECISIÓN -----------------------------------------------------------------
    def decide(self):
        """Elige la siguiente acción a partir de la creencia actual (utilidad esperada)."""
        post = self._posterior(self.evidence)
        level, exp_cost = self._best_prediction(post)
        info = {"belief": {L: round(p, 3) for L, p in post.items()},
                "expected_cost": round(exp_cost, 3), "queue_length": self.queue_len}

        if self.red_flag:                                     # riesgo vital: escalar ya
            f, ans, lvl = self.red_flag
            act = Action("ESCALATE", level=lvl, info=info,
                         reason=f"bandera roja: {f}={ans}")
        elif self.patient_left:
            act = Action("CLASSIFY", level=level, info={**info, "incomplete": True},
                         reason="el paciente abandonó la conversación")
        elif self.pending_flag:                               # confirmar antes de escalar
            act = Action("ASK", feature=self.pending_flag[0], info=info,
                         reason=f"confirmar bandera roja: {self.pending_flag[0]}="
                                f"{self.pending_flag[1]}")
        else:
            remaining = [n for n, f in self.features.items() if f.askable and n not in self.asked]
            tamizaje = [f for f in self.screening if f in remaining]
            if tamizaje:                                      # tamizaje obligatorio de riesgo vital
                act = Action("ASK", feature=tamizaje[0], info=info,
                             reason="tamizaje obligatorio de riesgo vital")
            elif not remaining or self.n_questions >= self.max_questions:
                act = Action("CLASSIFY", level=level, info=info,
                             reason="no quedan preguntas útiles o se alcanzó el máximo")
            elif self.policy == "fixed" or self.explore:      # cuestionario rígido / exploración
                act = Action("ASK", feature=remaining[0], info=info,
                             reason="cuestionario fijo" if self.policy == "fixed"
                             else "exploración (entrenamiento)")
            else:
                # el costo de preguntar sube con la congestión de la cola
                cost_q = self.q_cost * (1 + min(self.queue_len, 30) / 10)
                gains = {f: self._expected_gain(f, post) for f in remaining}
                best = max(gains, key=gains.get)
                info.update(gains={f: round(g, 3) for f, g in gains.items()},
                            question_cost=round(cost_q, 3))
                if gains[best] > cost_q:
                    act = Action("ASK", feature=best, info=info,
                                 reason=f"ganancia {gains[best]:.3f} > costo {cost_q:.3f}")
                else:
                    act = Action("CLASSIFY", level=level, info=info,
                                 reason=f"ganancia {gains[best]:.3f} <= costo {cost_q:.3f}")
        self.log.log(self.episode, "decision", kind=act.kind, feature=act.feature,
                     level=act.level, reason=act.reason, **act.info)
        return act

    # 3) ACCIÓN -------------------------------------------------------------------
    def act(self, action, env):
        """Ejecuta la acción. ASK devuelve la siguiente percepción; el resto cierra el caso."""
        if action.kind == "ASK":
            self.asked.add(action.feature)
            self.n_questions += 1
            self.log.log(self.episode, "action", tool="ask_patient",
                         question=self.features[action.feature].question)
            return env.step(action.feature)
        summary = self._summary(action)
        if action.kind == "ESCALATE":
            r = env.tools.alert_staff(self.patient["id"], action.level, action.reason)
            self.log.log(self.episode, "action", tool="alert_staff", result=r)
        r1 = env.tools.assign_slot(self.patient["id"], action.level)
        r2 = env.tools.send_summary(summary)
        self.log.log(self.episode, "action", tool="assign_slot", result=r1)
        self.log.log(self.episode, "action", tool="send_summary", result=r2,
                     summary=summary)
        return None

    def _summary(self, action):
        post = self._posterior(self.evidence)
        return {"patient_id": self.patient["id"], "age": self.patient["age"],
                "chief_complaint": self.patient["chief_complaint"],
                "answers": {fname: self.features[fname].answers[a]
                            for fname, a in self.evidence},
                "suggested_level": action.level,
                "level_name": LEVEL_NAMES[action.level],
                "confidence": round(post[action.level], 3),
                "justification": action.reason,
                "questions_asked": self.n_questions,
                "incomplete": bool(action.info.get("incomplete")),
                "requires_human_validation": True}      # supervisión humana obligatoria

    # ------------------------------------------------------------------ aprendizaje --
    def learn(self, clinician_level):
        """Actualiza las tablas con la corrección del profesional (retroalimentación)."""
        if not self.learning:
            return
        self.level_counts[clinician_level] += 1
        for f, a in self.evidence:
            self.counts[clinician_level][f][a] += 1

    # ------------------------------------------------------------------ bucle ------
    def run_episode(self, env, episode_id):
        """Atiende un paciente completo: percepción -> decisión -> acción, repetido."""
        self.start_episode(episode_id)
        percept = env.reset()
        while True:
            self.perceive(percept)
            action = self.decide()
            percept = self.act(action, env)
            if percept is None:
                break
        label = env.clinician_label()
        self.learn(label)
        self.log.log(episode_id, "feedback", clinician_level=label,
                     predicted=action.level, learned=self.learning)
        res = {"episode": episode_id, "predicted": action.level, "clinician": label, "true_level": env.true_level,
               "questions": self.n_questions, "escalated": action.kind == "ESCALATE",
               "incomplete": bool(action.info.get("incomplete")),
               "used_ehr": self.ehr_used}
        self.log.log(episode_id, "episode_end", **{k: v for k, v in res.items() if k != "episode"})
        return res
