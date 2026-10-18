"""Entorno simulado del servicio de urgencias.

Propiedades (sección 3 del diseño):
- Parcialmente observable: el nivel real y los valores reales están ocultos; el agente
  solo ve lo que el paciente reporta.
- Estocástico: los síntomas se muestrean, y el paciente se equivoca o no sabe responder.
- Secuencial: cada pregunta cambia lo que el agente sabe.
- Dinámico: el paciente puede empeorar o abandonar durante la conversación, y la cola
  de atención cambia sola.
- Agente único.

Sensores simulados (sección 5): respuestas del paciente (chat), historia clínica
(solo con consentimiento) y estado de la cola (API de agendamiento).
"""
import math
import random

from .features import (FEATURES, LEVELS, RED_FLAGS, SIM_PRIOR, likelihood_table)

COMPLAINTS = {
    1: ["no respira bien y se siente muy mal", "dolor de pecho opresivo e intenso"],
    2: ["dolor fuerte en el pecho", "golpe fuerte en la cabeza con mareo"],
    3: ["dolor abdominal intenso", "fiebre alta desde ayer"],
    4: ["dolor de garganta y tos", "dolor de espalda leve"],
    5: ["control de rutina", "resfriado leve"],
}


def _poisson(rng, lam):
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


class QueueSystem:
    """Cola de atención con prioridad (menor nivel = antes)."""

    def __init__(self, rng, arrivals=0.15, service=0.5):
        self.rng, self.arrivals, self.service = rng, arrivals, service
        self._seq = 0
        self.waiting = []
        for _ in range(rng.randint(4, 12)):
            self._add(rng.choices(LEVELS, SIM_PRIOR)[0], f"ext{self._seq}")

    def _add(self, level, pid):
        self.waiting.append((level, self._seq, pid))
        self._seq += 1
        self.waiting.sort()

    def tick(self):
        """Pasa el tiempo: llegan pacientes externos y se atiende al de mayor prioridad."""
        for _ in range(_poisson(self.rng, self.arrivals)):
            self._add(self.rng.choices(LEVELS, SIM_PRIOR)[0], f"ext{self._seq}")
        if self.waiting and self.rng.random() < self.service:
            self.waiting.pop(0)

    def state(self):
        return {"length": len(self.waiting),
                "urgent_waiting": sum(1 for w in self.waiting if w[0] <= 2)}

    def assign(self, pid, level):
        self._add(level, pid)
        pos = [w[2] for w in self.waiting].index(pid) + 1
        return {"position": pos, "queue_length": len(self.waiting)}


class ToolKit:
    """Herramientas que el agente invoca (tool calling) sobre el sistema institucional."""

    def __init__(self, queue):
        self.queue, self.outbox, self.alerts = queue, [], []

    def assign_slot(self, patient_id, level):
        return self.queue.assign(patient_id, level)

    def send_summary(self, summary):
        self.outbox.append(summary)
        return {"delivered": True}

    def alert_staff(self, patient_id, level, reason):
        self.alerts.append((patient_id, level, reason))
        return {"alerted": True}


# Respuestas fuera de dominio, para probar robustez ante entradas inválidas (Fase 3)
INVALID_ANSWERS = ["???", "", 42, "asdf", True]


class PatientSimulator:
    def __init__(self, seed=0, noise=0.10, p_unknown=0.08, p_deteriorate=0.03,
                 p_abandon=0.02, p_consent=0.7, clinician_agreement=0.88,
                 level_prior=None, p_invalid=0.0, p_queue_down=0.0):
        # level_prior: distribución de niveles de los pacientes (por defecto SIM_PRIOR)
        # p_invalid: prob. de que una respuesta llegue fuera de dominio (entrada inválida)
        # p_queue_down: prob. de que la API de la cola no responda (sensor sin dato)
        # Los parámetros nuevos solo consumen números aleatorios si valen > 0, así que con
        # sus valores por defecto el simulador se comporta exactamente como antes.
        self.level_prior = tuple(level_prior) if level_prior else SIM_PRIOR
        self.p_invalid, self.p_queue_down = p_invalid, p_queue_down
        self.rng = random.Random(seed)
        self.noise, self.p_unknown = noise, p_unknown
        self.p_deteriorate, self.p_abandon = p_deteriorate, p_abandon
        self.p_consent, self.clinician_agreement = p_consent, clinician_agreement
        self.queue = QueueSystem(self.rng)
        self.tools = ToolKit(self.queue)
        self._n = 0
        # Estado oculto (solo para evaluación; el agente nunca lo lee)
        self.true_level = None
        self.deteriorated = False

    def _draw_features(self):
        self.true_values = {
            name: self.rng.choices(range(len(f.answers)),
                                   likelihood_table(f, self.true_level))[0]
            for name, f in FEATURES.items()}

    # ---- interfaz con el agente -------------------------------------------------
    def reset(self):
        """Llega un paciente nuevo. Devuelve la percepción inicial."""
        self.patient_id = f"P{self._n:05d}"
        self._n += 1
        self.true_level = self.rng.choices(LEVELS, self.level_prior)[0]
        self.deteriorated = False
        self._draw_features()
        # ¿llega con un signo de riesgo vital real? (solo para evaluar, el agente no lo ve)
        self.true_red_flag = any(FEATURES[f].answers[self.true_values[f]] == a
                                 for (f, a) in RED_FLAGS)
        self.queue.tick()
        ehr = None
        if self.rng.random() < self.p_consent:
            f = FEATURES["chronic"]
            ehr = {"chronic": f.answers[self.true_values["chronic"]]}
        return {"type": "initial", "patient_id": self.patient_id,
                "age": int(min(95, max(18, self.rng.gauss(50, 18)))),
                "chief_complaint": self.rng.choice(COMPLAINTS[self.true_level]),
                "ehr": ehr, "queue": self._queue_state()}

    def step(self, feature):
        """El agente hace una pregunta; devuelve la respuesta (con ruido) del paciente."""
        self.queue.tick()
        if self.rng.random() < self.p_abandon:
            return {"type": "patient_left", "queue": self._queue_state()}
        if self.true_level > 1 and self.rng.random() < self.p_deteriorate:
            self.true_level -= 1          # el paciente empeora durante la conversación
            self.deteriorated = True
            self._draw_features()
        f = FEATURES[feature]
        t, r = self.true_values[feature], self.rng.random()
        if r < self.p_unknown:
            ans = None                     # "no sé"
        elif r < self.p_unknown + self.noise:
            ans = f.answers[self.rng.choice([j for j in range(len(f.answers)) if j != t])]
        else:
            ans = f.answers[t]
        if self.p_invalid and self.rng.random() < self.p_invalid:
            ans = self.rng.choice(INVALID_ANSWERS)      # entrada inválida
        return {"type": "answer", "feature": feature, "answer": ans,
                "queue": self._queue_state()}

    def _queue_state(self):
        """Estado de la cola; None si la API de agendamiento no responde."""
        if self.p_queue_down and self.rng.random() < self.p_queue_down:
            return None
        return self.queue.state()

    def clinician_label(self):
        """Nivel que confirma/corrige el profesional (con desacuerdo entre evaluadores)."""
        if self.rng.random() < self.clinician_agreement:
            return self.true_level
        return min(5, max(1, self.true_level + self.rng.choice([-1, 1])))
