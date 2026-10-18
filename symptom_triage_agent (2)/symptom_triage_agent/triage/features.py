"""Síntomas (percepciones), niveles de urgencia y modelos de verosimilitud.

Los niveles siguen la escala de cinco niveles del diseño (Manchester/ESI/KTAS):
1 inmediato, 2 muy urgente, 3 urgente, 4 estándar, 5 no urgente.

Cada síntoma es una variable ordinal discreta: la respuesta 0 es la más benigna
y la última la más grave. P(respuesta | nivel) se modela con una gaussiana
discretizada centrada en s**gamma, donde s = (5 - nivel) / 4 es la gravedad.
"""
import math
from dataclasses import dataclass

LEVELS = (1, 2, 3, 4, 5)
LEVEL_NAMES = {1: "Inmediato", 2: "Muy urgente", 3: "Urgente",
               4: "Estándar", 5: "No urgente"}

# Distribución de niveles: la del mundo (simulador) y la que el agente asume al inicio.
SIM_PRIOR = (0.03, 0.12, 0.38, 0.34, 0.13)
PROTOCOL_PRIOR = (0.05, 0.15, 0.35, 0.30, 0.15)


@dataclass(frozen=True)
class Feature:
    name: str
    question: str
    answers: tuple
    gamma: float          # >1: el síntoma solo aparece en niveles muy graves
    sigma: float          # dispersión: menor = más informativo
    askable: bool = True  # False = llega por la historia clínica, no se pregunta


FEATURES = {f.name: f for f in [
    Feature("breathing", "¿Tiene dificultad para respirar?",
            ("normal", "dificultad_leve", "dificultad_severa"), 2.5, 0.12),
    Feature("consciousness", "¿Está alerta y orientado/a?",
            ("alerta", "confuso", "no_responde"), 4.0, 0.12),
    Feature("bleeding", "¿Presenta sangrado?",
            ("ninguno", "leve", "abundante"), 2.5, 0.12),
    Feature("pain", "Del 0 al 10, ¿qué tan fuerte es el dolor?",
            ("leve", "moderado", "intenso"), 1.0, 0.14),
    Feature("fever", "¿Ha tenido fiebre?",
            ("no", "febril", "fiebre_alta"), 1.5, 0.14),
    Feature("onset", "¿Empezó de forma gradual o repentina?",
            ("gradual", "subito"), 1.2, 0.16),
    Feature("duration", "¿Hace cuánto empezó?",
            ("mas_24h", "1_24h", "menos_1h"), 1.0, 0.20),
    Feature("chronic", "Antecedentes crónicos (historia clínica)",
            ("ninguno", "uno", "varios"), 1.0, 0.24, askable=False),
]}

# Banderas rojas: (síntoma, respuesta) -> nivel con el que se escala de inmediato.
RED_FLAGS = {
    ("breathing", "dificultad_severa"): 1,
    ("consciousness", "no_responde"): 1,
    ("bleeding", "abundante"): 2,
}


def likelihood_table(feature, level, gamma=None, sigma=None):
    """P(respuesta | nivel) como lista normalizada."""
    gamma = feature.gamma if gamma is None else gamma
    sigma = feature.sigma if sigma is None else sigma
    s = (5 - level) / 4
    center = s ** gamma
    k = len(feature.answers)
    w = [math.exp(-((j / (k - 1) - center) ** 2) / (2 * sigma ** 2)) for j in range(k)]
    z = sum(w)
    return [x / z for x in w]


def protocol_likelihood(feature, level):
    """Conocimiento inicial del agente: vago (gamma=1, sigma doble). Debe mejorar aprendiendo."""
    return likelihood_table(feature, level, gamma=1.0, sigma=feature.sigma * 2)
