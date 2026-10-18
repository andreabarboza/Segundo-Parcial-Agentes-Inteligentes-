"""Modo con datos reales: dataset KTAS de Kaggle (Emergency Service - Triage Application).

Fuente: https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application
Origen del estudio: Moon et al. (2019), PLOS ONE, doi:10.1371/journal.pone.0216972

Qué usa el agente (solo lo disponible al momento del triage):
    Mental, Saturation, RR, HR, SBP, BT, NRS_pain/Pain  -> síntomas que el agente "pregunta"
    Age, Arrival mode, Injury                          -> datos de admisión (se conocen al llegar)
Objetivo / referencia:  KTAS_expert (objetivo), KTAS_RN (enfermera, solo para comparar)
NO se usa nunca: Diagnosis in ED, Disposition, Length of stay_min, KTAS duration_min,
                 Error_group, mistriage (ocurren después del triage o se derivan de la etiqueta).
Group y Patients number per hour no entran al modelo (solo sirven para análisis).
"""
import hashlib
import random

import pandas as pd

from .environment import QueueSystem, ToolKit
from .features import Feature

EXPECTED_COLUMNS = [
    "Group", "Sex", "Age", "Patients number per hour", "Arrival mode", "Injury",
    "Chief_complain", "Mental", "Pain", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT",
    "Saturation", "KTAS_RN", "Diagnosis in ED", "Disposition", "KTAS_expert",
    "Error_group", "Length of stay_min", "KTAS duration_min", "mistriage"]
MISSING_TOKENS = {"", "??", "#BOŞ!"}
NUMERIC_COLUMNS = ["Age", "Arrival mode", "Injury", "Mental", "Pain", "NRS_pain", "SBP",
                   "DBP", "HR", "RR", "BT", "Saturation", "KTAS_RN", "KTAS_expert", "Group"]

KTAS_FEATURES = {f.name: f for f in [
    # Se "preguntan" / miden en el triage
    Feature("consciousness", "¿Cuál es su nivel de conciencia?",
            ("alerta", "voz", "dolor", "no_responde"), 1.0, 0.30),
    Feature("spo2", "¿Cuál es su saturación de oxígeno?", ("normal", "baja", "muy_baja"), 1.0, 0.30),
    Feature("rr", "¿Cuál es su frecuencia respiratoria?", ("normal", "elevada", "grave"), 1.0, 0.30),
    Feature("hr", "¿Cuál es su frecuencia cardíaca?", ("normal", "alterada", "muy_alterada"), 1.0, 0.30),
    Feature("sbp", "¿Cuál es su presión sistólica?", ("normal", "alterada", "grave"), 1.0, 0.30),
    Feature("bt", "¿Cuál es su temperatura?", ("normal", "alterada", "extrema"), 1.0, 0.30),
    Feature("pain", "Del 0 al 10, ¿qué tan fuerte es el dolor?", ("leve", "moderado", "intenso"), 1.0, 0.30),
    # Datos de admisión: se conocen al llegar, no se preguntan
    Feature("age_group", "Edad", ("menor_40", "40_64", "65_mas"), 1.0, 0.40, askable=False),
    Feature("arrival", "Forma de llegada", ("caminando", "vehiculo_privado", "ambulancia"),
            1.0, 0.40, askable=False),
    Feature("injury", "¿Es una lesión (trauma)?", ("no", "si"), 1.0, 0.40, askable=False),
]}
KTAS_RED_FLAGS = {("consciousness", "no_responde"): 1}
ADMISSION_FEATURES = ("age_group", "arrival", "injury")


# ---------------------------------------------------------------- discretización --
def _in(x, lo, hi):
    return (not pd.isna(x)) and lo <= x <= hi     # pd.isna cubre NaN y pd.NA


def d_consciousness(m):
    return None if pd.isna(m) else {1: "alerta", 2: "voz", 3: "dolor", 4: "no_responde"}.get(int(m))


def d_spo2(x):
    if not _in(x, 30, 100):
        return None
    return "muy_baja" if x < 90 else "baja" if x < 95 else "normal"


def d_rr(x):
    if not _in(x, 1, 80):
        return None
    return "grave" if (x >= 30 or x <= 8) else "elevada" if (x >= 21 or x <= 11) else "normal"


def d_hr(x):
    if not _in(x, 10, 300):
        return None
    return "muy_alterada" if (x >= 130 or x <= 40) else "alterada" if (x >= 101 or x <= 50) else "normal"


def d_sbp(x):
    if not _in(x, 30, 300):
        return None
    return "grave" if x < 90 else "alterada" if (x < 100 or x >= 180) else "normal"


def d_bt(x):
    if not _in(x, 28, 45):
        return None
    return "extrema" if (x >= 39 or x < 35) else "alterada" if (x >= 37.5 or x < 36) else "normal"


def d_pain(pain_flag, nrs):
    if _in(nrs, 0, 10):
        return "leve" if nrs <= 3 else "moderado" if nrs <= 6 else "intenso"
    return "leve" if (not pd.isna(pain_flag) and int(pain_flag) == 0) else None  # sin dolor


def d_age(a):
    return None if pd.isna(a) else "menor_40" if a < 40 else "40_64" if a < 65 else "65_mas"


def d_arrival(a):
    if pd.isna(a):
        return None
    return {1: "caminando", 3: "vehiculo_privado", 2: "ambulancia", 4: "ambulancia"}.get(int(a))


def d_injury(i):
    return None if pd.isna(i) else {1: "no", 2: "si"}.get(int(i))


# ---------------------------------------------------------------- carga ---------
def _numeric(raw, col):
    s = raw[col].astype("string").str.strip()
    s = s.where(~s.isin(MISSING_TOKENS))
    out = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
    bad = int((s.notna() & out.isna()).sum())
    # float64 plano con NaN (evita pd.NA, que rompe comparaciones simples)
    return pd.Series(out.to_numpy(dtype="float64", na_value=float("nan")), index=s.index), bad


def load_ktas(path):
    """Lee data.csv y devuelve (casos, info). Falla con un mensaje claro si el formato cambió."""
    raw = pd.read_csv(path, sep=";", encoding="cp1254", dtype=str, keep_default_na=False)
    faltan = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas {faltan}. Columnas leídas: {list(raw.columns)}. "
                         "Revisa separador (;) y codificación (cp1254), o ajusta EXPECTED_COLUMNS.")
    num, bad = {}, {}
    for c in NUMERIC_COLUMNS:
        num[c], bad[c] = _numeric(raw, c)
    cases = pd.DataFrame({
        "consciousness": [d_consciousness(v) for v in num["Mental"]],
        "spo2": [d_spo2(v) for v in num["Saturation"]],
        "rr": [d_rr(v) for v in num["RR"]],
        "hr": [d_hr(v) for v in num["HR"]],
        "sbp": [d_sbp(v) for v in num["SBP"]],
        "bt": [d_bt(v) for v in num["BT"]],
        "pain": [d_pain(p, n) for p, n in zip(num["Pain"], num["NRS_pain"])],
        "age_group": [d_age(v) for v in num["Age"]],
        "arrival": [d_arrival(v) for v in num["Arrival mode"]],
        "injury": [d_injury(v) for v in num["Injury"]],
        "age": num["Age"], "group": num["Group"],
        "chief_complaint": raw["Chief_complain"].str.strip(),
        "ktas_expert": num["KTAS_expert"], "ktas_rn": num["KTAS_RN"]}, dtype=object)
    for c in ("ktas_expert", "ktas_rn"):
        cases[c] = pd.to_numeric(cases[c], errors="coerce")
    validos = cases.ktas_expert.isin([1, 2, 3, 4, 5])
    cases = cases[validos].reset_index(drop=True)
    cases["ktas_expert"] = cases.ktas_expert.astype(int)
    cases["ktas_rn"] = cases.ktas_rn.astype("Int64")
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    info = {"filas_leidas": len(raw), "filas_validas": len(cases), "sha256": sha,
            "no_numericos_inesperados": {k: v for k, v in bad.items() if v},
            "faltantes_por_sintoma": {n: float(cases[n].isna().mean()) for n in KTAS_FEATURES},
            "acuerdo_enfermera_vs_experto": float((cases.ktas_rn == cases.ktas_expert).mean())}
    return cases, info


def describe_ranges(path):
    """Mínimo / mediana / máximo de las columnas numéricas, para comprobar las unidades."""
    raw = pd.read_csv(path, sep=";", encoding="cp1254", dtype=str, keep_default_na=False)
    filas = []
    for c in ["Age", "SBP", "DBP", "HR", "RR", "BT", "Saturation", "NRS_pain"]:
        v, _ = _numeric(raw, c)
        filas.append({"columna": c, "n_validos": int(v.notna().sum()), "min": v.min(),
                      "mediana": v.median(), "max": v.max()})
    return pd.DataFrame(filas).set_index("columna")


# ---------------------------------------------------------------- entorno -------
class KTASReplayEnvironment:
    """Reproduce casos reales del dataset con la misma interfaz que PatientSimulator.

    Con datos reales NO se aplican ruido, abandono ni deterioro simulados: la
    incertidumbre viene de los propios datos (valores faltantes). Solo la cola de
    atención sigue siendo simulada (el dataset no trae información de cola).
    """

    def __init__(self, cases, seed=0):
        self.cases = cases.reset_index(drop=True)
        self.rng = random.Random(seed)
        self.queue = QueueSystem(self.rng)
        self.tools = ToolKit(self.queue)
        self._order, self._pos, self.row = [], 0, None

    def set_cases(self, indices):
        self._order, self._pos = list(indices), 0

    @property
    def true_level(self):
        return int(self.row["ktas_expert"])

    @property
    def nurse_level(self):
        v = self.row["ktas_rn"]
        return None if pd.isna(v) else int(v)

    def reset(self):
        idx = self._order[self._pos]
        self._pos += 1
        self.row = self.cases.iloc[idx]
        self.patient_id = f"K{idx:04d}"
        self.queue.tick()
        known = {n: self.row[n] for n in ADMISSION_FEATURES if not pd.isna(self.row[n])}
        age = self.row["age"]
        return {"type": "initial", "patient_id": self.patient_id,
                "age": None if pd.isna(age) else int(age),
                "chief_complaint": self.row["chief_complaint"],
                "ehr": known or None, "queue": self.queue.state()}

    def step(self, feature):
        self.queue.tick()
        v = self.row[feature]
        return {"type": "answer", "feature": feature, "answer": None if pd.isna(v) else v,
                "queue": self.queue.state()}

    def clinician_label(self):
        return self.true_level           # nivel que asignó el experto (KTAS_expert)
