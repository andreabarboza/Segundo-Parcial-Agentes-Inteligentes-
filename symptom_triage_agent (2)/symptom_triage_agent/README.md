# Symptom Triage Agent — Fase 2

Agente de pre-triage conversacional **basado en utilidad con aprendizaje**
(Russell & Norvig), probado en un entorno de urgencias **simulado**.
Curso de Inteligencia Artificial — Ingeniería Biomédica, Universidad Tecnológica de Bolívar.

El agente mantiene una creencia sobre los 5 niveles de urgencia (1 inmediato … 5 no urgente),
elige en cada paso la pregunta que más reduce el costo esperado de equivocarse, y deja de
preguntar cuando la ganancia deja de compensar el costo. **Siempre** entrega su resultado como
sugerencia que requiere validación humana (`requires_human_validation: true`).

## Ejecutar en otro equipo

Requiere Python 3.9 o superior. El agente y el simulador no usan dependencias externas; el modo con datos reales usa `pandas` (ya instalado en Colab).

```bash
git clone <URL-DEL-REPOSITORIO>
cd symptom_triage_agent
python run_simulation.py                    # agente de utilidad: 500 casos de entrenamiento + 300 de prueba
python run_simulation.py --policy fixed     # línea base: cuestionario rígido
python run_simulation.py --no-learning      # sin aprendizaje
python run_simulation.py --seed 7           # otra semilla (misma semilla = mismos resultados)
pip install -r requirements.txt && python -m pytest -q    # 13 pruebas
python ceiling.py                           # techo de exactitud del simulador
python run_ktas.py --data data.csv          # con el dataset REAL KTAS (ver sección "Modo con datos reales")
python run_fase3.py                         # Fase 3: pruebas con umbrales fijados de antemano (≈ 15 s)
```

Cada ejecución escribe en `runs/`: un **log** `log_<config>.jsonl` (una línea por evento, con
marca de tiempo UTC) y `results_<config>.csv` (una fila por paciente). En
`docs/ejemplo/` hay un log de ejemplo de 3 pacientes.

## Estructura

```
run_simulation.py        entrenamiento + prueba, métricas, escritura de log y CSV
ceiling.py               techo de exactitud del simulador
triage/features.py       síntomas, niveles, banderas rojas, modelos de verosimilitud
triage/environment.py    simulador: paciente, cola de atención, herramientas (tool calling)
triage/agent.py          el agente (percepción -> decisión -> acción, utilidad, aprendizaje)
triage/logger.py         log JSON Lines con marca de tiempo
triage/ktas.py           modo con datos reales: cargador del dataset KTAS + entorno que reproduce casos
run_ktas.py              evalúa el agente con casos reales (particiones entrenamiento/prueba)
tests/test_agent.py      pruebas del agente y el simulador
tests/test_ktas.py       pruebas del cargador con un CSV mínimo del formato KTAS
tests/test_robustez.py   pruebas de robustez (entradas inválidas, sensores sin dato)
run_fase3.py             Fase 3: ejecuta 6 escenarios x 10 repeticiones y arma la tabla de resultados
fase3/umbrales.json      umbrales y escenarios FIJADOS antes de ejecutar (hacer commit antes de correr)
fase3/diagnostico.py     causas de las medidas incumplidas y "qué cambiaría en el agente"
fase3_resultados/        tabla, conteos y logs de la corrida oficial
fase3_resultados_corrida1_antes_de_corregir/   primera corrida, con las fallas reales encontradas
docs/DISENO_VS_IMPLEMENTACION.md   cambios frente al diseño y su justificación
```

## Dónde está el ciclo percepción → decisión → acción

Todo en `triage/agent.py`, clase `SymptomTriageAgent`:

| Etapa | Método | Qué hace |
|---|---|---|
| Percepción | `perceive()` | Recibe respuesta del paciente (chat), historia clínica (con consentimiento) y estado de la cola; actualiza memoria y banderas rojas |
| Decisión | `decide()` | Calcula la creencia, compara preguntas por ganancia esperada vs. costo, y elige `ASK`, `CLASSIFY` o `ESCALATE` |
| Acción | `act()` | Hace la pregunta, o invoca las herramientas: `assign_slot`, `send_summary`, `alert_staff` |
| Aprendizaje | `learn()` | Con la corrección del clínico actualiza `P(respuesta\|nivel)` y la distribución de niveles |
| Bucle | `run_episode()` | Repite el ciclo hasta clasificar un paciente |

## Coherencia con el tipo de agente

- **Utilidad:** el costo de error es asimétrico (`W_UNDER=3` por nivel de sub-triage, `W_OVER=1` por
  sobre-triage) y cada pregunta tiene costo, que sube con la congestión de la cola.
  Pregunta solo si `ganancia esperada > costo`.
- **Aprendizaje:** las tablas arrancan con conocimiento vago y mejoran con la retroalimentación.
- **Memoria:** corto plazo = evidencia y preguntas de la conversación (no repite preguntas);
  largo plazo = historia clínica y tablas aprendidas.

## Sobre el uso de LLM y herramientas externas

**No se usa ningún LLM ni librería externa.** Todo el razonamiento (creencia, utilidad, selección de
preguntas, aprendizaje, banderas rojas) lo hace el código de `triage/agent.py`. El simulador
reemplaza al entorno real. Ver `docs/DISENO_VS_IMPLEMENTACION.md`.

## Resultados de referencia (prueba congelada, promedio de 5 semillas)

| Configuración | Coincidencia exacta | ±1 nivel | Sub-triage | Sobre-triage | Preguntas |
|---|---|---|---|---|---|
| Utilidad + aprendizaje | 60,3 % ± 2,2 | 96,4 % | 8,5 % | 31,1 % | 1,95 |
| Utilidad sin aprendizaje | 39,0 % ± 2,1 | 95,3 % | 44,9 % | 16,1 % | 2,65 |
| Cuestionario fijo + aprendizaje | 58,3 % ± 2,9 | 95,8 % | 12,0 % | 29,7 % | 6,49 |

**Cómo leerlos:** son propiedades de un *simulador*, no evidencia clínica. La meta de diseño
(≥ 80 % de coincidencia exacta) **no se alcanza**: aun un agente perfecto que conoce las tablas
reales tendría un techo de ~84 % contra el nivel real y ~75 % contra el clínico simulado
(`python ceiling.py`). Hay dos resultados útiles: (1) el aprendizaje sube la coincidencia unos
20 puntos y baja el sub-triage de 45 % a 8,5 %; (2) el agente de utilidad iguala al cuestionario fijo con
casi un tercio de las preguntas.

## Modo con datos reales (dataset KTAS de Kaggle)

Dataset: <https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application>
(1.267 registros de dos urgencias de Corea; estudio de origen: Moon et al., 2019, PLOS ONE,
doi:10.1371/journal.pone.0216972). Descarga `data.csv` (Kaggle pide iniciar sesión) y ejecuta:

```bash
python run_ktas.py --data data.csv                      # 5 particiones aleatorias 70/30
python run_ktas.py --data data.csv --seeds 10 --train-frac 0.8
```

Muestra el acuerdo enfermera-vs-experto del propio dataset (la **referencia humana**), los rangos de
cada columna (para verificar unidades) y una tabla comparando: agente de utilidad con y sin
aprendizaje, cuestionario fijo, la enfermera (`KTAS_RN`) y la clase mayoritaria.

**Qué usa y qué no usa el agente**

| Rol | Columnas |
|---|---|
| Síntomas que "pregunta" | `Mental`, `Saturation`, `RR`, `HR`, `SBP`, `BT`, `NRS_pain`/`Pain` |
| Datos de admisión (ya conocidos) | `Age`, `Arrival mode`, `Injury` |
| Objetivo | `KTAS_expert` (la corrección del experto) |
| Solo para comparar | `KTAS_RN` (enfermera) |
| **Nunca se usan** (ocurren después del triage o se derivan de la etiqueta) | `Diagnosis in ED`, `Disposition`, `Length of stay_min`, `KTAS duration_min`, `Error_group`, `mistriage` |
| Solo análisis | `Group`, `Patients number per hour` |

**Limitaciones que debes declarar**
- `Chief_complain` (el texto del motivo de consulta) **no se usa en la decisión**; en KTAS el motivo de
  consulta pesa mucho, así que es probable que el agente quede por debajo de la enfermera.
  Incorporarlo requiere procesamiento de lenguaje natural (trabajo futuro).
- Los umbrales (p. ej. SpO2 < 90 % = "muy baja") son convenciones clínicas comunes que yo asumí, no
  reglas oficiales de KTAS; el dataset no documenta las unidades de cada columna.
- Hasta 55 % de `Saturation` y 44 % de `NRS_pain` están sin dato. Se tratan como "no sé", no como 0.
- No hay identificador de paciente: no se puede garantizar que una persona no quede en entrenamiento y
  prueba a la vez, y las 1.267 filas pueden no ser pacientes distintos.
- Con datos reales no se aplican ruido, abandono ni deterioro simulados; solo la cola es simulada.
- KTAS no es Manchester ni ESI (es una variante de CTAS con cinco niveles equivalentes).
- El cargador se probó con un CSV mínimo que replica el formato descrito; **los resultados con el
  archivo real los debes generar tú**.
- Durante el entrenamiento el agente pregunta todo (exploración) para poder aprender; en la prueba usa su
  política de utilidad. Ver `docs/DISENO_VS_IMPLEMENTACION.md`.

## Fase 3 — Pruebas basadas en las medidas de desempeño

```bash
python run_fase3.py                    # resultados oficiales
python run_fase3.py --keep-logs all    # conserva los logs de las 10 repeticiones
python fase3/diagnostico.py            # por qué fallaron M1 y M4 y qué cambiaría en el agente
```

Informe completo (tabla de la rúbrica, escenarios y análisis): `docs/FASE3_INFORME.md`.

- 8 medidas (M1 es la del diseño; M2-M8 derivan de sus riesgos y requisitos), 6 escenarios (normal, límite y
  3 de falla esperada), 10 repeticiones x 100 pacientes. Cada medida declara fórmula, umbral y
  escenario de aceptación en `fase3/umbrales.json`; el arnés imprime su SHA-256.
- **Resultado: 6 de 8 medidas cumplidas.** No se cumplen M1 (coincidencia exacta, 60,5 % vs ≥ 80 %) y
  M4 (escalamiento ante riesgo vital, 87,1 % vs ≥ 90 %). Las causas están analizadas en el informe.
- Para que el "umbral fijado antes" sea demostrable, haz **commit de `fase3/umbrales.json` antes de ejecutar**
  y cita ese commit en el informe.
