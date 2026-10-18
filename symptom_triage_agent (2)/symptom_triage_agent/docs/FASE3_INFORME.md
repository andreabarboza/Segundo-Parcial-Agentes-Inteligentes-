# Fase 3 — Pruebas basadas en las medidas de desempeño

**Agente:** Symptom Triage Agent (basado en utilidad con aprendizaje). **Entorno:** simulador de urgencias.
**Umbrales fijados en** `fase3/umbrales.json` (SHA-256 `0675639c241c1bc4c41a0e4d5b66dda36f0d7ac183939771d28897ad8f23dd09`) **antes de ejecutar**; no se modificaron después de ver resultados.
**Diseño de la prueba:** 10 repeticiones por escenario (semillas distintas), 100 pacientes por escenario y repetición, agente entrenado con 500 pacientes y luego congelado. Resultado = media ± desviación estándar entre repeticiones (pp = puntos porcentuales).

## 1. Tabla de resultados

M1 es la medida SMART de la sección 7 del diseño; M2–M8 se derivan de los riesgos y requisitos del propio diseño (columna `origen` en `umbrales.json`). Completa la columna *Evidencia* con la captura y el minuto del video.

| ID | Medida de desempeño | Fórmula | Umbral | Casos ejecutados | Resultado obtenido | ¿Cumple? | Evidencia (log · captura/video) |
|---|---|---|---|---|---|---|---|
| M1 | Coincidencia exacta con el profesional | casos con nivel del agente = nivel del profesional / casos totales | >= 80% | 1000 pacientes | 60.5% ± 4.5 pp | No | logs/E1_rep01.jsonl · resultados_por_corrida.csv (escenario E1) · captura/video: min ___ |
| M2 | Tasa de sub-triage | casos con nivel del agente MENOS urgente que el del profesional / casos totales | <= 10% | 1000 pacientes | 7.6% ± 3.4 pp | Sí | logs/E1_rep01.jsonl · resultados_por_corrida.csv (escenario E1) · captura/video: min ___ |
| M3 | Sensibilidad en pacientes graves | pacientes que el profesional marcó nivel 1-2 y el agente también marcó 1-2 / pacientes nivel 1-2 | >= 90% | 964 pacientes graves (de 1000) | 93.9% ± 4.5 pp | Sí | logs/E3_rep01.jsonl · resultados_por_corrida.csv (escenario E3) · captura/video: min ___ |
| M4 | Escalamiento ante riesgo vital | pacientes con signo de riesgo vital real que el agente escaló al personal / pacientes con signo de riesgo vital real | >= 90% | 517 pacientes con riesgo vital real (de 1000) | 87.1% ± 12.2 pp | No | logs/E3_rep01.jsonl · resultados_por_corrida.csv (escenario E3) · captura/video: min ___ |
| M5 | Eficiencia del interrogatorio | preguntas formuladas en total / pacientes atendidos | <= 4.0 | 1000 pacientes | 1.96 ± 0.29 | Sí | logs/E1_rep01.jsonl · resultados_por_corrida.csv (escenario E1) · captura/video: min ___ |
| M6 | Robustez ante fallas de entradas y sensores | casos que terminan sin excepción y con nivel 1-5 y resumen marcado para validación humana / casos totales | >= 100% | 3000 pacientes | 100.0% ± 0.0 pp | Sí | logs/E4_rep01.jsonl · resultados_por_corrida.csv (escenario E4, E5, E6) · captura/video: min ___ |
| M7 | Trazabilidad del log | episodios cuyo log contiene percepción, decisión, acción, retroalimentación y cierre con marca de tiempo / episodios totales | >= 100% | 6000 pacientes | 100.0% ± 0.0 pp | Sí | logs/E1_rep01.jsonl · resultados_por_corrida.csv (escenario E1, E2, E3, E4, E5, E6) · captura/video: min ___ |
| M8 | Reproducibilidad | repeticiones con la misma semilla que dan resultados idénticos / repeticiones | >= 100% | 10 repeticiones | 100.0% ± 0.0 pp | Sí | logs/E1_rep01.jsonl · resultados_por_corrida.csv (escenario E1) · captura/video: min ___ |

Logs (repetición 1 de cada escenario): `fase3_resultados/logs/`. Conteos de las 60 ejecuciones: `fase3_resultados/resultados_por_corrida.csv`.

## 2. Escenarios de prueba (normales, límite y de falla esperada)

| ID | Tipo | Descripción |
|---|---|---|
| E1 | normal | Pacientes típicos: simulador estándar (10 % de respuestas erróneas, 8 % de 'no sé', 3 % de deterioro por pregunta, 2 % de abandono, 70 % de consentimiento) |
| E2 | limite | Casos de frontera: solo niveles 2, 3 y 4, donde los síntomas se solapan más |
| E3 | normal | Pacientes graves: solo niveles 1 y 2 (para las medidas de seguridad) |
| E4 | falla_esperada | Respuestas poco fiables: 35 % erróneas, 35 % 'no sé', 15 % de abandono |
| E5 | falla_esperada | Sensores sin dato: sin historia clínica, API de la cola caída el 50 % del tiempo, 50 % de 'no sé' |
| E6 | falla_esperada | Entradas inválidas: 15 % de las respuestas llegan fuera de dominio ('???', vacío, número, texto libre, booleano) |

### Resultados por escenario (informativos)

| Medida | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|
| M1 Coincidencia exacta con el profesional | 60.5% ± 4.5 pp | 69.8% ± 4.3 pp | 64.7% ± 11.3 pp | 32.8% ± 4.5 pp | 47.7% ± 5.0 pp | 56.5% ± 6.2 pp |
| M2 Tasa de sub-triage | 7.6% ± 3.4 pp | 11.1% ± 2.5 pp | 30.6% ± 12.8 pp | 15.0% ± 3.6 pp | 12.4% ± 3.2 pp | 10.2% ± 3.5 pp |
| M3 Sensibilidad en pacientes graves | 75.1% ± 9.8 pp | 81.3% ± 6.0 pp | 93.9% ± 4.5 pp | 52.7% ± 15.2 pp | 63.1% ± 12.9 pp | 72.8% ± 9.4 pp |
| M4 Escalamiento ante riesgo vital | 77.4% ± 34.3 pp | — | 87.1% ± 12.2 pp | 29.5% ± 24.4 pp | 61.5% ± 42.0 pp | 87.4% ± 14.4 pp |
| M5 Eficiencia del interrogatorio | 1.96 ± 0.29 | 2.02 ± 0.27 | 2.55 ± 0.30 | 2.83 ± 0.14 | 4.04 ± 0.17 | 2.22 ± 0.32 |
| M6 Robustez ante fallas de entradas y sensores | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp |
| M7 Trazabilidad del log | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp |

## 3. Análisis de resultados

**Cumplidas (6 de 8):** M2 sub-triage 7,6 % (≤ 10 %), M3 sensibilidad en graves 93,9 % (≥ 90 %), M5 1,96 preguntas (≤ 4), M6 robustez, M7 trazabilidad y M8 reproducibilidad, las tres en 100 %.

**M1 no se cumple: 60,5 % ± 4,5 pp (meta ≥ 80 %).** Ningún factor aislado lo explica (`fase3/diagnostico.py`): sin desacuerdo entre clínicos sube a 65,9 %; sin ruido ni deterioro, a 63,0 %; con costo simétrico, a 62,1 % (pero el sub-triage sube de 7,6 % a 17,5 %); las tres a la vez, 68,0 %. `ceiling.py` estima un techo cercano a 75 % contra el clínico simulado, así que **el 80 % no era alcanzable en este simulador**: el criterio "alcanzable" del SMART no se había validado. Cambiar el agente tampoco basta (más entrenamiento, menor costo de pregunta y tamizaje juntos: 65,2 %). Propuesta para el documento ajustado: validar M1 con datos reales (KTAS) y fijar la meta relativa a la referencia humana (según la guía consultada, enfermera y experto coinciden en ≈ 85 %). El umbral no se cambió tras ver resultados.

**M4 no se cumple: 87,1 % ± 12,2 pp (meta ≥ 90 %).** De 67 pacientes con riesgo vital real no escalados, en 40 % el agente nunca preguntó por el signo, en 37 % la respuesta fue errónea o "no sé" y en 22 % la confirmación contradijo la primera respuesta. Cambio propuesto: **tamizaje obligatorio de respiración y conciencia al inicio**; en el diagnóstico sube M4 a 94,5 % con 3,10 preguntas (aún ≤ 4). No se incluyó en los resultados oficiales. La dispersión es alta porque hay pocos pacientes con riesgo vital por repetición (≈ 52).

**Lecturas con cuidado.** M3 cumple en E3 (solo graves), pero en la mezcla normal E1 baja a 75,1 %: los graves son raros y el agente aprendió esa prevalencia. Además M3 agrupa los niveles 1 y 2 y oculta que el 80 % del sub-triage en graves es nivel 1 clasificado como 2: solo ≈ 51 % de los nivel 1 se identifican como tales. M2 también supera 10 % en casos límite (11,1 % en E2).

**Fallas halladas y corregidas.** La primera corrida (`fase3_resultados_corrida1_antes_de_corregir/`) dio M6 = 59,6 % y M7 = 79,8 %: el agente se caía con la API de la cola sin respuesta (`TypeError`, 95 casos) y con respuestas fuera de dominio (`ValueError`, 35 casos). Ahora una entrada inválida se trata como "no sé" y un sensor sin dato conserva el último valor, con advertencia en el log. Un 0 % inicial en M7 fue un error de mi arnés (log sin conectar), no del agente.

**Limitaciones.** Todo se probó contra un simulador propio: no es evidencia clínica. Los umbrales se fijaron antes de la Fase 3, pero conociendo los resultados de la Fase 2.

## 4. Cómo reproducir

```bash
python run_fase3.py                       # resultados oficiales (≈ 15 s)
python run_fase3.py --keep-logs all       # conserva los logs de las 10 repeticiones
python fase3/diagnostico.py               # causas de M1 y M4 y qué cambiaría en el agente
python -m pytest -q                       # 13 pruebas unitarias
```
