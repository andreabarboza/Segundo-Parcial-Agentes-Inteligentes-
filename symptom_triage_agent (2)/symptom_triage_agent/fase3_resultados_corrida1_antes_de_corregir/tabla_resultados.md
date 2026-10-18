# Fase 3 — Resultados de las pruebas

- Fecha de ejecución: 2026-10-18T16:41:39
- SHA-256 de `umbrales.json` (fijado antes de ejecutar): `0675639c241c1bc4c41a0e4d5b66dda36f0d7ac183939771d28897ad8f23dd09`
- 10 repeticiones por escenario, 100 pacientes por escenario y repetición, entrenamiento previo de 500 pacientes; total de pacientes ejecutados: 6000
- Resultado = media ± desviación estándar entre repeticiones (pp = puntos porcentuales)

## Tabla de resultados

| ID | Medida de desempeño | Fórmula | Umbral | Casos ejecutados | Resultado obtenido | ¿Cumple? | Evidencia |
|---|---|---|---|---|---|---|---|
| M1 | Coincidencia exacta con el profesional | casos con nivel del agente = nivel del profesional / casos totales | >= 80% | 1000 pacientes | 60.5% ± 4.5 pp | No | logs/E1_rep01.jsonl · resultados_por_corrida.csv (escenario E1) |
| M2 | Tasa de sub-triage | casos con nivel del agente MENOS urgente que el del profesional / casos totales | <= 10% | 1000 pacientes | 7.6% ± 3.4 pp | Sí | logs/E1_rep01.jsonl · resultados_por_corrida.csv (escenario E1) |
| M3 | Sensibilidad en pacientes graves | pacientes que el profesional marcó nivel 1-2 y el agente también marcó 1-2 / pacientes nivel 1-2 | >= 90% | 964 pacientes graves (de 1000) | 93.9% ± 4.5 pp | Sí | logs/E3_rep01.jsonl · resultados_por_corrida.csv (escenario E3) |
| M4 | Escalamiento ante riesgo vital | pacientes con signo de riesgo vital real que el agente escaló al personal / pacientes con signo de riesgo vital real | >= 90% | 517 pacientes con riesgo vital real (de 1000) | 87.1% ± 12.2 pp | No | logs/E3_rep01.jsonl · resultados_por_corrida.csv (escenario E3) |
| M5 | Eficiencia del interrogatorio | preguntas formuladas en total / pacientes atendidos | <= 4.0 | 1000 pacientes | 1.96 ± 0.29 | Sí | logs/E1_rep01.jsonl · resultados_por_corrida.csv (escenario E1) |
| M6 | Robustez ante fallas de entradas y sensores | casos que terminan sin excepción y con nivel 1-5 y resumen marcado para validación humana / casos totales | >= 100% | 3000 pacientes | 59.6% ± 2.0 pp | No | logs/E4_rep01.jsonl · resultados_por_corrida.csv (escenario E4, E5, E6) |
| M7 | Trazabilidad del log | episodios cuyo log contiene percepción, decisión, acción, retroalimentación y cierre con marca de tiempo / episodios totales | >= 100% | 6000 pacientes | 79.8% ± 1.0 pp | No | logs/E1_rep01.jsonl · resultados_por_corrida.csv (escenario E1, E2, E3, E4, E5, E6) |
| M8 | Reproducibilidad | repeticiones con la misma semilla que dan resultados idénticos / repeticiones | >= 100% | 10 repeticiones | 100.0% ± 0.0 pp | Sí | logs/E1_rep01.jsonl · resultados_por_corrida.csv (escenario E1) |

## Escenarios

| ID | Tipo | Descripción |
|---|---|---|
| E1 | normal | Pacientes típicos: simulador estándar (10 % de respuestas erróneas, 8 % de 'no sé', 3 % de deterioro por pregunta, 2 % de abandono, 70 % de consentimiento) |
| E2 | limite | Casos de frontera: solo niveles 2, 3 y 4, donde los síntomas se solapan más |
| E3 | normal | Pacientes graves: solo niveles 1 y 2 (para las medidas de seguridad) |
| E4 | falla_esperada | Respuestas poco fiables: 35 % erróneas, 35 % 'no sé', 15 % de abandono |
| E5 | falla_esperada | Sensores sin dato: sin historia clínica, API de la cola caída el 50 % del tiempo, 50 % de 'no sé' |
| E6 | falla_esperada | Entradas inválidas: 15 % de las respuestas llegan fuera de dominio ('???', vacío, número, texto libre, booleano) |

## Resultados por escenario (casos límite y de falla esperada; informativos)

| Medida | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|
| M1 Coincidencia exacta con el profesional | 60.5% ± 4.5 pp | 69.8% ± 4.3 pp | 64.7% ± 11.3 pp | 32.8% ± 4.5 pp | 2.4% ± 1.3 pp | 45.3% ± 3.5 pp |
| M2 Tasa de sub-triage | 7.6% ± 3.4 pp | 11.1% ± 2.5 pp | 30.6% ± 12.8 pp | 15.0% ± 3.6 pp | 0.8% ± 0.9 pp | 5.9% ± 2.0 pp |
| M3 Sensibilidad en pacientes graves | 75.1% ± 9.8 pp | 81.3% ± 6.0 pp | 93.9% ± 4.5 pp | 52.7% ± 15.2 pp | 2.3% ± 4.4 pp | 51.8% ± 9.7 pp |
| M4 Escalamiento ante riesgo vital | 77.4% ± 34.3 pp | — | 87.1% ± 12.2 pp | 29.5% ± 24.4 pp | 0.0% ± 0.0 pp | 59.3% ± 29.5 pp |
| M5 Eficiencia del interrogatorio | 1.96 ± 0.29 | 2.02 ± 0.27 | 2.55 ± 0.30 | 2.83 ± 0.14 | 0.13 ± 0.04 | 1.27 ± 0.08 |
| M6 Robustez ante fallas de entradas y sensores | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 4.8% ± 1.1 pp | 74.1% ± 6.0 pp |
| M7 Trazabilidad del log | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 100.0% ± 0.0 pp | 4.8% ± 1.1 pp | 74.1% ± 6.0 pp |
