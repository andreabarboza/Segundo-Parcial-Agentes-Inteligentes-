# Diseño vs. implementación

Lo que cambió respecto al documento de diseño (Fase 1) y por qué.

## 1. Cambios y justificación

| Aspecto | Diseño | Implementación | Justificación |
|---|---|---|---|
| Entorno | Servicio de urgencias real | Simulador (`triage/environment.py`) | No hay acceso a un servicio de urgencias ni a pacientes reales; el simulador genera las percepciones de la sección 5 |
| Entrada de lenguaje natural | NLP: comprensión de intención y extracción de entidades | Las respuestas del paciente ya llegan como categorías (p. ej. `dificultad_severa`) | El simulador no genera texto libre. El módulo de NLP queda como trabajo futuro (ver sección 3) |
| Protocolo | Manchester o ESI | Escala de 5 niveles genérica con costo asimétrico; sin las reglas completas de Manchester ni ESI | Implementar un protocolo completo excede el alcance; los 5 niveles son los mismos |
| Aprendizaje | Fine-tuning sobre cuestionarios y casos históricos | Actualización de conteos bayesiana con la corrección del clínico | Es un aprendizaje por retroalimentación sencillo y verificable; no se dispone de un conjunto de casos reales para fine-tuning |
| Escalamiento por riesgo vital | Escalar de inmediato | Se confirma **una vez** la respuesta de riesgo vital; si no se puede confirmar ("no sé") o el paciente se va, se escala igual | Las respuestas tienen ~10 % de error: sin confirmar, ~14 % de pacientes disparaba una falsa alarma y quedaba en nivel 1-2. Se detectó al medir los primeros resultados. Costo: una pregunta más ante una emergencia real |
| Historial clínico | Con consentimiento explícito | Se simula con 70 % de consentimiento; solo aporta el antecedente crónico | Es el único dato de historia clínica que usa el modelo |
| Agendamiento | API del sistema institucional | `ToolKit.assign_slot()` sobre una cola simulada con llegadas y atención aleatorias | No hay sistema institucional; la interfaz (llamadas a herramientas) es la misma |
| Medida de desempeño | ≥ 80 % de coincidencia con el profesional | Se mide igual, pero **no se alcanza** (≈ 60 %) | Ver sección 4 |

## 2. Propiedades del ambiente en el simulador

| Propiedad (sección 3) | Cómo se respeta |
|---|---|
| Parcialmente observable | Nivel real y valores reales están ocultos; el agente solo ve respuestas |
| Estocástico | Síntomas muestreados; 10 % de respuestas erróneas, 8 % de "no sé" |
| Secuencial | Cada pregunta depende de la creencia construida con las anteriores |
| Dinámico | El paciente puede empeorar (3 % por pregunta) o abandonar (2 %); la cola cambia sola |
| Discreto | Niveles, síntomas y respuestas son categorías |
| Agente único | Un agente, sin interacción con otros |

## 3. Qué hace el código y qué las herramientas externas

- **Todo el razonamiento lo hace el código propio** (`triage/agent.py`): creencia, función de
  utilidad, valor esperado de cada pregunta, aprendizaje y banderas rojas.
- **No se usa LLM.** Por eso no hay modelo, prompt de sistema ni decisión dependiente de un LLM.
- **Librerías externas:** ninguna (solo la biblioteca estándar de Python). `pytest` solo para pruebas.
- **Dónde se podría conectar un LLM más adelante** sin cambiar el núcleo de decisión: (a) extraer
  los síntomas de texto libre y convertirlos a las categorías de `features.py`; (b) redactar las
  preguntas con lenguaje natural y empático. La decisión de qué preguntar y qué nivel asignar
  seguiría en el código, que es auditable.

## 4. Límites de los resultados

- Los números describen el **simulador**, cuyas tablas diseñé yo; no demuestran desempeño clínico.
- El simulador se corrigió una vez: la primera versión tenía síntomas tan solapados que ni un agente
  perfecto superaba ~56 % de exactitud, y se volvieron más discriminativos (techo ~84 %).
  Esto es un ajuste del simulador, no del agente, y debe declararse como tal.
- No se alcanza 80 % de coincidencia exacta, por el techo del simulador y por la asimetría
  del costo (prefiere sobre-triage a sub-triage). La concordancia a ±1 nivel es ~96 %.
- Siguiente paso recomendado: calibrar el simulador con un dataset real (p. ej. KTAS de Kaggle)
  para que las tablas de síntomas dejen de ser inventadas.

## 5. Modo con datos reales (dataset KTAS)

Además del simulador, el agente se puede evaluar con casos reales (`triage/ktas.py`, `run_ktas.py`).
Cambios y supuestos de ese modo:

| Aspecto | Qué se hizo | Justificación |
|---|---|---|
| Síntomas | Se discretizan `Mental`, `Saturation`, `RR`, `HR`, `SBP`, `BT` y dolor en 3-4 categorías ordinales | El agente razona con categorías; los umbrales son convenciones clínicas comunes que asumí, no reglas oficiales de KTAS |
| Texto libre | `Chief_complain` se guarda en el resumen pero **no entra a la decisión** | Requiere NLP; es la limitación más importante de este modo |
| Datos faltantes | `""`, `??` y `#BOŞ!` se tratan como "no sé" (sin evidencia) | Rellenar con 0 o con la media inventaría datos clínicos |
| Fugas de información | El agente no tiene variables de diagnóstico, destino, tiempos ni de la etiqueta | Evita que "vea" el resultado; hay una prueba automática que lo comprueba |
| Dinámica | Sin ruido, abandono ni deterioro simulados; la cola sigue simulada | Los datos reales ya traen su propia incertidumbre (faltantes) |
| Evaluación | Partición estratificada 70/30 repetida con varias semillas; objetivo `KTAS_expert` | Mide la coincidencia con el criterio experto, como en la sección 7 del diseño |
| Referencia humana | Se reporta el acuerdo `KTAS_RN` vs `KTAS_expert` del propio dataset | Da sentido a la meta de 80 %: es la vara contra la que se compara |
| Exploración | **Durante el entrenamiento el agente pregunta todo**; en la prueba usa su política de utilidad | Se detectó una trampa de aprendizaje: con datos reales cada síntoma aporta poco por separado, así que el agente creía que no valía la pena preguntar y nunca aprendía lo contrario |

Hallazgo de diseño: la selección de preguntas por ganancia esperada es *miope* (mide una pregunta a
la vez). Cuando ninguna pregunta aislada cambia la decisión, el agente deja de preguntar aunque varias
juntas sí ayudarían. La exploración en el entrenamiento lo mitiga; una mejora futura sería una
búsqueda de más de un paso.

Limitaciones de los datos: sin identificador de paciente (posible mezcla entre entrenamiento y
prueba), 1.267 registros de solo dos hospitales y 20 días del año, y KTAS no equivale a Manchester ni a
ESI.

## 6. Cambios hechos a raíz de la Fase 3

| Hallazgo de las pruebas | Cambio | Resultado |
|---|---|---|
| El agente se caía si la API de la cola no respondía (`TypeError`, 95 casos) | Un sensor sin dato conserva el último valor conocido y deja una advertencia en el log | M6 pasó de 59,6 % a 100 % |
| El agente se caía ante respuestas fuera de dominio (`ValueError`, 35 casos) | Una entrada inválida se trata como "no sé" y se registra como advertencia | M7 pasó de 79,8 % a 100 % |
| Solo 87,1 % de los pacientes con riesgo vital se escalaban (M4) | Se añadió la opción `screening` (tamizaje obligatorio de respiración y conciencia). **Apagada por defecto**: los resultados oficiales no la incluyen | En el diagnóstico, M4 sube a 94,5 % con 3,10 preguntas |
| M1 (coincidencia exacta) queda en ~60 % | Sin cambio en el agente: ni más entrenamiento, ni menor costo de pregunta, ni tamizaje lo llevan cerca de 80 % (máx. 65,2 %) | Se propone validar con datos reales y fijar la meta relativa a la referencia humana |

Los umbrales de `fase3/umbrales.json` no se modificaron tras ver resultados.
