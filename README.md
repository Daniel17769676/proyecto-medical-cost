# Proyecto: Estimación de Costos Médicos

**Asignatura:** Gestión de Proyectos de Datos · Sigla: SCY1102  
**Docente:** C. Aníbal Sotelo Becerra  
**Integrantes (Grupo 2):**
- Daniel Cabrera Araya
- Martín Carrasco Lineros
- Fabián Solar
- David Vásquez Ovalle

**Descripción:**  
Implementación de un modelo predictivo para el dataset **Medical Cost Personal Datasets**, orientado a la estimación agregada de costos médicos anuales para una aseguradora de salud privada. El proyecto se desarrolla en el marco de la asignatura Gestión de Proyectos de Datos y utiliza la metodología CRISP-DM, con énfasis en la ética, la privacidad y el uso exclusivamente agregado de los resultados.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| [`01_Evaluacion3/`](01_Evaluacion3/) | Informe de la Evaluación 3 (PDF) |
| [`03_Presentaciones/`](03_Presentaciones/) | Presentación final del Grupo 2 (PDF) |
| [`04_Implementacion/`](04_Implementacion/) | Código Python, notebook, dataset y evidencias generadas |
| [`05_Gestion/`](05_Gestion/) | Planilla de seguimiento y registros de gestión (CSV) |

## Implementación técnica

- **Notebook:** [`Modelo_MedicalCost_Final.ipynb`](04_Implementacion/Modelo_MedicalCost_Final.ipynb), con la ejecución y las figuras.
- **Módulo:** [`medical_cost.py`](04_Implementacion/medical_cost.py), que contiene la carga, el entrenamiento, la comparación de modelos y la exportación de evidencias.
- **Entregable funcional:** el pipeline entrenado [`pipeline_medical_cost.joblib`](04_Implementacion/evidencias/pipeline_medical_cost.joblib).
- **Evidencias:** en [`04_Implementacion/evidencias/`](04_Implementacion/evidencias/) están las métricas, la comparación de modelos, las pruebas funcionales, la auditoría por subgrupos, las brechas de equidad, la deriva simulada y las figuras.

### Cómo reproducir

Entorno verificado: Python 3.10.11. Desde `04_Implementacion/`:

```bash
python -m pip install -r requirements.txt
python ejecutar_validacion.py    # regenera las evidencias desde insurance.csv
python verificar_evidencias.py   # verifica la integridad de las evidencias
```

## Monitoreo y control

[`Seguimiento_MedicalCost.xlsx`](05_Gestion/Seguimiento_MedicalCost.xlsx) es la herramienta de gestión. Sus hojas también se publican como CSV:

| Archivo | Uso |
|---|---|
| [`tareas.csv`](05_Gestion/tareas.csv) | Tareas planificadas vs. ejecutadas |
| [`requisitos.csv`](05_Gestion/requisitos.csv) | Requisitos y su estado de cumplimiento |
| [`riesgos.csv`](05_Gestion/riesgos.csv) | Registro de riesgos |
| [`acciones.csv`](05_Gestion/acciones.csv) | Acciones correctivas |
| [`cambios.csv`](05_Gestion/cambios.csv) | Control de cambios |
| [`simulacion.csv`](05_Gestion/simulacion.csv) | Simulación de seguimiento |

> `charges` se expresa en unidades monetarias (u.m.) sin moneda acreditada. Los resultados son de uso interno y agregado.
