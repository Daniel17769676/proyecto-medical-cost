"""Ejecutar desde esta carpeta: python ejecutar_validacion.py.

Reconstruye métricas, auditoría, gráficos, pruebas y pipeline a partir del CSV.
No necesita conexión de red una vez instaladas las dependencias.
"""
from pathlib import Path
import os, json, hashlib, time
from medical_cost import cargar_datos, entrenar_comparar, exportar_evidencias, crear_figuras

def main():
    folder=Path(__file__).resolve().parent;os.chdir(folder)
    started=time.perf_counter()
    print('Cargando y validando datos',flush=True)
    raw,df=cargar_datos('insurance.csv')
    print('Comparando modelos y realizando validación cruzada',flush=True)
    result=entrenar_comparar(df)
    print('Exportando métricas y ejecutando pruebas funcionales',flush=True)
    metrics,details,gaps,tests,drift,alerts,examples,importance=exportar_evidencias(raw,df,result,'evidencias')
    crear_figuras(df,result,details,gaps,importance,'evidencias/figuras')
    metrics['sha256_dataset']=hashlib.sha256(Path('insurance.csv').read_bytes()).hexdigest()
    metrics['segundos_ejecucion_validacion']=round(time.perf_counter()-started,3)
    Path('evidencias/metricas.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
    print(tests.to_string(index=False),flush=True)

if __name__=='__main__':main()
