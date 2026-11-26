"""Comprobación de solo lectura en un proceso independiente.

Ejecutar desde esta carpeta: python verificar_evidencias.py
"""
from pathlib import Path
import json,hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from medical_cost import FEATURES,SEED,cargar_datos,alerta_gasto

def main():
    base=Path(__file__).resolve().parent;ev=base/'evidencias'
    m=json.loads((ev/'metricas.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((base/'insurance.csv').read_bytes()).hexdigest()==m['sha256_dataset']
    raw,df=cargar_datos(base/'insurance.csv')
    assert (len(raw),len(df))==(1338,1337)
    _,X,_,y=train_test_split(df[FEATURES],df.charges,test_size=.2,random_state=SEED)
    pipeline=joblib.load(ev/'pipeline_medical_cost.joblib')
    assert isinstance(pipeline,Pipeline)
    pred=pipeline.predict(X)
    scores={'r2_prueba':r2_score(y,pred),'mae_prueba':mean_absolute_error(y,pred),'rmse_prueba':mean_squared_error(y,pred)**.5}
    for name,value in scores.items():assert np.isclose(value,m[name],rtol=0,atol=1e-9),name
    saved=pd.read_csv(ev/'predicciones_prueba.csv')
    assert np.allclose(saved['pred'],pred,rtol=0,atol=1e-9)
    assert np.allclose(saved['real'],y,rtol=0,atol=1e-9)
    covered=(y>=np.maximum(0,pred+m['residuo_q10'])) & (y<=pred+m['residuo_q90'])
    assert int(covered.sum())==209 and np.isclose(covered.mean(),m['cobertura_observada'])
    checks=pd.read_csv(ev/'pruebas_funcionales.csv')
    assert len(checks)==18 and checks.resultado.eq('OK').all()
    gaps=pd.read_csv(ev/'brechas_equidad.csv')
    assert gaps.loc[~gaps.cumple,'variable'].tolist()==['region','smoker','rango_imc']
    candidate=json.loads((ev/'candidato_reentrenamiento.json').read_text(encoding='utf-8'))
    assert candidate['meta_rf1']==.8 and candidate['cumple_rf1']
    assert candidate['meta_aceptacion']==1 and not candidate['cumple_meta_aceptacion'] and not candidate['promocion_automatica']
    try:alerta_gasto(df.head(19),100,pipeline)
    except ValueError:pass
    else:raise AssertionError('No rechazó un grupo pequeño')
    assert m['latencia_lote_prueba_268_filas_ms_mediana_10']>0
    print('OK: datos, pipeline recargado, métricas, predicciones, cobertura, 18 pruebas, RF4 y candidato coherentes.')

if __name__=='__main__':main()
