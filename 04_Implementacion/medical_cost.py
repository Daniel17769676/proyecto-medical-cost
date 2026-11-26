"""Prototipo académico Medical Cost. No autoriza decisiones sobre personas.

El modelo serializado usa exclusivamente clases de scikit-learn. La API
operativa pública alerta_gasto valida el esquema y exige grupos de 20 filas.
El artefacto sigue siendo accesible: ese mínimo no acredita anonimización.
"""
from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
VERSION = '2.0-academica'
FEATURES = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
NUMERIC = ['age', 'bmi', 'children']
CATEGORICAL = ['sex', 'smoker', 'region']
CATEGORIES = [['female', 'male'], ['no', 'yes'], ['northeast', 'northwest', 'southeast', 'southwest']]
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)


def validar_datos(datos, con_objetivo=False, minimo=1):
    if not isinstance(datos, pd.DataFrame):
        raise ValueError('Se requiere un DataFrame.')
    if len(datos) < minimo:
        raise ValueError(f'Se requieren al menos {minimo} filas.')
    required = FEATURES + (['charges'] if con_objetivo else [])
    missing = set(required) - set(datos.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    d = datos[required].copy()
    for column in NUMERIC + (['charges'] if con_objetivo else []):
        if not pd.api.types.is_numeric_dtype(d[column]) or pd.api.types.is_bool_dtype(d[column]):
            raise ValueError(f'{column} debe ser numérica.')
        if not np.isfinite(d[column].to_numpy(dtype=float)).all():
            raise ValueError(f'{column} contiene nulos o valores no finitos.')
    if not d.age.between(18, 64).all() or not (d.age % 1 == 0).all():
        raise ValueError('age debe ser entero entre 18 y 64, rango cubierto por la muestra.')
    if not d.bmi.between(10, 70).all():
        raise ValueError('bmi debe estar entre 10 y 70. Los extremos requieren revisión de distribución.')
    if not d.children.between(0, 5).all() or not (d.children % 1 == 0).all():
        raise ValueError('children debe ser entero entre 0 y 5.')
    for column, allowed in zip(CATEGORICAL, CATEGORIES):
        if not d[column].isin(allowed).all():
            raise ValueError(f'{column} solo admite {allowed}.')
    if con_objetivo and not (d.charges > 0).all():
        raise ValueError('charges debe ser positivo.')
    return d


def cargar_datos(ruta='insurance.csv'):
    raw = pd.read_csv(ruta)
    validar_datos(raw, con_objetivo=True)
    return raw, raw.drop_duplicates().reset_index(drop=True)


def crear_pipeline(estimador):
    # Categorías y orden equivalentes al cuaderno original, sin inferir el
    # esquema desde prueba ni desde el lote de predicción o reentrenamiento.
    encoder = ColumnTransformer([
        ('numericas', 'passthrough', NUMERIC),
        ('categoricas', OneHotEncoder(categories=CATEGORIES, drop='first',
                                     sparse_output=False, handle_unknown='error'), CATEGORICAL)
    ], remainder='drop', verbose_feature_names_out=False)
    return Pipeline([('preparacion', encoder), ('modelo', estimador)])


def entrenar_comparar(df):
    d = validar_datos(df, con_objetivo=True)
    X_train, X_test, y_train, y_test = train_test_split(d[FEATURES], d.charges, test_size=.2, random_state=SEED)
    estimadores = {
        'Regresión lineal': LinearRegression(),
        'RF base': RandomForestRegressor(n_estimators=100, random_state=SEED),
        'RF manual': RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=SEED),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=.05, random_state=SEED)
    }
    rows = []
    for name, estimator in estimadores.items():
        pipe = crear_pipeline(estimator)
        cv = cross_validate(pipe, X_train, y_train, cv=CV, scoring=['r2', 'neg_mean_absolute_error'])
        pipe.fit(X_train, y_train)
        prediction = pipe.predict(X_test)
        rows.append({'modelo': name, 'r2_cv': cv['test_r2'].mean(), 'sd_r2_cv': cv['test_r2'].std(),
                     'mae_cv': -cv['test_neg_mean_absolute_error'].mean(), 'r2_train': pipe.score(X_train, y_train),
                     'r2_test': r2_score(y_test, prediction), 'mae_test': mean_absolute_error(y_test, prediction)})
    grid = {'modelo__max_depth': [4, 6, 8, 12], 'modelo__min_samples_leaf': [3, 5, 10, 20], 'modelo__n_estimators': [200, 400]}
    gs = GridSearchCV(crear_pipeline(RandomForestRegressor(random_state=SEED)), grid,
                      cv=CV, scoring='r2', n_jobs=1)
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    cv = cross_validate(model, X_train, y_train, cv=CV, scoring=['r2', 'neg_mean_absolute_error'])
    pred = model.predict(X_test)
    rows.append({'modelo': 'RF grilla seleccionado', 'r2_cv': gs.best_score_, 'sd_r2_cv': cv['test_r2'].std(),
                 'mae_cv': -cv['test_neg_mean_absolute_error'].mean(), 'r2_train': model.score(X_train, y_train),
                 'r2_test': r2_score(y_test, pred), 'mae_test': mean_absolute_error(y_test, pred)})
    oof_residuals = y_train.to_numpy() - cross_val_predict(model, X_train, y_train, cv=CV)
    q10, q90 = np.quantile(oof_residuals, [.1, .9])
    return {'modelo': model, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'pred': pred, 'comparacion': pd.DataFrame(rows), 'parametros': gs.best_params_, 'q10': float(q10), 'q90': float(q90)}


def evaluar(resultado):
    y = resultado['y_test'].to_numpy(); pred = resultado['pred']
    lower = np.maximum(0, pred + resultado['q10']); upper = pred + resultado['q90']
    top = y >= np.quantile(y, .9)
    cvrow = resultado['comparacion'].iloc[-1]
    return {'r2_prueba': float(r2_score(y, pred)), 'mae_prueba': float(mean_absolute_error(y, pred)),
            'rmse_prueba': float(mean_squared_error(y, pred) ** .5), 'r2_cv': float(cvrow.r2_cv),
            'sd_r2_cv': float(cvrow.sd_r2_cv), 'r2_train': float(cvrow.r2_train),
            'error_cartera_pct': float(100 * (pred.sum() / y.sum() - 1)),
            'sesgo_decil_superior_pct': float(100 * (pred[top].sum() / y[top].sum() - 1)),
            'n_decil_superior': int(top.sum()), 'cobertura_nominal': .8,
            'residuo_q10': resultado['q10'], 'residuo_q90': resultado['q90'],
            'cobertura_observada': float(((y >= lower) & (y <= upper)).mean()),
            'cubiertos': int(((y >= lower) & (y <= upper)).sum()), 'n_prueba': len(y),
            'n_entrenamiento': len(resultado['X_train']), 'unidad_charges': 'u.m. sin moneda acreditada',
            'semilla': SEED, 'parametros_seleccionados': resultado['parametros']}


def auditar_subgrupos(resultado, n_boot=1000):
    a = resultado['X_test'].copy()
    a['real'] = resultado['y_test'].to_numpy(); a['pred'] = resultado['pred']
    a['err_abs'] = (a.real-a.pred).abs()
    a['rango_imc'] = pd.cut(a.bmi, [0,25,30,100], labels=['hasta 25','mayor 25 hasta 30','mayor 30'])
    a['rango_edad'] = pd.cut(a.age, [17,30,45,65], labels=['18 a 30','31 a 45','46 a 64'])
    rng = np.random.default_rng(SEED)
    details, gaps = [], []
    for variable in ['sex','region','smoker','rango_imc','rango_edad']:
        maes = []
        for group, part in a.groupby(variable, observed=True):
            errors = part.err_abs.to_numpy(); mae = float(errors.mean()); maes.append(mae)
            means = rng.choice(errors, size=(n_boot,len(errors)), replace=True).mean(axis=1)
            low, high = np.quantile(means,[.025,.975])
            details.append({'variable': variable, 'subgrupo': str(group), 'n': len(part), 'mae': mae,
                            'costo_medio': part.real.mean(), 'error_relativo_pct': 100*mae/part.real.mean(),
                            'ic95_mae_inferior': low, 'ic95_mae_superior': high})
        gap = 100*(max(maes)-min(maes))/min(maes)
        gaps.append({'variable':variable, 'brecha_mae_pct':gap, 'limite_pct':15, 'cumple':bool(gap<=15)})
    return pd.DataFrame(details), pd.DataFrame(gaps)


def predecir_didactico(perfiles, modelo, q10, q90):
    d = validar_datos(perfiles)
    output = perfiles.copy(); output['prediccion_um'] = modelo.predict(d)
    output['rango80_inferior'] = np.maximum(0, output.prediccion_um + q10)
    output['rango80_superior'] = output.prediccion_um + q90
    return output


def perfiles_ejemplo():
    return pd.DataFrame([
        ['Adulto joven',25,'female',22.,0,'no','southwest'],
        ['Adulto con IMC alto',40,'male',33.5,2,'no','southeast'],
        ['Fumador de mediana edad',45,'male',28.,1,'yes','northwest'],
        ['Fumador con IMC alto',35,'female',35.,0,'yes','northeast'],
        ['Adulto mayor no fumador',60,'female',27.,0,'no','northeast']
    ], columns=['perfil']+FEATURES)


def alerta_gasto(grupo, umbral, modelo):
    d = validar_datos(grupo, minimo=20)
    if isinstance(umbral, (bool, np.bool_)) or not isinstance(umbral,(float,int,np.number)) or not np.isfinite(umbral) or umbral <= 0:
        raise ValueError('El umbral debe ser un número finito positivo en las mismas u.m. de charges.')
    cost = float(modelo.predict(d).sum())
    return {'n_afiliados':len(d), 'costo_proyectado_um':cost, 'umbral_um':float(umbral),
            'alerta':bool(cost>umbral), 'uso_porcentual_umbral':100*cost/umbral}


def deriva_datos(nuevos, referencia_entrenamiento, alfa=.01):
    ref = validar_datos(referencia_entrenamiento, minimo=20)
    new = validar_datos(nuevos, minimo=20)
    if not 0 < alfa < 1:
        raise ValueError('alfa debe estar entre 0 y 1.')
    rows = []
    for column in ['age','bmi']:
        p = stats.ks_2samp(ref[column],new[column]).pvalue
        rows.append({'variable':column,'prueba':'KS','p_sin_ajuste':p,'esperado_minimo':None})
    for column, cats in [('children',list(range(6)))] + list(zip(CATEGORICAL,CATEGORIES)):
        counts = np.array([[int((ref[column]==v).sum()),int((new[column]==v).sum())] for v in cats])
        counts = counts[counts.sum(axis=1)>0]
        if len(counts)<2:
            p,minimum = 1.,None
        else:
            _,p,_,expected = stats.chi2_contingency(counts)
            minimum = float(expected.min())
        rows.append({'variable':column,'prueba':'Chi cuadrado','p_sin_ajuste':p,'esperado_minimo':minimum})
    pvals = np.array([r['p_sin_ajuste'] for r in rows]); order = np.argsort(pvals)
    adjusted = np.empty(len(rows)); adjusted[order] = np.minimum(1,np.maximum.accumulate(pvals[order]*(len(rows)-np.arange(len(rows)))))
    for r,p in zip(rows,adjusted):
        r['p_ajuste_holm']=float(p);r['alfa']=alfa;r['deriva']=bool(p<alfa)
    return pd.DataFrame(rows)


def reentrenar_candidato(datos, parametros, ruta_candidato, meta=.8):
    """Genera un candidato separado. Nunca reemplaza el modelo validado."""
    if isinstance(meta,bool) or not isinstance(meta,(float,int,np.number)) or not np.isfinite(meta) or not 0 < meta <= 1:
        raise ValueError('La meta de aceptación debe estar entre 0 y 1.')
    path=Path(ruta_candidato);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError('El candidato ya existe. Use una ruta nueva para conservar trazabilidad.')
    d = validar_datos(datos,con_objetivo=True,minimo=100).drop_duplicates()
    a,b,ya,yb = train_test_split(d[FEATURES],d.charges,test_size=.2,random_state=SEED)
    estimator_params = {k.replace('modelo__',''):v for k,v in parametros.items()}
    m = crear_pipeline(RandomForestRegressor(**estimator_params,random_state=SEED)).fit(a,ya)
    pred = m.predict(b)
    record={'semilla':SEED,'version':VERSION,'registros':len(d),'r2_prueba':float(r2_score(yb,pred)),
            'mae_prueba':float(mean_absolute_error(yb,pred)),'meta_rf1':.8,
            'cumple_rf1':bool(r2_score(yb,pred)>=.8),'meta_aceptacion':meta,
            'cumple_meta_aceptacion':bool(r2_score(yb,pred)>=meta),'promocion_automatica':False,
            'parametros':parametros,'fecha_utc':datetime.now(timezone.utc).isoformat()}
    joblib.dump(m,path)
    path.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return m,record


def probar_funciones(df, resultado, carpeta):
    out=Path(carpeta);out.mkdir(exist_ok=True,parents=True)
    model=resultado['modelo'];base=df[FEATURES]
    normal=base.sample(100,random_state=1)
    risk=pd.concat([base[base.smoker=='yes'].sample(60,random_state=1),base[base.smoker=='no'].sample(40,random_state=1)])
    records=[]
    def check(name, fn):
        try:
            assert bool(fn())
            records.append({'prueba':name,'resultado':'OK','detalle':'Criterio comprobado'})
        except Exception as exc:
            records.append({'prueba':name,'resultado':'ERROR','detalle':str(exc)})
    def reject(fn):
        try: fn()
        except ValueError: return True
        return False
    check('Alerta negativa en grupo habitual',lambda:not alerta_gasto(normal,1_300_000,model)['alerta'])
    check('Alerta positiva en grupo simulado',lambda:alerta_gasto(risk,1_300_000,model)['alerta'])
    check('Rechazo de grupo de 19 filas',lambda:reject(lambda:alerta_gasto(base.head(19),1,model)))
    check('Rechazo de umbral cero',lambda:reject(lambda:alerta_gasto(normal,0,model)))
    check('Rechazo de umbral negativo',lambda:reject(lambda:alerta_gasto(normal,-1,model)))
    check('Rechazo de edad negativa',lambda:reject(lambda:alerta_gasto(normal.assign(age=-1),1,model)))
    check('Rechazo de IMC fuera de rango',lambda:reject(lambda:alerta_gasto(normal.assign(bmi=0),1,model)))
    check('Rechazo de hijos fraccionarios',lambda:reject(lambda:alerta_gasto(normal.assign(children=.5),1,model)))
    check('Rechazo de categoria desconocida',lambda:reject(lambda:alerta_gasto(normal.assign(smoker='otro'),1,model)))
    check('Rechazo de columna faltante',lambda:reject(lambda:alerta_gasto(normal.drop(columns='sex'),1,model)))
    check('Rechazo de valores no finitos',lambda:reject(lambda:alerta_gasto(normal.assign(bmi=np.nan),1,model)))
    target=out/'pipeline_medical_cost.joblib';joblib.dump(model,target)
    loaded=joblib.load(target)
    check('Persistencia de pipeline equivalente',lambda:np.allclose(loaded.predict(resultado['X_test']),resultado['pred'],rtol=0,atol=1e-9))
    check('Pipeline rechaza categorias desconocidas',lambda:reject(lambda:loaded.predict(normal.assign(region='otra'))))
    sim=df.sample(300,random_state=5).copy();fum=sim[sim.smoker=='yes']
    sim=pd.concat([sim[sim.smoker=='no'].sample(165,random_state=2),fum.sample(135,replace=True,random_state=2)])
    sim['age']=(sim.age+8).clip(upper=64)
    drift=deriva_datos(sim,resultado['X_train'])
    check('Deriva simulada en edad y tabaquismo',lambda:drift.set_index('variable').loc[['age','smoker'],'deriva'].all())
    check('Deriva rechaza referencia invalida',lambda:reject(lambda:deriva_datos(sim,resultado['X_train'].assign(age=-1))))
    candidate_path=out/'candidato_reentrenamiento.joblib'
    if candidate_path.exists():candidate_path.unlink()
    _,record=reentrenar_candidato(df,resultado['parametros'],candidate_path,meta=1.)
    check('Candidato sin promocion automatica',lambda:not record['promocion_automatica'] and not record['cumple_meta_aceptacion'])
    check('Modelo validado conservado',lambda:np.allclose(joblib.load(target).predict(resultado['X_test']),resultado['pred'],rtol=0,atol=1e-9))
    try:
        reentrenar_candidato(df,resultado['parametros'],candidate_path)
        blocked=False
    except FileExistsError:blocked=True
    check('Candidato existente no se sobrescribe',lambda:blocked)
    results=pd.DataFrame(records)
    results.to_csv(out/'pruebas_funcionales.csv',index=False,encoding='utf-8-sig')
    drift.to_csv(out/'deriva_simulada.csv',index=False,encoding='utf-8-sig')
    alerts={'grupo_habitual':alerta_gasto(normal,1_300_000,model),'grupo_simulado':alerta_gasto(risk,1_300_000,model)}
    (out/'alertas_simuladas.json').write_text(json.dumps(alerts,ensure_ascii=False,indent=2),encoding='utf-8')
    return results,drift,alerts


def exportar_evidencias(raw,df,resultado,carpeta):
    out=Path(carpeta);out.mkdir(parents=True,exist_ok=True)
    metrics=evaluar(resultado)
    latencias=[]
    for _ in range(10):
        start=time.perf_counter();resultado['modelo'].predict(resultado['X_test'])
        latencias.append(1000*(time.perf_counter()-start))
    metrics['latencia_lote_prueba_268_filas_ms_mediana_10']=float(np.median(latencias))
    details,gaps=auditar_subgrupos(resultado)
    tests,drift,alerts=probar_funciones(df,resultado,out)
    import sklearn,scipy,sys
    metrics.update({'version':VERSION,'registros_originales':len(raw),'registros_unicos':len(df),
                    'duplicados_eliminados':len(raw)-len(df),'nulos':int(raw.isna().sum().sum()),
                    'pruebas_ok':int((tests.resultado=='OK').sum()),'pruebas_total':len(tests),
                    'versiones':{'python':sys.version.split()[0],'scikit-learn':sklearn.__version__,
                                 'pandas':pd.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'joblib':joblib.__version__},
                    'fecha_utc':datetime.now(timezone.utc).isoformat()})
    (out/'metricas.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    resultado['comparacion'].to_csv(out/'comparacion_modelos.csv',index=False,encoding='utf-8-sig')
    details.to_csv(out/'auditoria_subgrupos.csv',index=False,encoding='utf-8-sig')
    gaps.to_csv(out/'brechas_equidad.csv',index=False,encoding='utf-8-sig')
    examples=predecir_didactico(perfiles_ejemplo(),resultado['modelo'],resultado['q10'],resultado['q90'])
    examples.to_csv(out/'predicciones_didacticas.csv',index=False,encoding='utf-8-sig')
    predictions=resultado['X_test'].copy();predictions['real']=resultado['y_test'];predictions['pred']=resultado['pred']
    predictions.to_csv(out/'predicciones_prueba.csv',index=False,encoding='utf-8-sig')
    importance=pd.DataFrame({'variable':resultado['modelo'].named_steps['preparacion'].get_feature_names_out(),
                             'importancia':resultado['modelo'].named_steps['modelo'].feature_importances_}).sort_values('importancia',ascending=False)
    importance.to_csv(out/'importancia_variables.csv',index=False,encoding='utf-8-sig')
    if not (tests.resultado=='OK').all():raise AssertionError('Hay pruebas funcionales fallidas.')
    return metrics,details,gaps,tests,drift,alerts,examples,importance


def crear_figuras(df, resultado, detalles, brechas, importancia, carpeta):
    import matplotlib.pyplot as plt
    import seaborn as sns
    out=Path(carpeta);out.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style='whitegrid',font='Arial',font_scale=1.05)
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/name,dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,3.5))
    sns.histplot(df.charges,bins=32,ax=axs[0],color='#294b73');axs[0].set(xlabel='Costo observado en u.m.',ylabel='Registros',title='Distribución de charges')
    sns.boxplot(data=df,x='smoker',y='charges',ax=axs[1],color='#9fb7cc');axs[1].set(xlabel='Tabaquismo',ylabel='Costo en u.m.',title='Asociación descriptiva')
    sns.scatterplot(data=df,x='bmi',y='charges',hue='smoker',ax=axs[2],alpha=.5,s=16);axs[2].set(xlabel='IMC',ylabel='Costo en u.m.',title='IMC y costos observados')
    save(fig,'eda.png')
    fig,axs=plt.subplots(1,2,figsize=(11,3.7))
    y=resultado['y_test'];pred=resultado['pred'];lim=max(y.max(),pred.max())
    axs[0].scatter(y,pred,s=16,alpha=.55,color='#294b73');axs[0].plot([0,lim],[0,lim],'--',color='#bb653b');axs[0].set(xlabel='Real en u.m.',ylabel='Predicho en u.m.',title='Prueba interna de 268 filas')
    axs[1].hist(y.to_numpy()-pred,bins=30,color='#9fb7cc');axs[1].set(xlabel='Real menos predicho en u.m.',ylabel='Registros',title='Errores individuales')
    save(fig,'diagnostico.png')
    fig,ax=plt.subplots(figsize=(8,3.8));order=importancia.iloc[::-1]
    labels={'smoker_yes':'Tabaquismo','bmi':'IMC','age':'Edad','children':'Hijos','sex_male':'Sexo','region_northwest':'Región noroeste','region_southeast':'Región sureste','region_southwest':'Región suroeste'}
    ax.barh([labels.get(x,x) for x in order.variable],order.importancia*100,color='#294b73');ax.set(xlabel='Importancia por reducción de impureza (%)',title='Importancia predictiva de variables')
    save(fig,'importancia.png')
    fig,ax=plt.subplots(figsize=(9,4.6));labels=detalles.variable+' '+detalles.subgrupo
    ax.errorbar(detalles.mae,np.arange(len(detalles)),xerr=[detalles.mae-detalles.ic95_mae_inferior,detalles.ic95_mae_superior-detalles.mae],fmt='o',color='#294b73',ecolor='#9fb7cc',capsize=3)
    ax.set_yticks(np.arange(len(detalles)),labels);ax.invert_yaxis();ax.set(xlabel='MAE en u.m. e intervalo bootstrap del 95 %',title='Error e incertidumbre por subgrupo')
    save(fig,'equidad_intervalos.png')
    fig,ax=plt.subplots(figsize=(8,3.5));colors=['#3d8066' if b else '#b75c43' for b in brechas.cumple]
    ax.bar(brechas.variable,brechas.brecha_mae_pct,color=colors);ax.axhline(15,color='#294b73',ls='--',label='Criterio RF4 de 15 %');ax.legend();ax.set(ylabel='Brecha relativa de MAE (%)',title='RF4 mantiene tres incumplimientos')
    save(fig,'brechas_equidad.png')
