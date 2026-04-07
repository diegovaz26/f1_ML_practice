# 🏎️ Pipeline de Machine Learning — Fórmula 1

Actividad de Machine Learning implementada con **Scikit-Learn**, **GridSearchCV**, **Cross-Validation** y exportación con **joblib**. Los datos corresponden a estadísticas históricas de la Fórmula 1 desde 1950.

---

## 📂 Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `f1_pipeline.ipynb` | Notebook principal con todo el flujo de ML |
| `pipeline_regresion_lineal.pkl` | Pipeline entrenado — predicción de puntos |
| `pipeline_regresion_logistica.pkl` | Pipeline entrenado — predicción de podio |
| `requirements.txt` | Dependencias del proyecto |
| `driver_details.csv` | Resultados por carrera y piloto (~19 800 registros) |
| `driver_standings.csv` | Clasificaciones finales de pilotos por temporada |
| `fastest_laps.csv` | Vueltas rápidas por Gran Premio |
| `constructor_standings.csv` | Clasificaciones de constructores por temporada |

---

## 🎯 Objetivos

1. Implementar un **Pipeline de Scikit-Learn** que combine múltiples pasos de preprocesamiento y entrenamiento de modelos.
2. Guardar y cargar pipelines entrenados utilizando **`joblib`** para su uso en producción.

---

## 🔬 Técnicas utilizadas

### Feature Engineering
A partir del dataset crudo se construyeron las siguientes variables derivadas:

- `Position_Num` — posición numérica de llegada (DNF/DSQ → NaN).
- `Finished` — bandera binaria: 1 si el piloto terminó la carrera, 0 si no.
- `Races_So_Far` — número de carreras acumuladas del piloto hasta esa fecha (proxy de experiencia).
- `Car_Season_PTS` — puntos totales del constructor en esa temporada.
- `Car_Cat` — constructor codificado (top 20 + categoría "Other").
- `Podium` — variable objetivo binaria: 1 si el piloto terminó en posiciones 1–3.

### Pipeline (Scikit-Learn)
Cada pipeline sigue esta estructura:

```
ColumnTransformer
├── Numéricas → SimpleImputer → StandardScaler
└── Categóricas → SimpleImputer → OneHotEncoder
        ↓
    Modelo (LinearRegression / LogisticRegression)
```

### GridSearchCV
Se realizó búsqueda de hiperparámetros con validación cruzada:

- **Regresión lineal**: estrategia de imputación, `fit_intercept`, `positive`.
- **Regresión logística**: `C`, `solver`, `penalty`, `class_weight`.

### Cross-Validation
- **KFold (k=5)** para regresión — métrica: R².
- **StratifiedKFold (k=5)** para clasificación — métrica: F1 (preserva balance de clases).

---

## 📊 Modelos y Tareas

| Tarea | Modelo | Variable Objetivo |
|---|---|---|
| Regresión | `LinearRegression` | `PTS` — puntos obtenidos en carrera |
| Clasificación | `LogisticRegression` | `Podium` — Top 3 (1) o no (0) |

---

## 📈 Evaluación del Modelo

### Regresión Lineal — Predicción de Puntos
| Métrica | Valor |
|---|---|
| R² (test) | Ver notebook |
| RMSE | Ver notebook |
| MAE | Ver notebook |
| R² Cross-Val (media ± std) | Ver notebook |

Visualizaciones incluidas:
- Gráfico de puntos predichos vs reales.
- Barras de R² por fold.

### Regresión Logística — Predicción de Podio
| Métrica | Valor |
|---|---|
| Accuracy (test) | Ver notebook |
| F1 Cross-Val (media ± std) | Ver notebook |

Visualizaciones incluidas:
- Matriz de confusión.
- Barras de F1 por fold.

---

## 💾 Exportación y Carga del Pipeline

Los pipelines entrenados (con los mejores hiperparámetros encontrados por GridSearchCV) se guardan como archivos `.pkl`:

```python
import joblib

# Guardar
joblib.dump(best_reg, 'pipeline_regresion_lineal.pkl')
joblib.dump(best_clf, 'pipeline_regresion_logistica.pkl')

# Cargar y predecir
modelo = joblib.load('pipeline_regresion_lineal.pkl')
predicciones = modelo.predict(nuevos_datos)
```

Los archivos `.pkl` contienen **todo el pipeline**: transformadores ajustados + modelo entrenado. No es necesario repetir el preprocesamiento al momento de hacer predicciones en producción.

---

## 🚀 Cómo ejecutar

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Coloca los cuatro CSV en el mismo directorio que el notebook.

3. Abre el notebook:
   ```bash
   jupyter lab f1_ml_pipeline.ipynb
   ```

4. Ejecuta todas las celdas en orden (`Run All`).

Los archivos `.pkl` se generarán automáticamente al ejecutar la sección 12 del notebook.

---

## 🗂️ Estructura del Notebook

| Sección | Contenido |
|---|---|
| 1 | Importación de librerías |
| 2 | Carga y exploración de datos |
| 3 | Análisis Exploratorio (EDA) |
| 4 | Feature Engineering |
| 5 | Preparación del dataset |
| 6 | Pipeline — Regresión Lineal |
| 7 | GridSearchCV — Regresión Lineal |
| 8 | Evaluación — Regresión Lineal |
| 9 | Pipeline — Regresión Logística |
| 10 | Evaluación — Regresión Logística |
| 11 | Resumen de métricas |
| 12 | Exportación con `joblib` |
