# Champions League Predictor ⚽

Predicciones de resultados de la UEFA Champions League basadas en machine learning.

**[Ver la app en vivo](https://champions-predictor.streamlit.app)** *(placeholder)*

![App Screenshot](screenshot.png) *(placeholder)*

## Qué hace

Predice el resultado (victoria local, empate, victoria visitante) de partidos de Champions League usando estadísticas históricas de rendimiento de los equipos. Compara las probabilidades del modelo con las cuotas del mercado de apuestas (Pinnacle).

## Metodología

- **709 partidos** de 3 temporadas (2023-2025) de la UEFA Champions League
- **38 variables** basadas en promedios móviles de los últimos 5 partidos: xG, goles, posesión, tiros, pases, duelos, etc.
- **Modelo:** HistGradient Boosting (scikit-learn)
- **Validación:** Walk-forward temporal (entrena con pasado, evalúa con futuro)
- **Precisión:** ~52% en validación walk-forward (vs ~50% del baseline)

## Stack

- **Python 3.11** + scikit-learn para el modelo
- **Streamlit** para la interfaz web
- **Datos:** API-Football (estadísticas de partido y jugadores)
- **Altair** para visualizaciones

## Estructura

```
├── app.py                    # Streamlit app
├── 03_feature_engineering.py # Pipeline de features
├── 04_train_model.py         # Entrenamiento y evaluación
├── data/processed/           # CSVs listos para el modelo
├── models/                   # Modelo entrenado y métricas
└── .streamlit/config.toml    # Tema oscuro Champions League
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```
