# Importar las herramientas y librerias 
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Cargar el dataset en csv y seleccionar la cartera de pólizas de 2024
base = Path(__file__).resolve().parent
ruta_csv = base / "data" / "raw" / "polizas.csv"

polizas = pd.read_csv(ruta_csv)
polizas_2024 = polizas.loc[polizas["year"] == 2024].copy()


# Identificar las variables numéricas, las categóricas y el estado a predecir
numericas_eda = [
    "total_premium",    # Importe del premio
    "total_claims",     # Cantidad de siniestros declarados en el período
    "driver_age",       # Edad del conductor principal
    "vehicle_age",      # Antigüedad del vehículo
    "vehicle_value",    # Valor asegurado del vehículo
]

categoricas = [
    "policy_type",        # Tipo de cobertura
    "business_type",      # Nueva contratación o cartera existente
    "payment_frequency",  # Frecuencia de pago
    "bonus_score",        # Clasificación según el historial de siniestros
    "circulation_area",   # Entorno de circulación del vehículo
]

objetivo = "policy_status"
columnas_eda = numericas_eda + categoricas + [objetivo]


# Inspeccioar dimensiones y tipos de datos 
print("Dimensiones de la cartera 2024:", polizas_2024.shape)

print("\nPrimeras cinco filas:")
print(polizas_2024[columnas_eda].head())

print("\nInformación de las variables seleccionadas:")
polizas_2024[columnas_eda].info()


# Revisar filas completamente duplicadas y valores faltantes por variable
print("\nFilas completamente duplicadas:", polizas_2024.duplicated().sum())

print("\nValores faltantes por variable:")
print(polizas_2024[columnas_eda].isna().sum())


# Resumir las variables numéricas y contar las categorías y los estados de póliza
print("\nResumen de las variables numéricas:")
print(polizas_2024[numericas_eda].describe().round(2).T)

for columna in categoricas + [objetivo]:
    print(f"\nDistribución de {columna}:")
    print(polizas_2024[columna].value_counts(dropna=False))


# Comparar la distribución del premio entre pólizas activas y canceladas
fig, ax = plt.subplots(figsize=(8, 4))

sns.boxplot(
    data=polizas_2024,
    x=objetivo,
    y="total_premium",
    order=["A", "C"],
    showfliers=True,
    fliersize=2,
    ax=ax,
)

ax.set_xticks([0, 1], labels=["Activa", "Cancelada"])
ax.set_title("Distribución del premio por estado de la póliza · 2024")
ax.set_xlabel("Estado de la póliza")
ax.set_ylabel("Importe del premio")
fig.tight_layout()


# Calcular el porcentaje de bajas dentro de cada frecuencia de pago
resumen_pago = polizas_2024.groupby("payment_frequency")[objetivo].agg(
    polizas="count",
    canceladas=lambda estados: estados.eq("C").sum(),
)

resumen_pago["porcentaje_bajas"] = (
    100 * resumen_pago["canceladas"] / resumen_pago["polizas"]
)

print("\nCancelación por frecuencia de pago:")
print(resumen_pago.round(2))


# Graficar las tasas de cancelación usando los nombres de las frencuencias de pago
tasas = resumen_pago["porcentaje_bajas"].rename(
    index={"A": "Anual", "S": "Semestral", "Q": "Trimestral"}
)

fig, ax = plt.subplots(figsize=(8, 4))
tasas.sort_values().plot.bar(ax=ax, rot=0)

ax.set_title("Cancelación por frecuencia de pago · 2024")
ax.set_xlabel("Frecuencia de pago")
ax.set_ylabel("Pólizas canceladas (%)")
fig.tight_layout()


# 2. Seleccionar las variables del modelo y codificar el estado de la póliza
numericas_modelo = ["driver_age", "vehicle_age", "vehicle_value"]
columnas_modelo = numericas_modelo + categoricas

X = polizas_2024[columnas_modelo]
y = polizas_2024[objetivo].map({"A": 0, "C": 1})

if y.isna().any():
    raise ValueError("Cada póliza debe tener estado A o C.")


# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print(f"\nPólizas de entrenamiento: {len(X_train)}")
print(f"Pólizas de prueba: {len(X_test)}")


# Completar los faltantes de cada variable numérica con su mediana y escalarla
pipeline_numerico = Pipeline(steps=[
    ("imputar", SimpleImputer(strategy="median")),
    ("escalar", StandardScaler()),
])


# Completar los faltantes categóricos con la moda y codificar las categorías
pipeline_categorico = Pipeline(steps=[
    ("imputar", SimpleImputer(strategy="most_frequent")),
    ("codificar", OneHotEncoder(handle_unknown="ignore")),
])


# Asignar a las columnas numéricas y categóricas sus respectivas transformaciones
preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", pipeline_numerico, numericas_modelo),
        ("categoricas", pipeline_categorico, categoricas),
    ],
    sparse_threshold=1.0,
)


# 3. Integrar el preprocesamiento y la regresión logística en un único pipeline
pipeline = Pipeline(steps=[
    ("preprocesar", preprocesador),
    ("clasificar", LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )),
])


# 4. Entrenar el pipeline completo y predecir los estados del conjunto de prueba
pipeline.fit(X_train, y_train)
predicciones = pipeline.predict(X_test)


# Evaluar las predicciones mediante el reporte y la matriz de confusión
print("\nReporte de clasificación:")
print(classification_report(
    y_test,
    predicciones,
    labels=[0, 1],
    target_names=["Activa", "Cancelada"],
    zero_division=0,
))

print("Matriz de confusión (filas: estado real, columnas: predicción)")
print(confusion_matrix(y_test, predicciones, labels=[0, 1]))


# 5. Guardar el pipeline entrenado y recuperarlo para clasificar nuevas pólizas
ruta_modelo = base / "modelos" / "pipeline_bajas.joblib"
ruta_modelo.parent.mkdir(exist_ok=True)

joblib.dump(pipeline, ruta_modelo)
modelo_guardado = joblib.load(ruta_modelo)


# Consultar las medianas que el imputador aprendió durante el entrenamiento
rama_numerica = modelo_guardado.named_steps["preprocesar"].named_transformers_["numericas"]
imputador_numerico = rama_numerica.named_steps["imputar"]

medianas_entrenamiento = pd.Series(
    imputador_numerico.statistics_,
    index=numericas_modelo,
)

print("\nMedianas aprendidas para completar faltantes:")
print(medianas_entrenamiento.round(2))


# Usar una póliza del conjunto de prueba para verificar el modelo
poliza_ejemplo = X_test.iloc[[0]]
prediccion = modelo_guardado.predict(poliza_ejemplo)[0]

# Comparar el estado predicho con el estado registrado en el dataset
estados = {0: "Activa", 1: "Cancelada"}
print("\nEstado predicho:", estados[prediccion])
print("Estado real:", estados[y_test.iloc[0]])


# Mostrar los gráficos al finalizar la ejecución
plt.show()