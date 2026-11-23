# Pre-Entrega 9 - Pipeline de Preprocesamiento

Pipeline de preprocesamiento y clasificación supervisada con **Scikit-Learn** para predecir la cancelación de pólizas de vehículos (`policy_status`), clasificándolas como activas o canceladas. Incluye imputación de valores faltantes, transformación de variables numéricas y categóricas, regresión logística y persistencia, garantizando consistencia y evitando la fuga de datos (*data leakage*).

## Descripción del Dataset

El archivo fuente se encuentra en `data/raw/polizas.csv` y contiene el registro histórico de la cartera de pólizas. Para este análisis y modelado se trabaja específicamente con la **cartera del año 2024**, cuyo filtro se aplica dentro del script.

Se utiliza el dataset *A detailed dataset of motor insurance policies with coverage-specific financial information*, de Priscila Espinosa, Josep Lledó y David Atance. La fuente completa contiene **354.140 registros** del período 2022–2024.

* **Fuente original:** Mendeley Data (disponible buscando el título del dataset).
* **Descarga directa:** [Descargar `polizas.csv` desde Google Drive](https://drive.google.com/file/d/1UK__sthQJsfVM-6EMeaNnfXJq701Z9zQ/view?usp=share_link).

### Variable objetivo
* `policy_status`: Estado de la póliza, codificado como:
  * `A` → `0` (Activa)
  * `C` → `1` (Cancelada)

### Variables utilizadas
* **Numéricas (modelo):**
  * `driver_age`: Edad del conductor principal.
  * `vehicle_age`: Antigüedad del vehículo.
  * `vehicle_value`: Valor asegurado del vehículo.
* **Categóricas (modelo):**
  * `policy_type`: Tipo de cobertura.
  * `business_type`: Nueva contratación o cartera existente.
  * `payment_frequency`: Frecuencia de pago (`A`: Anual, `S`: Semestral, `Q`: Trimestral).
  * `bonus_score`: Clasificación según historial de siniestros.
  * `circulation_area`: Entorno de circulación del vehículo.
* **Variables solo de exploración (EDA):** `total_premium` (importe del premio) y `total_claims` (siniestros declarados) se analizan en la etapa exploratoria, pero se excluyen del modelo predictivo para evitar *data leakage* o sesgos asociados al cierre del período, ya que habría que contemplar variables adicionales como `total_exposure` para calcular la exposición al riesgo del período y el premio promedio.

---

## Arquitectura del Pipeline

Toda la lógica de limpieza, transformación y modelado se encuentra en el archivo `pipeline.py`, encapsulada en un único objeto `Pipeline`:

1. **División previa de datos:** Se ejecuta `train_test_split` (80% entrenamiento / 20% prueba) con estratificación (`stratify=y`) antes de ajustar cualquier parámetro de preprocesamiento.
2. **Preprocesador (`ColumnTransformer`):**
   * **Rama numérica:** `SimpleImputer(strategy="median")` seguido de `StandardScaler()`.
   * **Rama categórica:** `SimpleImputer(strategy="most_frequent")` seguido de `OneHotEncoder(handle_unknown="ignore")` para tolerar categorías no vistas durante el entrenamiento.
3. **Estimador:** `LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)` para contemplar el posible desbalance entre pólizas activas y canceladas.
4. **Evaluación:** Generación de `classification_report` y matriz de confusión sobre el conjunto de prueba.
5. **Persistencia:** Exportación del pipeline entrenado con `joblib` en `modelos/pipeline_bajas.joblib` y prueba de recarga + inferencia (`.predict()`) sobre una póliza de ejemplo.

---

## Estructura del Repositorio

```text
.
├── data/
│   └── raw/
│       └── polizas.csv         # Descargar desde el enlace de Google Drive
├── modelos/
│   └── pipeline_bajas.joblib   # Generado automáticamente al ejecutar el script
├── pipeline.py                 # Script principal (EDA, entrenamiento, evaluación y exportación)
├── requirements.txt            # Dependencias del proyecto
└── README.md


   ```

---

## Instalación y Ejecución

1. Clonar el repositorio.
2. Descargar `polizas.csv` desde el enlace de Google Drive y guardarlo en `data/raw/polizas.csv`.
3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
4. Ejecutar el pipeline:
   ```bash
   python pipeline.py
   ```
