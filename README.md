# Agriculture Yield Analysis & Prediction

This project provides a complete workflow to analyse an agricultural dataset,
train machine‑learning models to predict crop yield, and offer an
interactive Streamlit application for forecasting and recommendations.

## 📦 Dataset Overview

The **Agriculture and Farming** dataset (sourced from Kaggle) contains
farm‑level information such as crop type, irrigation method, soil type and
season, together with numerical metrics like farm area, fertilizer and
pesticide usage, yield and water usage:contentReference[oaicite:0]{index=0}.  Each row
represents one farm’s performance over a growing season.  The target
variable is `Yield(tons)`, expressed in metric tons per farm.

### Columns

| Column                     | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `Farm_ID`                 | Unique identifier for each farm.                                             |
| `Crop_Type`               | Type of crop grown (e.g., Cotton, Carrot, Sugarcane, Tomato, etc.).           |
| `Farm_Area(acres)`        | Size of the farm in acres.                                                   |
| `Irrigation_Type`         | Irrigation method used (Manual, Drip, Flood, Rainfed, Sprinkler).            |
| `Fertilizer_Used(tons)`   | Amount of fertilizer applied in tons.                                        |
| `Pesticide_Used(kg)`      | Quantity of pesticide applied in kilograms.                                  |
| `Soil_Type`               | Dominant soil type (e.g., Silty, Peaty, Clayey, Sandy, Loamy).               |
| `Season`                  | Cropping season: **Kharif**, **Rabi** or **Zaid**.  Kharif crops are sown
|                           | in July–October and harvested during the monsoon; Rabi crops are sown
|                           | October–November and harvested in late winter; Zaid crops are short‑season
|                           | crops grown in March–June, requiring warm, dry weather:contentReference[oaicite:1]{index=1}. |
| `Water_Usage(cubic meters)`| Total water consumed during the season.                                     |
| `Yield(tons)`             | Total yield harvested in metric tons (target variable).                      |

The dataset was used to explore how crop type, irrigation method, season
and other factors influence yield.  Average yields by crop, irrigation
and season are visualised in the Exploratory Data Analysis section.

## 🧪 Exploratory Data Analysis

Run the script `agriculture_analysis.py` to generate summary statistics
and charts from the dataset.  The script loads the CSV file, cleans and
describes the data, and produces visualisations such as:

- **Average yield by crop type:** identifies high‑yield crops like carrots
  and tomatoes.
- **Yield distribution by irrigation method:** compares manual, drip,
  flood, rainfed and sprinkler irrigation.
- **Average yield by season:** compares Kharif, Rabi and Zaid seasons.
- **Correlation matrix:** shows relationships between numeric variables
  (farm area, fertilizer, pesticide, water and yield).

The figures are saved into the `agri_plots` directory for inclusion in
reports.  You can customise colours, labels or add new plots by editing
the script.

## 🤖 Model Training

The training workflow is implemented in `model_training.py`.  It performs
the following steps:

1. **Data loading & preprocessing:** Categorical variables (crop type,
   irrigation type, soil type, season) are one‑hot encoded; numerical
   variables are passed through unchanged.  This is done via a
   `ColumnTransformer` pipeline.
2. **Splitting:** The dataset is split into training and test sets.
3. **Model training:** Three tree‑based regressors are trained –
   RandomForest, GradientBoosting and XGBRegressor.  Each model is
   evaluated with cross‑validation using root mean squared error (RMSE).
   The best model (based on cross‑validation performance) is selected.
4. **Saving:** The preprocessing pipeline and best model are serialised
   using `joblib` to `preprocessor.pkl` and `best_model.pkl`.
5. **Reporting:** Results (RMSE and R²) are saved to `results.csv`.

A random forest is a meta estimator that fits a number of decision tree
regressors on bootstrapped samples of the data and averages their
predictions to improve accuracy and reduce over‑fitting:contentReference[oaicite:2]{index=2}.  Gradient
boosting and XGBoost build ensembles of weak learners iteratively to
minimise prediction errors.  A simple LSTM training function is also
provided for demonstration (requires TensorFlow).

## 🌱 Streamlit Application

The interactive dashboard is implemented in `agri_app.py`.  After
training a model, launch the app with:

```bash
streamlit run agri_app.py
