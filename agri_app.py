"""
Streamlit App with Analysis and Prediction
-----------------------------------------

This Streamlit application combines exploratory data analysis with a
machine learning regression model to predict crop yield. The model is
trained dynamically inside the app to ensure compatibility with
Streamlit Cloud environments.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "agriculture_dataset.csv")


# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# -----------------------------
# Model + Preprocessor (TRAIN INSIDE APP)
# -----------------------------
@st.cache_resource
def load_model_and_preprocessor():
    df = load_data()

    X = df.drop("Yield(tons)", axis=1)
    y = df["Yield(tons)"]

    categorical_features = [
        "Crop_Type",
        "Irrigation_Type",
        "Soil_Type",
        "Season",
    ]

    numerical_features = [
        "Farm_Area(acres)",
        "Fertilizer_Used(tons)",
        "Pesticide_Used(kg)",
        "Water_Usage(cubic meters)",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    return pipeline


# -----------------------------
# Plotting Functions
# -----------------------------
def make_plots(df):
    plots = {}

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    avg_by_crop = df.groupby("Crop_Type")["Yield(tons)"].mean().sort_values()
    ax1.bar(avg_by_crop.index, avg_by_crop.values)
    ax1.set_title("Average Yield by Crop Type")
    ax1.tick_params(axis="x", rotation=45)
    plots["avg_by_crop"] = fig1

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    irrigation_types = df["Irrigation_Type"].unique()
    data = [df[df["Irrigation_Type"] == ir]["Yield(tons)"] for ir in irrigation_types]
    ax2.boxplot(data, labels=irrigation_types)
    ax2.set_title("Yield Distribution by Irrigation Type")
    plots["yield_by_irrigation"] = fig2

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    avg_by_season = df.groupby("Season")["Yield(tons)"].mean().sort_values()
    ax3.bar(avg_by_season.index, avg_by_season.values)
    ax3.set_title("Average Yield by Season")
    plots["avg_by_season"] = fig3

    return plots


# -----------------------------
# Prediction Function
# -----------------------------
def predict_yield(pipeline, inputs: dict) -> float:
    input_df = pd.DataFrame(inputs)
    prediction = pipeline.predict(input_df)
    return float(prediction[0])


# -----------------------------
# Recommendation Logic
# -----------------------------
def get_recommendations(selected_crop, selected_irrigation, selected_season, df):
    recommendations = []

    if selected_crop != df.groupby("Crop_Type")["Yield(tons)"].mean().idxmax():
        recommendations.append("Consider high-yield crop varieties.")

    if selected_irrigation != df.groupby("Irrigation_Type")["Yield(tons)"].mean().idxmax():
        recommendations.append("Efficient irrigation methods may improve yield.")

    if selected_season != df.groupby("Season")["Yield(tons)"].mean().idxmax():
        recommendations.append("Season selection impacts yield significantly.")

    recommendations.append("Maintain balanced fertilization.")
    recommendations.append("Adopt efficient water management practices.")

    return recommendations


# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(page_title="Agriculture Yield Analysis & Prediction", layout="wide")
    st.title("🌾 Agriculture Yield Analysis & Prediction")

    df = load_data()
    pipeline = load_model_and_preprocessor()

    page = st.sidebar.selectbox(
        "Select Section", ["Exploratory Analysis", "Yield Prediction"]
    )

    if page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        plots = make_plots(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(plots["avg_by_crop"])
        with col2:
            st.pyplot(plots["yield_by_irrigation"])
        with col3:
            st.pyplot(plots["avg_by_season"])

        st.dataframe(df.head(10))

    else:
        st.header("Yield Prediction")

        selected_crop = st.sidebar.selectbox("Crop Type", sorted(df["Crop_Type"].unique()))
        selected_irrigation = st.sidebar.selectbox(
            "Irrigation Type", sorted(df["Irrigation_Type"].unique())
        )
        selected_soil = st.sidebar.selectbox("Soil Type", sorted(df["Soil_Type"].unique()))
        selected_season = st.sidebar.selectbox("Season", sorted(df["Season"].unique()))

        farm_area = st.sidebar.number_input("Farm Area (acres)", 0.1, 1000.0, 10.0)
        fertilizer = st.sidebar.number_input("Fertilizer Used (tons)", 0.0, 100.0, 10.0)
        pesticide = st.sidebar.number_input("Pesticide Used (kg)", 0.0, 500.0, 5.0)
        water_usage = st.sidebar.number_input(
            "Water Usage (cubic meters)", 0.0, 200000.0, 50000.0
        )

        if st.sidebar.button("Predict"):
            inputs = {
                "Crop_Type": [selected_crop],
                "Irrigation_Type": [selected_irrigation],
                "Soil_Type": [selected_soil],
                "Season": [selected_season],
                "Farm_Area(acres)": [farm_area],
                "Fertilizer_Used(tons)": [fertilizer],
                "Pesticide_Used(kg)": [pesticide],
                "Water_Usage(cubic meters)": [water_usage],
            }

            predicted_yield = predict_yield(pipeline, inputs)
            st.success(f"🌱 Predicted Yield: **{predicted_yield:.2f} tons**")

            avg_crop = df[df["Crop_Type"] == selected_crop]["Yield(tons)"].mean()
            avg_irrig = df[df["Irrigation_Type"] == selected_irrigation]["Yield(tons)"].mean()
            avg_season = df[df["Season"] == selected_season]["Yield(tons)"].mean()

            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ["Predicted", "Crop Avg", "Irrigation Avg", "Season Avg"]
            values = [predicted_yield, avg_crop, avg_irrig, avg_season]
            bar_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
            ax.bar(labels, values, color=bar_colors)
            ax.set_ylabel("Yield (tons)")
            ax.set_title("Predicted Yield vs Dataset Averages")
            
            st.pyplot(fig)
            # Generate and display recommendations recs = get_recommendations( selected_crop, selected_irrigation, selected_soil, selected_season, predicted_yield, df, )

            st.subheader("Recommendations")
            for rec in get_recommendations(
                selected_crop, selected_irrigation, selected_season, df
            ):
                st.markdown(f"- {rec}")
                


if __name__ == "__main__":
    main()

