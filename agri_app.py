"""
Streamlit App with Analysis and Prediction
-----------------------------------------

This Streamlit application combines exploratory data analysis with a
regression model to predict crop yield.  It reads the Agriculture and
Farming dataset, displays summary charts (average yields by crop and
season and yield distribution by irrigation type), and allows the user
to input farm characteristics to obtain a yield prediction from a
pretrained model.

Prerequisites:
    pip install streamlit pandas numpy matplotlib joblib

Run the application with:

    streamlit run agri_app_graphs.py

Ensure that the following files are in the same directory:
    - agriculture_dataset.csv  (the dataset)
    - preprocessor.pkl         (saved preprocessing pipeline)
    - best_model.pkl           (saved trained model)

"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# Paths to data and model files
DATA_PATH = os.path.join(os.path.dirname(__file__), 'agriculture_dataset.csv')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pkl')


@st.cache_data
def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the agriculture dataset into a pandas DataFrame."""
    return pd.read_csv(path)


@st.cache_resource
def load_model_and_preprocessor(preprocessor_path=PREPROCESSOR_PATH, model_path=MODEL_PATH):
    """Load the serialized preprocessing pipeline and trained model."""
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model


def make_plots(df: pd.DataFrame):
    """Generate Matplotlib figures for exploratory analysis.

    Returns a dictionary mapping descriptive names to matplotlib Figure objects.
    """
    plots = {}

    # Average yield by crop type
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    avg_by_crop = df.groupby('Crop_Type')['Yield(tons)'].mean().sort_values()
    ax1.bar(avg_by_crop.index, avg_by_crop.values, color='skyblue', edgecolor='black')
    ax1.set_title('Average Yield by Crop Type')
    ax1.set_xlabel('Crop Type')
    ax1.set_ylabel('Yield (tons)')
    ax1.tick_params(axis='x', rotation=45)
    fig1.tight_layout()
    plots['avg_by_crop'] = fig1

    # Yield distribution by irrigation type (boxplot)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    irrigation_types = df['Irrigation_Type'].unique()
    data = [df[df['Irrigation_Type'] == ir]['Yield(tons)'] for ir in irrigation_types]
    ax2.boxplot(data, labels=irrigation_types)
    ax2.set_title('Yield Distribution by Irrigation Type')
    ax2.set_xlabel('Irrigation Type')
    ax2.set_ylabel('Yield (tons)')
    fig2.tight_layout()
    plots['yield_by_irrigation'] = fig2

    # Average yield by season
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    avg_by_season = df.groupby('Season')['Yield(tons)'].mean().sort_values()
    ax3.bar(avg_by_season.index, avg_by_season.values, color='coral', edgecolor='black')
    ax3.set_title('Average Yield by Season')
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Yield (tons)')
    fig3.tight_layout()
    plots['avg_by_season'] = fig3

    return plots


def predict_yield(preprocessor, model, inputs: dict) -> float:
    """Prepare input DataFrame from user inputs, preprocess and predict yield."""
    input_df = pd.DataFrame(inputs)
    X_transformed = preprocessor.transform(input_df)
    prediction = model.predict(X_transformed)
    return float(prediction[0])


def get_recommendations(selected_crop: str, selected_irrigation: str, selected_soil: str,
                        selected_season: str, predicted_yield: float, df: pd.DataFrame) -> list:
    """
    Generate simple recommendations to improve yield based on user selections and
    dataset insights.  This function examines which crop, irrigation method and
    soil type achieve the highest average yields in the dataset and suggests
    potential improvements.  Additional generic agronomic best practices are
    provided regardless of the selection.

    Parameters
    ----------
    selected_crop : str
        The crop type chosen by the user.
    selected_irrigation : str
        The irrigation method chosen by the user.
    selected_soil : str
        The soil type chosen by the user.
    selected_season : str
        The cropping season chosen by the user.
    predicted_yield : float
        The yield predicted by the model.
    df : pandas.DataFrame
        The original dataset used for contextual comparisons.

    Returns
    -------
    list of str
        A list of recommendation strings.
    """
    recommendations = []

    # Compute high‑yielding crop in the dataset
    crop_means = df.groupby('Crop_Type')['Yield(tons)'].mean().sort_values(ascending=False)
    top_crop = crop_means.index[0]
    if selected_crop != top_crop:
        recommendations.append(
            f"Consider cultivating **{top_crop}** if agro‑climatic conditions permit, as it has the highest average "
            "yield in the dataset."
        )

    # Compute highest yielding irrigation method
    irrigation_means = df.groupby('Irrigation_Type')['Yield(tons)'].mean().sort_values(ascending=False)
    top_irrig = irrigation_means.index[0]
    if selected_irrigation != top_irrig:
        recommendations.append(
            f"Switching to **{top_irrig}** irrigation may help improve yields; it shows the highest average yield "
            "among irrigation methods in the dataset."
        )

    # Compute highest yielding season
    season_means = df.groupby('Season')['Yield(tons)'].mean().sort_values(ascending=False)
    top_season = season_means.index[0]
    if selected_season != top_season:
        recommendations.append(
            f"Planting during the **{top_season}** season could lead to higher yields based on dataset averages."
        )

    # Generic agronomic recommendations
    recommendations.append(
        "Maintain balanced fertilization by matching nutrient application to crop needs and avoid overuse of pesticides."
    )
    recommendations.append(
        "Adopt efficient water management practices (e.g., drip or sprinkler systems) to deliver the right amount of water "
        "at the right time."
    )
    recommendations.append(
        "Practice crop rotation and cover cropping to maintain soil fertility, reduce pest pressure and improve soil health."
    )
    recommendations.append(
        "Implement integrated pest management (IPM) techniques to manage pests sustainably and minimise chemical inputs."
    )

    return recommendations


def main():
    st.set_page_config(page_title="Agriculture Analysis & Prediction", layout="wide")
    st.title("Agriculture Yield Analysis & Prediction")
    st.write(
        "This app provides exploratory analysis charts from the Agriculture and Farming dataset "
        "and allows you to predict crop yield based on farm characteristics."
    )

    # Load data and model
    df = load_data()
    preprocessor, model = load_model_and_preprocessor()

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a section", ("Exploratory Analysis", "Yield Prediction")
    )

    if page == "Exploratory Analysis":
        st.header("Exploratory Analysis")
        st.markdown(
            "Below are some visual summaries of the dataset to help you understand "
            "the variation in yields across crops, irrigation methods and seasons."
        )

        # Generate and display plots
        plots = make_plots(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(plots['avg_by_crop'])
        with col2:
            st.pyplot(plots['yield_by_irrigation'])
        with col3:
            st.pyplot(plots['avg_by_season'])

        # Show a preview of the data
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))

    else:  # Yield Prediction
        st.header("Yield Prediction")
        st.markdown(
            "Enter the farm details on the left and click **Predict** to estimate the "
            "crop yield."
        )

        # Input fields in sidebar
        st.sidebar.header("Input Farm Details")

        crop_options = df['Crop_Type'].unique().tolist()
        irrigation_options = df['Irrigation_Type'].unique().tolist()
        soil_options = df['Soil_Type'].unique().tolist()
        season_options = df['Season'].unique().tolist()

        selected_crop = st.sidebar.selectbox("Crop Type", sorted(crop_options))
        selected_irrigation = st.sidebar.selectbox("Irrigation Type", sorted(irrigation_options))
        selected_soil = st.sidebar.selectbox("Soil Type", sorted(soil_options))
        selected_season = st.sidebar.selectbox("Season", sorted(season_options))

        farm_area = st.sidebar.number_input(
            "Farm Area (acres)", min_value=0.1, max_value=1000.0, value=10.0, step=0.1
        )
        fertilizer = st.sidebar.number_input(
            "Fertilizer Used (tons)", min_value=0.0, max_value=100.0, value=10.0, step=0.1
        )
        pesticide = st.sidebar.number_input(
            "Pesticide Used (kg)", min_value=0.0, max_value=500.0, value=5.0, step=0.1
        )
        water_usage = st.sidebar.number_input(
            "Water Usage (cubic meters)", min_value=0.0, max_value=200000.0, value=50000.0,
            step=100.0
        )

        # Predict button
        if st.sidebar.button("Predict"):
            inputs = {
                'Crop_Type': [selected_crop],
                'Irrigation_Type': [selected_irrigation],
                'Soil_Type': [selected_soil],
                'Season': [selected_season],
                'Farm_Area(acres)': [farm_area],
                'Fertilizer_Used(tons)': [fertilizer],
                'Pesticide_Used(kg)': [pesticide],
                'Water_Usage(cubic meters)': [water_usage],
            }
            predicted_yield = predict_yield(preprocessor, model, inputs)
            st.success(f"Predicted Yield: {predicted_yield:.2f} tons")

            # Compute averages for comparison
            avg_crop = df[df['Crop_Type'] == selected_crop]['Yield(tons)'].mean()
            avg_irrig = df[df['Irrigation_Type'] == selected_irrigation]['Yield(tons)'].mean()
            avg_season = df[df['Season'] == selected_season]['Yield(tons)'].mean()

            # Create a comparison chart
            fig_user, ax_user = plt.subplots(figsize=(6, 4))
            labels = ['Your input', f'{selected_crop} avg', f'{selected_irrigation} avg', f'{selected_season} avg']
            values = [predicted_yield, avg_crop, avg_irrig, avg_season]
            bar_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
            ax_user.bar(labels, values, color=bar_colors)
            ax_user.set_ylabel('Yield (tons)')
            ax_user.set_title('Predicted Yield vs. Dataset Averages')
            for idx, val in enumerate(values):
                ax_user.text(idx, val + 0.5, f"{val:.2f}", ha='center')
            fig_user.tight_layout()
            st.pyplot(fig_user)

            # Generate and display recommendations
            recs = get_recommendations(
                selected_crop,
                selected_irrigation,
                selected_soil,
                selected_season,
                predicted_yield,
                df,
            )
            st.subheader("Recommendations to Improve Yield")
            for rec in recs:
                st.markdown(f"- {rec}")

        # Optionally show the dataset summary and charts below the prediction form
        if st.checkbox("Show analysis charts below"):
            st.subheader("Analysis Charts")
            plots = make_plots(df)
            st.pyplot(plots['avg_by_crop'])
            st.pyplot(plots['yield_by_irrigation'])
            st.pyplot(plots['avg_by_season'])


if __name__ == "__main__":
    main()