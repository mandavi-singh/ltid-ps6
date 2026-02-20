"""
Agriculture Dataset Analysis Script
=================================

This script performs exploratory data analysis and visualization on the
agriculture dataset (`agriculture_dataset.csv`).  It loads the data,
generates descriptive statistics, plots distributions of numeric variables,
analyzes yield across categorical variables (crop type, irrigation type, soil
type, and season), examines relationships between yield and resource inputs
(fertilizer, pesticide, and water usage), and computes correlations between
numeric features.  All plots are saved into an `agri_plots` directory in the
current working directory.

Steps performed by this script:

1. **Import libraries** – pandas for data manipulation; matplotlib and
   seaborn for plotting.
2. **Load the dataset** – reads the CSV into a pandas DataFrame.
3. **Data overview** – prints the first few rows and summary statistics.
4. **Create output directory** – ensures a directory exists for saving plots.
5. **Visualize numeric distributions** – draws histograms for each numeric
   feature (farm area, fertilizer used, pesticide used, yield, and water
   usage).
6. **Average yield by crop type** – computes and plots the mean yield for
   each crop type using a horizontal bar chart.
7. **Yield distribution by irrigation type** – displays a box plot showing
   yield variability across different irrigation methods.
8. **Average yield by season** – plots the mean yield for each growing
   season.
9. **Relationships between yield and inputs** – creates scatter plots with
   regression lines for yield against fertilizer, pesticide, and water
   usage.
10. **Correlation analysis** – computes the correlation matrix for numeric
    features and visualizes it with a heatmap; prints correlations of
    yield with other numeric variables.

To run this script, ensure that the required packages (`pandas`,
`matplotlib`, and `seaborn`) are installed in your Python environment.

"""

import os
import pandas as pd
import matplotlib

# Use a non-interactive backend for Matplotlib so plots can be saved
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def main() -> None:
    """Run the exploratory data analysis on the agriculture dataset."""
    # ------------------------------------------------------------------
    # 1. Load the data
    # ------------------------------------------------------------------
    data_path = os.path.join(os.path.dirname(__file__), "agriculture_dataset.csv")
    df = pd.read_csv(data_path)

    # ------------------------------------------------------------------
    # 2. Inspect the data
    # ------------------------------------------------------------------
    print("First five rows of the dataset:\n", df.head(), "\n")
    print("Data types:\n", df.dtypes, "\n")
    print("Summary statistics (including categorical variables):\n",
          df.describe(include='all'))

    # ------------------------------------------------------------------
    # 3. Prepare the output directory for plots
    # ------------------------------------------------------------------
    output_dir = os.path.join(os.path.dirname(__file__), "agri_plots")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Plot distributions of numeric features
    # ------------------------------------------------------------------
    numeric_columns = df.select_dtypes(include='float64').columns
    for col in numeric_columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col], bins=10, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_hist.png'))
        plt.close()

    # ------------------------------------------------------------------
    # 5. Yield by crop type
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    yield_by_crop = df.groupby('Crop_Type')['Yield(tons)'].mean().sort_values()
    yield_by_crop.plot(kind='barh', color='skyblue', edgecolor='black')
    plt.title('Average Yield by Crop Type')
    plt.xlabel('Yield (tons)')
    plt.ylabel('Crop Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yield_by_crop.png'))
    plt.close()

    # ------------------------------------------------------------------
    # 6. Yield distribution by irrigation type
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='Irrigation_Type', y='Yield(tons)')
    plt.title('Yield Distribution by Irrigation Type')
    plt.xlabel('Irrigation Type')
    plt.ylabel('Yield (tons)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yield_by_irrigation.png'))
    plt.close()

    # ------------------------------------------------------------------
    # 7. Average yield by season
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    yield_by_season = df.groupby('Season')['Yield(tons)'].mean().sort_values()
    yield_by_season.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Average Yield by Season')
    plt.xlabel('Season')
    plt.ylabel('Yield (tons)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yield_by_season.png'))
    plt.show()
    plt.close()
    

    # ------------------------------------------------------------------
    # 8. Relationships between yield and resource inputs
    # ------------------------------------------------------------------
    # Fertilizer vs. Yield
    plt.figure(figsize=(6, 4))
    sns.regplot(data=df, x='Fertilizer_Used(tons)', y='Yield(tons)',
                scatter_kws={'alpha': 0.7})
    plt.title('Fertilizer Used vs Yield')
    plt.xlabel('Fertilizer Used (tons)')
    plt.ylabel('Yield (tons)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fertilizer_vs_yield.png'))
    plt.close()

    # Pesticide vs. Yield
    plt.figure(figsize=(6, 4))
    sns.regplot(data=df, x='Pesticide_Used(kg)', y='Yield(tons)', color='green',
                scatter_kws={'alpha': 0.7})
    plt.title('Pesticide Used vs Yield')
    plt.xlabel('Pesticide Used (kg)')
    plt.ylabel('Yield (tons)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pesticide_vs_yield.png'))
    plt.close()

    # Water Usage vs. Yield
    plt.figure(figsize=(6, 4))
    sns.regplot(data=df, x='Water_Usage(cubic meters)', y='Yield(tons)',
                color='purple', scatter_kws={'alpha': 0.7})
    plt.title('Water Usage vs Yield')
    plt.xlabel('Water Usage (cubic meters)')
    plt.ylabel('Yield (tons)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'water_vs_yield.png'))
    plt.close()

    # ------------------------------------------------------------------
    # 9. Correlation matrix
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include='float64')
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix (Numeric Variables)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # Print correlation of yield with other numeric features to console
    yield_correlations = corr_matrix['Yield(tons)'].drop('Yield(tons)')
    print("\nCorrelation of Yield with numeric features:\n", yield_correlations)

if __name__ == '__main__':
    main()
