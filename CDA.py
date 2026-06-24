import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
file_path = r"D:\GCU\SEM - 3\PP - Problem Solving with Python\PyCharm\customers-100\customers-100.csv"

df = pd.read_csv(file_path)

print("===== DATASET OVERVIEW =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Data Cleaning
df = df.drop_duplicates()

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].mean())

print("\nDataset cleaned successfully!")

# Statistical Summary
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# Customer Country Distribution
if 'Country' in df.columns:
    country_counts = df['Country'].value_counts()

    plt.figure(figsize=(10,5))
    country_counts.head(10).plot(kind='bar')
    plt.title("Top 10 Countries")
    plt.xlabel("Country")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.show()

# Customer Subscription Distribution
if 'Subscription Date' in df.columns:
    df['Subscription Date'] = pd.to_datetime(
        df['Subscription Date'],
        errors='coerce'
    )

    monthly_subs = (
        df['Subscription Date']
        .dt.month
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8,5))
    monthly_subs.plot(marker='o')
    plt.title("Monthly Customer Registrations")
    plt.xlabel("Month")
    plt.ylabel("Registrations")
    plt.grid(True)
    plt.show()

# Correlation Analysis
numeric_df = df.select_dtypes(include=['int64','float64'])

if not numeric_df.empty:
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(8,6))
    plt.imshow(correlation_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90
    )
    plt.yticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

print("\nAnalysis Completed Successfully!")