import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# 1. Load the two files ("..", the World Bank marker, = missing)
# ---------------------------------------------------------------
folder = r"C:\Users\Asus\Desktop\INT 234 CA2"

part1 = pd.read_csv(
    r"C:\Users\Asus\Desktop\INT 234 CA2\part-1.csv",
    na_values=[".."]
)
part2 = pd.read_csv(
    r"C:\Users\Asus\Desktop\INT 234 CA2\part-2.csv",
    na_values=[".."]
)
print("Part 1:", part1.shape, "| Part 2:", part2.shape)

# ---------------------------------------------------------------
# 2. Keep the seven indicators and combine the files
# ---------------------------------------------------------------
indicators = [
    "ER.H2O.FWTL.K3",        # Freshwater withdrawals
    "SP.DYN.CBRT.IN",        # Birth rate
    "EN.GHG.CO2.MT.CE.AR5",  # CO2 (target)
    "SP.DYN.CDRT.IN",        # Death rate
    "EG.USE.PCAP.KG.OE",     # Energy use per capita
    "NY.GDP.MKTP.CD",        # GDP
    "SP.POP.TOTL",           # Population
]


def clean(df):
    df = df[df["Series Code"].isin(indicators)]
    return df.dropna(subset=["Country Name", "Country Code", "Series Code"])


combined = pd.concat([clean(part1), clean(part2)], ignore_index=True)
combined = combined.drop_duplicates(subset=["Country Code", "Series Code"], keep="last")
print("Combined shape:", combined.shape)   # paper: (1862, 29)

# ---------------------------------------------------------------
# 3. Wide -> long -> pivot (one row per country-year)
# ---------------------------------------------------------------
year_cols = [c for c in combined.columns if "[YR" in c]

long = combined.melt(
    id_vars=["Country Name", "Country Code", "Series Code"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Value",
)
long["Year"] = long["Year"].str[:4].astype(int)
long["Value"] = pd.to_numeric(long["Value"], errors="coerce")

df = long.pivot(
    index=["Country Name", "Country Code", "Year"],
    columns="Series Code",
    values="Value",
).reset_index()
df.columns.name = None

df = df.rename(columns={
    "ER.H2O.FWTL.K3": "Freshwater",
    "SP.DYN.CBRT.IN": "Birth_Rate",
    "EN.GHG.CO2.MT.CE.AR5": "CO2_Emissions",
    "SP.DYN.CDRT.IN": "Death_Rate",
    "EG.USE.PCAP.KG.OE": "Energy_Use",
    "NY.GDP.MKTP.CD": "GDP",
    "SP.POP.TOTL": "Population",
})
num_ind = ["Freshwater", "Birth_Rate", "CO2_Emissions", "Death_Rate", "Energy_Use", "GDP", "Population"]
# Drop entity-years where every indicator is empty
df = df.dropna(subset=num_ind, how="all")
print("Pivoted shape:", df.shape)          # paper: 6,625 rows

# ---------------------------------------------------------------
# 4. Missing values
# ---------------------------------------------------------------
missing = df.isnull().sum()
print(pd.DataFrame({"Missing": missing, "Percent": (missing / len(df) * 100).round(2)}))

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values Distribution")
plt.show()

# Remove rows with no CO2 target
df = df.dropna(subset=["CO2_Emissions"])
print("Labelled rows:", len(df))           # paper: 6,275

# ---------------------------------------------------------------
# 5. Outliers (IQR) - flagged only, not removed
# ---------------------------------------------------------------
for col in ["CO2_Emissions", "GDP", "Population"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"{col}: {n} flagged outliers")

# ---------------------------------------------------------------
# 6. Exploratory plots
# ---------------------------------------------------------------
# Fig 3: average CO2 per year
df.groupby("Year")["CO2_Emissions"].mean().plot(marker="o", figsize=(10, 4))
plt.title("Global Average CO2 Emissions Trend")
plt.ylabel("Average CO2 Emissions")
plt.show()

# Fig 4: top 10 reporting entities in the latest year
latest = df[df["Year"] == df["Year"].max()]
top10 = latest.nlargest(10, "CO2_Emissions")
sns.barplot(data=top10, x="CO2_Emissions", y="Country Name")
plt.title("Top 10 CO2 Emitting Reporting Entities")
plt.show()

# Fig 5-8: scatter plots
for col in ["GDP", "Energy_Use", "Population", "Freshwater"]:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=col, y="CO2_Emissions", alpha=0.5)
    plt.title(f"{col} vs CO2 Emissions")
    plt.show()

# Fig 9: correlation heatmap
num_cols = ["Freshwater", "Birth_Rate", "CO2_Emissions", "Death_Rate",
            "Energy_Use", "GDP", "Population"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Between Environmental and Economic Indicators")
plt.show()

# Fig 10: year-wise box plot
plt.figure(figsize=(12, 4))
sns.boxplot(data=df, x="Year", y="CO2_Emissions")
plt.xticks(rotation=90)
plt.title("CO2 Emissions Distribution Across Years")
plt.show()

# ---------------------------------------------------------------
# 7. Train / test split (80:20, random_state=42)
# ---------------------------------------------------------------
features = ["GDP", "Population", "Freshwater", "Energy_Use",
            "Birth_Rate", "Death_Rate", "Year"]

X = df[features]
y = df["CO2_Emissions"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train:", X_train.shape, "| Test:", X_test.shape)  # 5020 / 1255

# Median imputation using TRAINING medians only
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# Scaling for KNN and SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 8. Train five models (default settings)
# ---------------------------------------------------------------
models = {
    "Linear Regression": (LinearRegression(), False),
    "Decision Tree": (DecisionTreeRegressor(random_state=42), False),
    "Random Forest": (RandomForestRegressor(random_state=42), False),
    "KNN": (KNeighborsRegressor(), True),
    "SVR": (SVR(), True),
}

results = []
predictions = {}

for name, (model, use_scaled) in models.items():
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled if use_scaled else X_test

    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    predictions[name] = pred

    results.append({
        "Model": name,
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    })

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print(results_df.round(4).to_string(index=False))

# ---------------------------------------------------------------
# 9. Result plots
# ---------------------------------------------------------------
sns.barplot(data=results_df, x="Model", y="R2")
plt.title("R2 Score Comparison of Five ML Models")
plt.xticks(rotation=20)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=results_df, x="Model", y="MAE", ax=ax[0])
ax[0].set_title("MAE Comparison")
sns.barplot(data=results_df, x="Model", y="RMSE", ax=ax[1])
ax[1].set_title("RMSE Comparison")
for a in ax:
    a.tick_params(axis="x", rotation=20)
plt.show()

# Actual vs predicted
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for a, (name, pred) in zip(axes.flat, predictions.items()):
    a.scatter(y_test, pred, alpha=0.5)
    a.set_title(name)
    a.set_xlabel("Actual CO2 Emissions")
    a.set_ylabel("Predicted CO2 Emissions")
axes.flat[-1].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 10. Random Forest feature importance
# ---------------------------------------------------------------
rf = models["Random Forest"][0]
importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(importance.round(4))

sns.barplot(x=importance.values, y=importance.index)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.show()
