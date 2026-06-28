import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("train_data.csv", low_memory=False)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("Shape after removing duplicates:")
print(df.shape)

# Remove unnecessary columns
df.drop(columns=[
    "CANCELLATION_REASON",
    "SYSTEM_DELAY",
    "SECURITY_DELAY",
    "TRAIN_OPERATOR_DELAY",
    "LATE_TRAIN_DELAY",
    "WEATHER_DELAY"
], inplace=True)

print("Shape after removing unnecessary columns:")
print(df.shape)
# Fill missing values

# Fill missing values

for col in df.columns:

    # If column is numeric
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())

    # Otherwise treat it as text
    else:
        df[col] = df[col].fillna("Unknown")

print("Missing values after cleaning:")
print(df.isnull().sum())

# Take only 100000 random rows
df = df.sample(n=100000, random_state=42)

print("Sample Dataset Shape:", df.shape)
# -------------------------------
# Feature Selection
# -------------------------------

features = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "TRAIN_OPERATOR",
    "SOURCE_STATION",
    "DESTINATION_STATION",
    "SCHEDULED_DEPARTURE",
    "ACTUAL_DEPARTURE",
    "DELAY_DEPARTURE",
    "DISTANCE_KM",
    "RUN_TIME"
]

X = df[features]
y = df["DELAY_ARRIVAL"]

print("X Shape:", X.shape)
print("y Shape:", y.shape)

# -------------------------------
# Encoding
# -------------------------------

# Encode all non-numeric columns
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

print("Encoding Completed!")
print(X.dtypes)
# -------------------------------
# Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# -------------------------------
# Train the Model
# -------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# -------------------------------
# Test the Model
# -------------------------------

y_pred = model.predict(X_test)

print("Prediction Completed!")
print(y_pred[:10])

# -------------------------------
# Model Evaluation
# -------------------------------


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R2 Score:", round(r2, 4))


comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nFirst 10 Predictions")
print(comparison.head(10))