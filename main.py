from xgboost import XGBRegressor
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/train.csv")

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=["object", "string"]).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

df = pd.get_dummies(df)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print("CV Scores:", scores)
print("Average R2:", scores.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

import matplotlib.pyplot as plt

model.fit(X_train, y_train)

importances = model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

top10 = importance_df.head(10)

print(top10)

plt.figure(figsize=(10, 6))

plt.barh(top10["Feature"], top10["Importance"])

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")

plt.gca().invert_yaxis()
plt.savefig("images/feature_importance.png")
plt.show()

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))