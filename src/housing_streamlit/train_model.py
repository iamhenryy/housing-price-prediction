import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from models_algorithms import K_means, KNN_regressor

# =========================
# Load data
# =========================
data = pd.read_csv("data/real_estate.csv")

data.columns = [
    'No', 'transaction_date', 'house_age', 'distance_to_MRT',
    'convenience_stores', 'latitude', 'longitude', 'house_price'
]
data.drop(['No', 'transaction_date'], axis=1, inplace=True)

# =========================
# Log transform
# =========================
data['distance_to_MRT'] = np.log1p(data['distance_to_MRT'])
data['house_price'] = np.log1p(data['house_price'])

# =========================
# Remove outliers (IQR)
# =========================
Q1 = data['house_price'].quantile(0.25)
Q3 = data['house_price'].quantile(0.75)
IQR = Q3 - Q1
mask = (
    (data['house_price'] >= Q1 - 1.5 * IQR) &
    (data['house_price'] <= Q3 + 1.5 * IQR)
)
data = data[mask]

# =========================
# Split
# =========================
X = data.drop(columns=['house_price'])
y = data['house_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Min–Max scaling (SAVE SCALER)
# =========================
scaler = {}
for col in X_train.columns:
    mn, mx = X_train[col].min(), X_train[col].max()
    scaler[col] = (mn, mx)
    X_train[col] = (X_train[col] - mn) / (mx - mn)
    X_test[col] = (X_test[col] - mn) / (mx - mn)

# =========================
# Train K-Means
# =========================
best_k = 3
kmeans = K_means(best_k)
kmeans.fit(X_train)
kmeans.fit_loop()

train_labels = kmeans.assign_labels(X_train)

# =========================
# Train KNN per cluster
# =========================
best_k_per_cluster = {
    0: 6,
    1: 8,
    2: 10
}
knn_models = []

for c in range(best_k):
    mask = train_labels == c
    X_c, y_c = X_train[mask], y_train[mask]

    k_knn = best_k_per_cluster[c]
    knn = KNN_regressor(k=k_knn)
    knn.fit(X_c, y_c)

    knn_models.append(knn)

# =========================
# Save models
# =========================
os.makedirs("models", exist_ok=True)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("models/knn_models.pkl", "wb") as f:
    pickle.dump(knn_models, f)

print("Training completed & models saved")
