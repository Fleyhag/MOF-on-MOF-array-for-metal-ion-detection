import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

# --- 1. load data ---
df = pd.read_csv(r"all_data.csv")
X_cols = ['PB','PB@Ni(OH)2','PB@Ni-MOF']
ions = ['Pb','Cd','Cu','Fe','K']


# --- 2. simple regression ---
X = df[['PB','PB@Ni(OH)2','PB@Ni-MOF']]
ions = ['Pb', 'Cd', 'Cu', 'Fe', 'K']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
models_to_test = {
    'KNN': KNeighborsRegressor(),
    'MLP': MLPRegressor(max_iter=2000, random_state=42),
    'SVR': SVR(),
}
print("--- simple regression result ---")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for ion in ions:
    y = df[ion]
    for name, model in models_to_test.items():
        r2_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        print(f"{ion:2s} - {name:4s} R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")


# --- 3. polynomial regression ---


poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
model = XGBRegressor(random_state=42)
print("\n--- polynomial regression result (XGBoost) ---")
for ion in ions:
    y = df[ion]
    r2_scores = cross_val_score(model, X_poly, y, cv=cv, scoring='r2')
    print(f"{ion:2s} R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

# --- 4. multi-output RF ---


X = df[['PB','PB@Ni(OH)2','PB@Ni-MOF']]
y = df[['Pb','Cd','Cu','Fe','K']]

model = MultiOutputRegressor(RandomForestRegressor())

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
print(f"multi-output R²: {scores.mean():.3f} ± {scores.std():.3f}")

# --- 5. multi-output MLP ---
Y_all = df[ions]
X_input = X_poly
multi_mlp = MLPRegressor(hidden_layer_sizes=(100, 50, 25), alpha=0.001, max_iter=3000, random_state=42)
r2_scores = cross_val_score(multi_mlp, X_input, Y_all, cv=cv, scoring='r2')
print(f"\n---multi-output MLP avg R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

# --- 6. PLSR ---
pls = PLSRegression(n_components=3)
r2_scores = cross_val_score(pls, X_scaled, Y_all, cv=cv, scoring='r2')
print(f"\n--- PLSR (3 components) 平均 R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
