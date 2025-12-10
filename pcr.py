import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

# --- config ---
file_path = 'single_quantify.csv'
ion_cols = ['Pb', 'Fe', 'K']
sensor_cols = ['PB', 'PB@Ni(OH)2', 'PB@Ni-MOF']
results_pcr = {}
all_actual_conc = []
all_pred_conc = []
all_ion_labels = []
CUSTOM_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"-error file not found: {file_path}")
    exit()

plt.style.use('seaborn-v0_8-whitegrid')

CONFIDENCE_LEVEL_CHISQ = 5.991 
COLORS = cm.get_cmap('viridis', len(ion_cols))

# --- data division and model fitting ---
for ion in ion_cols:
    df_ion = df[df[ion] > 0].copy()
    
    X_raw = df_ion[sensor_cols]
    y = df_ion[ion]
    
    if len(df_ion) <= len(sensor_cols):
        print(f"{ion} data is not enough for model fitting")
        continue


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca_plot = PCA(n_components=3) 
    X_pca_all = pca_plot.fit_transform(X_scaled)
    
    pca_fit = PCA(n_components=0.95)
    X_pca_fit = pca_fit.fit_transform(X_scaled)
    n_components = X_pca_fit.shape[1]

# --- PCA visualization ---
    concentrations = y.unique()
    plt.figure(figsize=(9, 7))
    pc_data = pd.DataFrame(X_pca_all, columns=['PC1', 'PC2', 'PC3'])
    pc_data['Concentration'] = y.values
    for i, conc in enumerate(concentrations):
        group = pc_data[pc_data['Concentration'] == conc]
        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        plt.scatter(group['PC1'], group['PC2'], label=f'{conc:.2f}', alpha=0.8, marker=f'${i+1}$', color=color)
        if len(group) > 2:
            center_x = group['PC1'].mean()
            center_y = group['PC2'].mean()
            cov = np.cov(group[['PC1', 'PC2']].values.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            width = 2 * np.sqrt(CONFIDENCE_LEVEL_CHISQ * eigenvalues[0])
            height = 2 * np.sqrt(CONFIDENCE_LEVEL_CHISQ * eigenvalues[1])
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            ellipse = Ellipse((center_x, center_y),
                              width=width,
                              height=height,
                              angle=angle,
                              fill=True,
                              facecolor=color,
                              alpha=0.2,
                              edgecolor=color,
                              linewidth=2,
                              linestyle='-'
                              )
            plt.gca().add_patch(ellipse)
    explained_var = pca_plot.explained_variance_ratio_
    plt.xlabel(f'PC 1 ({explained_var[0]*100:.1f}%)', fontsize=24)
    plt.ylabel(f'PC 2 ({explained_var[1]*100:.1f}%)', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18, length=6, color='black')
    plt.gca().set_aspect('equal', adjustable='box') 
    score_plot_filename = f'./data/PCA_{ion}.png'
    plt.savefig(score_plot_filename)
    plt.close()
    print(f"figure saved: {score_plot_filename} {ion}")

    # --- PCR fitting result calculation ---
    model_pcr = LinearRegression()
    model_pcr.fit(X_pca_fit, y)
    y_pred_pcr = model_pcr.predict(X_pca_fit)
    
    r_squared_pcr = r2_score(y, y_pred_pcr)
    
    residuals_pcr = y - y_pred_pcr
    S_cr_pcr = np.std(residuals_pcr, ddof=n_components + 1)
    LOD_pcr = 3 * S_cr_pcr
    
    results_pcr[ion] = {
        'R_squared': r_squared_pcr,
        'LOD_3_times_S_cr': LOD_pcr,
        'Num_Components': n_components,
    }
    all_actual_conc.extend(y.tolist())
    all_pred_conc.extend(y_pred_pcr.tolist())
    all_ion_labels.extend([ion] * len(y))
    plt.figure(figsize=(9,7))
    plt.scatter(y, y_pred_pcr, alpha=0.7)
    
    max_val = max(y.max(), y_pred_pcr.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal Fit (y=x)')
    
    plt.xlabel(f'{ion} Actual',fontsize=24)
    plt.ylabel(f'{ion} Predicted',fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18, length=6, color='black')
    single_calib_filename = f'./data/PCR_Calibration_Curve_{ion}.png'
    plt.savefig(single_calib_filename)
    plt.close()
    print(f"fitting cureve saved: {ion} {single_calib_filename}")