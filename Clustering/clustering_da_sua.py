# =========================================
# 1. IMPORT & SETUP
# =========================================
import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.neighbors import NearestNeighbors

# Cấu hình vẽ đồ thị cho Paper
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# =========================================
# 2. LOAD DATA
# =========================================
print("--- 1. Loading Data ---")
try:
    df = pd.read_csv("E:/DE_TAI_CA_NHAN/thay_tung/cs-training.csv")
    if df.columns[0].startswith('Unnamed'):
        df = df.iloc[:, 1:]
except FileNotFoundError:
    print("Error: File 'credit.csv' not found. Please ensure the file exists.")
    exit()

TARGET_COL = "SeriousDlqin2yrs"
if TARGET_COL in df.columns:
    y_true_full = df[TARGET_COL]
    X_full = df.drop(columns=[TARGET_COL])
else:
    print(f"Error: Target column '{TARGET_COL}' not found.")
    exit()

print(f"Original Shape: {df.shape}")

# =========================================
# 3. FIGURE A: RAW DATA DIAGNOSTICS (EDA)
# =========================================
print("--- 2. Generating EDA Visualizations ---")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 Target Distribution
target_counts = df[TARGET_COL].value_counts()
axes[0].pie(target_counts, labels=['Good (0)', 'Bad/Default (1)'], 
            autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
axes[0].set_title("Target Distribution (Imbalance)")

# 3.2 Missing Values
missing_data = df.isnull().mean() * 100
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
if not missing_data.empty:
    sns.barplot(x=missing_data.values, y=missing_data.index, ax=axes[1], palette="viridis")
    axes[1].set_xlabel("% Missing")
    axes[1].set_title("Missing Values Percentage")
else:
    axes[1].text(0.5, 0.5, 'No Missing Values', ha='center')

# 3.3 Outliers (Boxplot)
sns.boxplot(y=df['MonthlyIncome'].dropna().sample(min(10000, len(df))), ax=axes[2], color='lightgreen')
axes[2].set_yscale('log') 
axes[2].set_title("Outliers in Monthly Income (Log Scale)")

plt.suptitle("Figure A: Raw Data Diagnostics (Before Processing)", fontsize=16)
plt.tight_layout()
plt.savefig("figA_raw_data_diagnostics.png", dpi=150)
plt.close()

# =========================================
# 4. PREPROCESSING
# =========================================
print("--- 3. Preprocessing Data ---")
X_full["MonthlyIncome"] = X_full["MonthlyIncome"].fillna(X_full["MonthlyIncome"].median())
X_full["NumberOfDependents"] = X_full["NumberOfDependents"].fillna(X_full["NumberOfDependents"].median())

lower = X_full.quantile(0.01)
upper = X_full.quantile(0.99)
X_full = X_full.clip(lower=lower, upper=upper, axis=1)

# =========================================
# 5. FIGURE 1: CORRELATION HEATMAP
# =========================================
plt.figure(figsize=(12, 10))
corr_matrix = X_full.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Figure 1: Correlation Heatmap of Original Features")
plt.tight_layout()
plt.savefig("fig1_correlation.png", dpi=150)
plt.close()

# =========================================
# 6. SCALING & PCA
# =========================================
print("--- 4. Scaling & PCA ---")
scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X_full)

pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_pca_full = pca.fit_transform(X_scaled_full)
print(f"PCA components retained: {pca.n_components_} (95% variance)")

# =========================================
# 7. SAMPLING (TỐI ƯU BỘ NHỚ)
# =========================================
SAMPLE_SIZE = 10000
np.random.seed(RANDOM_STATE)
sample_indices = np.random.choice(X_pca_full.shape[0], SAMPLE_SIZE, replace=False)

X_final = X_pca_full[sample_indices]
y_true_final = y_true_full.iloc[sample_indices].reset_index(drop=True)
X_original_sample = X_full.iloc[sample_indices].reset_index(drop=True)
print(f"Sampled Shape for Clustering: {X_final.shape}")

# =========================================
# 8. FIGURE 2 & 3: FINDING BEST K
# =========================================
print("--- 5. Finding Optimal K ---")
k_range = range(2, 8)
inertias, silhouettes, bics, aics = [], [], [], []

for k in k_range:
    # KMeans
    kmeans_tmp = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels_tmp = kmeans_tmp.fit_predict(X_final)
    inertias.append(kmeans_tmp.inertia_)
    silhouettes.append(silhouette_score(X_final, labels_tmp))
    
    # GMM
    gmm_tmp = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
    gmm_tmp.fit(X_final)
    bics.append(gmm_tmp.bic(X_final))
    aics.append(gmm_tmp.aic(X_final))

# Fig 2: KMeans Selection
fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Elbow)', color=color)
ax1.plot(k_range, inertias, marker='o', color=color, linewidth=2, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(k_range, silhouettes, marker='s', linestyle='--', color=color, linewidth=2, label='Silhouette')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Figure 2: KMeans Optimal K Selection (Elbow & Silhouette)")
fig.tight_layout()
plt.savefig("fig2_kmeans_selection.png", dpi=150)
plt.close()

# Fig 3: GMM Selection
plt.figure(figsize=(10, 5))
plt.plot(k_range, bics, marker='o', label='BIC', linewidth=2)
plt.plot(k_range, aics, marker='s', linestyle='--', label='AIC', linewidth=2)
plt.xlabel('Number of Components (k)')
plt.ylabel('Information Criterion Score')
plt.title("Figure 3: GMM Model Selection (BIC & AIC)")
plt.legend()
plt.savefig("fig3_gmm_selection.png", dpi=150)
plt.close()

# =========================================
# 9. FIGURE B: k-NN DISTANCE (DBSCAN EPS)
# =========================================
minPts = 15
neighbors = NearestNeighbors(n_neighbors=minPts)
neighbors_fit = neighbors.fit(X_final)
distances, indices = neighbors_fit.kneighbors(X_final)
distances = np.sort(distances[:, minPts-1], axis=0)

plt.figure(figsize=(10, 5))
plt.plot(distances, color='black', linewidth=1.5)
plt.title(f"Figure B: k-NN Distance Graph (k={minPts}) for DBSCAN")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{minPts}-NN distance")
plt.axhline(y=0.8, color='r', linestyle='--', label='Chosen eps = 0.8')
plt.legend()
plt.tight_layout()
plt.savefig("figB_knn_distance.png", dpi=150)
plt.close()

# =========================================
# 10. RUN FINAL MODELS (K=2)
# =========================================
print("--- 6. Running Final Models (k=2) ---")
final_results = []
model_labels = {}
K_COMPONENTS = 2

def evaluate_model(name, X, labels, y_true, runtime):
    mask = labels != -1
    core_samples = X[mask]
    core_labels = labels[mask]
    n_clusters = len(set(core_labels))
    
    if n_clusters > 1:
        sil = silhouette_score(core_samples, core_labels)
        dbi = davies_bouldin_score(core_samples, core_labels)
        chi = calinski_harabasz_score(core_samples, core_labels)
    else:
        sil, dbi, chi = np.nan, np.nan, np.nan

    ari = adjusted_rand_score(y_true, labels)
    final_results.append({
        "Model": name, "Silhouette": sil, "DBI": dbi, 
        "CHI": chi, "ARI": ari, "Time(s)": round(runtime, 4)
    })

# KMeans
start = time.time()
kmeans = KMeans(n_clusters=K_COMPONENTS, n_init=20, random_state=RANDOM_STATE)
model_labels['KMeans'] = kmeans.fit_predict(X_final)
evaluate_model("KMeans", X_final, model_labels['KMeans'], y_true_final, time.time() - start)

# Hierarchical
start = time.time()
hier = AgglomerativeClustering(n_clusters=K_COMPONENTS)
model_labels['Hierarchical'] = hier.fit_predict(X_final)
evaluate_model("Hierarchical", X_final, model_labels['Hierarchical'], y_true_final, time.time() - start)

# DBSCAN
start = time.time()
dbscan = DBSCAN(eps=0.8, min_samples=15)
model_labels['DBSCAN'] = dbscan.fit_predict(X_final)
evaluate_model("DBSCAN", X_final, model_labels['DBSCAN'], y_true_final, time.time() - start)

# GMM
start = time.time()
gmm = GaussianMixture(n_components=K_COMPONENTS, random_state=RANDOM_STATE)
model_labels['GMM'] = gmm.fit_predict(X_final)
evaluate_model("GMM", X_final, model_labels['GMM'], y_true_final, time.time() - start)

# =========================================
# 11. FIGURE 4: 2D CLUSTER PROJECTION
# =========================================
print("--- 7. Generating Visual Comparisons ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()
plot_models = ['KMeans', 'GMM', 'Hierarchical', 'DBSCAN']

for i, model_name in enumerate(plot_models):
    labels = model_labels[model_name]
    ax = axes[i]
    scatter = ax.scatter(X_final[:, 0], X_final[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
    ax.set_title(f"{model_name} Clustering (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
    ax.add_artist(legend)

plt.suptitle("Figure 4: Visual Comparison of Clustering Results (Sample n=10,000)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("fig4_cluster_comparison.png", dpi=150)
plt.close()

# =========================================
# 12. FIGURE C: MODEL PERFORMANCE BAR CHART
# =========================================
df_metrics = pd.DataFrame(final_results)
x = np.arange(len(df_metrics['Model']))
width = 0.35  

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, df_metrics['Silhouette'], width, label='Silhouette Score', color='#4c72b0')
rects2 = ax.bar(x + width/2, df_metrics['ARI'], width, label='ARI (Ground Truth Match)', color='#55a868')

ax.set_ylabel('Scores')
ax.set_title('Figure C: Model Performance Comparison (Silhouette vs ARI)')
ax.set_xticks(x)
ax.set_xticklabels(df_metrics['Model'])
ax.legend(loc='upper right')
ax.set_ylim(-0.2, 1.0)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, max(0, height)),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.tight_layout()
plt.savefig("figC_model_comparison_bar.png", dpi=150)
plt.close()

# =========================================
# 13. FIGURE 5: CLUSTER PROFILING
# =========================================
print("--- 8. Profiling Clusters ---")
df_profile = X_original_sample.copy()
df_profile['Cluster'] = model_labels['KMeans']

# SỬA LỖI KEYERROR: Gán cột Target vào để tính trung bình vỡ nợ cho từng cụm
df_profile[TARGET_COL] = y_true_final

cluster_profile = df_profile.groupby('Cluster').mean()

mm_scaler = MinMaxScaler()
cluster_profile_scaled = pd.DataFrame(
    mm_scaler.fit_transform(cluster_profile),
    columns=cluster_profile.columns,
    index=cluster_profile.index
)

plt.figure(figsize=(12, 7))
sns.heatmap(cluster_profile_scaled.T, annot=cluster_profile.T, fmt=".2f", cmap="YlGnBu", linewidths=.5)
plt.title("Figure 5: Cluster Profile Heatmap (Original Feature Means by KMeans Cluster)")
plt.xlabel("Cluster (0: Good, 1: Bad)")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("fig5_cluster_profile.png", dpi=150)
plt.close()

# =========================================
# 14. SAVE OUTPUTS TO CSV
# =========================================
print("--- 9. Saving Final CSVs ---")
df_metrics.to_csv("clustering_metrics_comparison.csv", index=False)

output_sample_df = X_original_sample.copy()
output_sample_df[f"{TARGET_COL}_true"] = y_true_final
for name, labels in model_labels.items():
    output_sample_df[f"label_{name.lower()}"] = labels

output_sample_df.to_csv("sampled_data_with_labels.csv", index=False)

print("\nSUCCESS! All 8 images and 2 CSV files have been generated.")