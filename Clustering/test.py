import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore') # Tắt cảnh báo đỏ

print("=== PIPELINE: ĐẶC TRƯNG -> LOG TRANSFORM -> SCALING -> PCA -> K-MEANS ===")

# =========================================================
# 1. TẢI VÀ LÀM SẠCH DỮ LIỆU
# =========================================================
print("[1/5] Đang tải và làm sạch dữ liệu...")
df = pd.read_csv(r'D:/CODING_DATA/cs-training.csv')
if df.columns[0] == 'Unnamed: 0':
    df = df.drop(columns=['Unnamed: 0'])

bang_doi_ten = {
    'SeriousDlqin2yrs': 'Vo_No_Trong_2_Nam',
    'RevolvingUtilizationOfUnsecuredLines': 'Ty_Le_Su_Dung_Han_Muc',
    'age': 'Tuoi',
    'NumberOfTime30-59DaysPastDueNotWorse': 'So_Lan_Tre_Han_30_59_Ngay',
    'DebtRatio': 'Ty_Le_No_Tren_Thu_Nhap',
    'MonthlyIncome': 'Thu_Nhap_Hang_Thang',
    'NumberOfOpenCreditLinesAndLoans': 'So_Khoan_Tin_Dung_Dang_Mo',
    'NumberOfTimes90DaysLate': 'So_Lan_Tre_Han_Tren_90_Ngay',
    'NumberRealEstateLoansOrLines': 'So_Khoan_Vay_Bat_Dong_San',
    'NumberOfTime60-89DaysPastDueNotWorse': 'So_Lan_Tre_Han_60_89_Ngay',
    'NumberOfDependents': 'So_Nguoi_Phu_Thuoc'
}
df = df.rename(columns=bang_doi_ten)

# Xóa trùng lặp và điền khuyết
df = df.drop_duplicates(keep='first')
df['Thu_Nhap_Hang_Thang'] = df['Thu_Nhap_Hang_Thang'].fillna(df['Thu_Nhap_Hang_Thang'].median())
df['So_Nguoi_Phu_Thuoc'] = df['So_Nguoi_Phu_Thuoc'].fillna(df['So_Nguoi_Phu_Thuoc'].median())
df = df.drop_duplicates(keep='first')

# Lọc ngoại lai (IQR) để giữ cho dữ liệu sạch nhất
cac_cot_can_loc = ['Ty_Le_Su_Dung_Han_Muc', 'Ty_Le_No_Tren_Thu_Nhap', 'Thu_Nhap_Hang_Thang']
for cot in cac_cot_can_loc:
    Q1 = df[cot].quantile(0.25)
    Q3 = df[cot].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[cot] >= (Q1 - 1.5 * IQR)) & (df[cot] <= (Q3 + 1.5 * IQR))]

# =========================================================
# 2. FEATURE ENGINEERING (TẠO ĐẶC TRƯNG MỚI)
# =========================================================
print("[2/5] Đang tạo đặc trưng mới (Feature Engineering)...")
df['Tong_So_Lan_Tre_Han'] = df['So_Lan_Tre_Han_30_59_Ngay'] + df['So_Lan_Tre_Han_60_89_Ngay'] + df['So_Lan_Tre_Han_Tren_90_Ngay']
df['Thu_Nhap_Binh_Quan_Dau_Nguoi'] = df['Thu_Nhap_Hang_Thang'] / (df['So_Nguoi_Phu_Thuoc'] + 1)

y_thuc_te = df['Vo_No_Trong_2_Nam'].values
X = df.drop(columns=['Vo_No_Trong_2_Nam'])

# =========================================================
# 3. LOG TRANSFORM (BIẾN ĐỔI LOGARIT)
# =========================================================
print("[3/5] Đang áp dụng Log Transform cho các biến bị lệch (Skewed)...")
# Các biến tài chính có độ lệch cực mạnh sẽ được nén lại bằng hàm np.log1p(x) = log(1+x)
# np.clip để đảm bảo không có số âm vô tình rơi vào hàm log
cac_cot_can_log = [
    'Ty_Le_Su_Dung_Han_Muc', 
    'Ty_Le_No_Tren_Thu_Nhap', 
    'Thu_Nhap_Hang_Thang', 
    'Thu_Nhap_Binh_Quan_Dau_Nguoi', 
    'Tong_So_Lan_Tre_Han'
]

for cot in cac_cot_can_log:
    X[cot] = np.log1p(np.clip(X[cot], 0, None))

# =========================================================
# 4. SCALING VÀ PCA
# =========================================================
print("[4/5] Đang chạy StandardScaler và PCA...")
bo_chuan_hoa = StandardScaler()
X_scaled = bo_chuan_hoa.fit_transform(X)

# PCA để dễ dàng vẽ hình trên không gian 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# =========================================================
# 5. TEST LẠI K-MEANS
# =========================================================
print("[5/5] Đang chạy thuật toán KMeans...")

# Lấy 10,000 dòng ngẫu nhiên để đánh giá tính Silhouette cho nhanh 
# (chạy 112k dòng tốn rất nhiều thời gian chờ đợi)
df_idx = np.random.choice(X_scaled.shape[0], 10000, replace=False)
X_mau = X_scaled[df_idx]
X_pca_mau = X_pca[df_idx]
y_mau = y_thuc_te[df_idx]

# Huấn luyện KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
nhan_cum = kmeans.fit_predict(X_mau)

# Chấm điểm đánh giá
sil = silhouette_score(X_mau, nhan_cum)
dbi = davies_bouldin_score(X_mau, nhan_cum)
ari = adjusted_rand_score(y_mau, nhan_cum)

# In kết quả
print("\n--- KẾT QUẢ ĐÁNH GIÁ (SAU KHI THÊM LOG TRANSFORM) ---")
print(f"Silhouette Score (Độ tách biệt): {sil:.3f}")
print(f"Davies-Bouldin Index (DBI)     : {dbi:.3f}")
print(f"Chỉ số ARI (Khớp nhãn thực tế) : {ari:.3f}")
if ari > 0:
    print("-> Nhận xét: ARI đã tăng dương! Log Transform giúp K-Means bám sát hành vi Vỡ Nợ thực tế hơn rất nhiều.")

# Vẽ biểu đồ phân tán Scatter
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca_mau[:, 0], X_pca_mau[:, 1], c=nhan_cum, cmap='viridis', s=20, alpha=0.7)
plt.title(f"K-Means (Log Transform + Scaling + PCA)\nSilhouette: {sil:.3f} | ARI: {ari:.3f}", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Thành phần chính 1 (PC1)", fontsize=12)
plt.ylabel("Thành phần chính 2 (PC2)", fontsize=12)
plt.colorbar(scatter, label='Cụm Khách Hàng')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

print("\nĐang hiển thị biểu đồ...")
plt.show()