<<<<<<< HEAD
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=== BƯỚC 3: KỸ NGHỆ ĐẶC TRƯNG, CHUẨN HÓA VÀ PCA ===")

# ---------------------------------------------------------
# 1. KẾ THỪA DỮ LIỆU TỪ BƯỚC 2 (Đọc file đã làm sạch)
# ---------------------------------------------------------
# Giả sử ở Bước 2, bạn đã lưu dữ liệu sạch (chưa scale) ra file này:
du_lieu = pd.read_excel(r'D:/CODING_DATA/Du_Lieu_Tin_Dung_Da_Xu_Ly_Full.xlsx')
print(f"-> Đã tải thành công dữ liệu sạch: {du_lieu.shape[0]} dòng.")

# ---------------------------------------------------------
# 2. KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING)
# ---------------------------------------------------------
# Tạo biến 1: Tổng số lần trễ hạn các loại
du_lieu['Tổng_Số_Lần_Trễ_Hạn'] = (
    du_lieu['Số_Lần_Trễ_Hạn_30_Đến_59_Ngày'] + 
    du_lieu['Số_Lần_Trễ_Hạn_60_Đến_89_Ngày'] + 
    du_lieu['Số_Lần_Trễ_Hạn_Trên_90_Ngày']
)

# Tạo biến 2: Thu nhập bình quân đầu người trong hộ gia đình
du_lieu['Thu_Nhập_Bình_Quân_Đầu_Người'] = du_lieu['Thu_Nhập_Hàng_Tháng'] / (du_lieu['Số_Người_Phụ_Thuộc'] + 1)
print("-> Đã tạo thêm 2 đặc trưng tài chính mới (Tổng_Số_Lần_Trễ_Hạn, Thu_Nhập_Bình_Quân_Đầu_Người).")

# ---------------------------------------------------------
# 3. TẠO MA TRẬN ĐẶC TRƯNG & CHUẨN HÓA (STANDARD SCALER)
# ---------------------------------------------------------
# Tách riêng nhãn ra khỏi đặc trưng để không bị chuẩn hóa
nhan = du_lieu['Vỡ_Nợ_Trong_2_Năm'].values
dac_trung = du_lieu.drop(columns=['Vỡ_Nợ_Trong_2_Năm'])

bo_chuan_hoa = StandardScaler()
ma_tran_dac_trung_da_chuan_hoa = bo_chuan_hoa.fit_transform(dac_trung)

# Lưu bộ dữ liệu Full (đã scale và thêm biến mới) để chạy phân cụm ở Bước 4
bang_full_san_sang = pd.DataFrame(ma_tran_dac_trung_da_chuan_hoa, columns=dac_trung.columns).round(3)
bang_full_san_sang['Vỡ_Nợ_Trong_2_Năm'] = nhan

bang_full_san_sang.to_csv('Du_Lieu_Full_San_Sang_Clustering.csv', index=False, encoding = "utf-8-sig")
print("-> Đã chuẩn hóa dữ liệu và xuất file: Du_Lieu_Full_San_Sang_Clustering.csv")

# ---------------------------------------------------------
# 4. GIẢM CHIỀU DỮ LIỆU BẰNG PCA (HỖ TRỢ TRỰC QUAN HÓA)
# ---------------------------------------------------------
pca = PCA(n_components=2)
ma_tran_pca = pca.fit_transform(ma_tran_dac_trung_da_chuan_hoa)

ty_le_thong_tin = pca.explained_variance_ratio_
print(f"-> Khả năng bảo toàn thông tin của PCA: PC1 ({ty_le_thong_tin[0]*100:.2f}%) + PC2 ({ty_le_thong_tin[1]*100:.2f}%) = Tổng {sum(ty_le_thong_tin)*100:.2f}%")

# Lưu kết quả PCA 2 chiều để vẽ đồ thị
bang_pca = pd.DataFrame(ma_tran_pca, columns=['Thành_Phần_Chính_1', 'Thành_Phần_Chính_2']).round(3)
bang_pca['Vỡ_Nợ_Trong_2_Năm'] = nhan

bang_pca.to_csv('Du_Lieu_PCA_2D.csv', index=False,encoding ="utf-8-sig")
print("-> Đã xuất file PCA 2 chiều: Du_Lieu_PCA_2D.csv")
=======
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=== BƯỚC 3: KỸ NGHỆ ĐẶC TRƯNG, CHUẨN HÓA VÀ PCA ===")

# ---------------------------------------------------------
# 1. KẾ THỪA DỮ LIỆU TỪ BƯỚC 2 (Đọc file đã làm sạch)
# ---------------------------------------------------------
# Giả sử ở Bước 2, bạn đã lưu dữ liệu sạch (chưa scale) ra file này:
du_lieu = pd.read_excel(r'D:/CODING_DATA/Du_Lieu_Tin_Dung_Da_Xu_Ly_Full.xlsx')
print(f"-> Đã tải thành công dữ liệu sạch: {du_lieu.shape[0]} dòng.")

# ---------------------------------------------------------
# 2. KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING)
# ---------------------------------------------------------
# Tạo biến 1: Tổng số lần trễ hạn các loại
du_lieu['Tổng_Số_Lần_Trễ_Hạn'] = (
    du_lieu['Số_Lần_Trễ_Hạn_30_Đến_59_Ngày'] + 
    du_lieu['Số_Lần_Trễ_Hạn_60_Đến_89_Ngày'] + 
    du_lieu['Số_Lần_Trễ_Hạn_Trên_90_Ngày']
)

# Tạo biến 2: Thu nhập bình quân đầu người trong hộ gia đình
du_lieu['Thu_Nhập_Bình_Quân_Đầu_Người'] = du_lieu['Thu_Nhập_Hàng_Tháng'] / (du_lieu['Số_Người_Phụ_Thuộc'] + 1)
print("-> Đã tạo thêm 2 đặc trưng tài chính mới (Tổng_Số_Lần_Trễ_Hạn, Thu_Nhập_Bình_Quân_Đầu_Người).")

# ---------------------------------------------------------
# 3. TẠO MA TRẬN ĐẶC TRƯNG & CHUẨN HÓA (STANDARD SCALER)
# ---------------------------------------------------------
# Tách riêng nhãn ra khỏi đặc trưng để không bị chuẩn hóa
nhan = du_lieu['Vỡ_Nợ_Trong_2_Năm'].values
dac_trung = du_lieu.drop(columns=['Vỡ_Nợ_Trong_2_Năm'])

bo_chuan_hoa = StandardScaler()
ma_tran_dac_trung_da_chuan_hoa = bo_chuan_hoa.fit_transform(dac_trung)

# Lưu bộ dữ liệu Full (đã scale và thêm biến mới) để chạy phân cụm ở Bước 4
bang_full_san_sang = pd.DataFrame(ma_tran_dac_trung_da_chuan_hoa, columns=dac_trung.columns).round(3)
bang_full_san_sang['Vỡ_Nợ_Trong_2_Năm'] = nhan

bang_full_san_sang.to_csv('Du_Lieu_Full_San_Sang_Clustering.csv', index=False, encoding = "utf-8-sig")
print("-> Đã chuẩn hóa dữ liệu và xuất file: Du_Lieu_Full_San_Sang_Clustering.csv")

# ---------------------------------------------------------
# 4. GIẢM CHIỀU DỮ LIỆU BẰNG PCA (HỖ TRỢ TRỰC QUAN HÓA)
# ---------------------------------------------------------
pca = PCA(n_components=2)
ma_tran_pca = pca.fit_transform(ma_tran_dac_trung_da_chuan_hoa)

ty_le_thong_tin = pca.explained_variance_ratio_
print(f"-> Khả năng bảo toàn thông tin của PCA: PC1 ({ty_le_thong_tin[0]*100:.2f}%) + PC2 ({ty_le_thong_tin[1]*100:.2f}%) = Tổng {sum(ty_le_thong_tin)*100:.2f}%")

# Lưu kết quả PCA 2 chiều để vẽ đồ thị
bang_pca = pd.DataFrame(ma_tran_pca, columns=['Thành_Phần_Chính_1', 'Thành_Phần_Chính_2']).round(3)
bang_pca['Vỡ_Nợ_Trong_2_Năm'] = nhan

bang_pca.to_csv('Du_Lieu_PCA_2D.csv', index=False,encoding ="utf-8-sig")
print("-> Đã xuất file PCA 2 chiều: Du_Lieu_PCA_2D.csv")
>>>>>>> e7ceb33b4136739c786d6fd6e2298706721ba2ec
print("\n=== HOÀN TẤT BƯỚC 3 ===")