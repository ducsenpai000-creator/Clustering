import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 1. ĐƯỜNG DẪN FILE (Hãy sửa lại cho đúng máy bạn)
duong_dan_file = r'D:/CODING_DATA/cs-training.csv'

if not os.path.exists(duong_dan_file):
    print(f"LỖI: Không tìm thấy file tại {duong_dan_file}. Hãy kiểm tra lại đường dẫn!")
else:
    # 2. TẢI DỮ LIỆU
    print("--- KHỞI TẠO QUY TRÌNH LÀM SẠCH TRIỆT ĐỂ ---")
    df_raw = pd.read_csv(duong_dan_file)
    so_dong_goc = len(df_raw)

    if 'Unnamed: 0' in df_raw.columns:
        df_raw = df_raw.drop(columns=['Unnamed: 0'])

    # 3. XÓA TRÙNG LẶP DỮ LIỆU THÔ (Lần 1)
    df = df_raw.drop_duplicates().reset_index(drop=True)
    print(f"-> Đã xóa {so_dong_goc - len(df)} bản ghi trùng lặp thô ban đầu.")

    # 4. CHUẨN HÓA TÊN CỘT
    bang_doi_ten = {
        'SeriousDlqin2yrs': 'Vo_No',
        'RevolvingUtilizationOfUnsecuredLines': 'Ty_Le_Su_Dung',
        'age': 'Tuoi',
        'NumberOfTime30-59DaysPastDueNotWorse': 'Tre_Han_30_59',
        'DebtRatio': 'Ty_Le_No',
        'MonthlyIncome': 'Thu_Nhap',
        'NumberOfOpenCreditLinesAndLoans': 'So_Khoan_Tin_Dung',
        'NumberOfTimes90DaysLate': 'Tre_Han_Tren_90',
        'NumberRealEstateLoansOrLines': 'Vay_BDS',
        'NumberOfTime60-89DaysPastDueNotWorse': 'Tre_Han_60_89',
        'NumberOfDependents': 'Nguoi_Phu_Thuoc'
    }
    df = df.rename(columns=bang_doi_ten)

    # 5. XỬ LÝ GIÁ TRỊ PHI LÝ & THIẾU (ML Imputation)
    df = df[df['Tuoi'] >= 18] # Loại bỏ trẻ em/lỗi nhập liệu tuổi

    # Dùng máy học để dự đoán giá trị thiếu, sample_posterior=True tạo sự khác biệt nhỏ tránh trùng lặp
    it_imputer = IterativeImputer(random_state=42, max_iter=10, sample_posterior=True)
    cols_to_fix = ['Tuoi', 'Ty_Le_No', 'Thu_Nhap', 'So_Khoan_Tin_Dung', 'Vay_BDS', 'Nguoi_Phu_Thuoc']
    df[cols_to_fix] = it_imputer.fit_transform(df[cols_to_fix])
    df['Nguoi_Phu_Thuoc'] = df['Nguoi_Phu_Thuoc'].round().astype(int)

    # 6. GIỚI HẠN NGOẠI LAI (WINSORIZATION)
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return series.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    for col in ['Ty_Le_Su_Dung', 'Ty_Le_No', 'Thu_Nhap']:
        df[col] = cap_outliers(df[col])

    # 7. CHUẨN HÓA DỮ LIỆU
    features = df.drop(columns=['Vo_No'])
    target = df['Vo_No']

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df_scaled['Vo_No'] = target.values

    # Ép kiểu và làm tròn để máy tính nhận diện trùng lặp chính xác 100%
    df_scaled = df_scaled.round(4)
    so_dong_truoc_cuoi = len(df_scaled)
    df_scaled = df_scaled.drop_duplicates()
    
    # Kỹ thuật bổ sung: Groupby để đảm bảo không còn dòng nào giống nhau
    df_scaled = df_scaled.groupby(df_scaled.columns.tolist(), as_index=False).first()

    # 9. XUẤT FILE
    df_scaled.to_excel('Du_Lieu_Tin_Dung_Da_Xu_Ly_Full.xlsx', index=False)

    print(f"\n--- KẾT QUẢ TỔNG KẾT ---")
    print(f"Số dòng ban đầu: {so_dong_goc}")
    print(f"Số dòng cuối cùng: {len(df_scaled)}")
    print(f"Tổng số rác đã dọn: {so_dong_goc - len(df_scaled)}")