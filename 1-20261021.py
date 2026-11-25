import pandas as pd
import numpy as np

# 1. 檔案路徑設定
file_paths = [
    r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\ALB_CR_E.xpt",
    r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\BIOPRO_E.xpt",
    r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\BMX_E.xpt",
    r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\CBC_E.xpt",
    r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\DEMO_E.xpt",
    r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\GHB_E.xpt",
    r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\HDL_E.xpt",
    r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\SSUSG_E.xpt",
    r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\THYROD_E.xpt"
]

# 2. 需要保留的 17 個指標欄位
target_vars = [
    'LBXSGTSI', 'LBDSTPSI', 'SSUSG', 'LBXTT4', 'BMXBMI',
    'LBDHDDSI', 'LBXLYPCT', 'LBDNENO', 'LBXHCT', 'LBXRDW',
    'RIDAGEYR', 'LBXGH', 'URXUMS', 'URXUCR', 'RIAGENDR',
    'SEQN', 'LBXSCR'
]

# 3. 讀取並合併數據（以 'SEQN' 為對應主鍵）
merged_df = None
for path in file_paths:
    df = pd.read_sas(path)
    # 僅提取目標欄位與 SEQN
    cols_to_keep = [c for c in df.columns if c in target_vars or c == 'SEQN']
    df = df[cols_to_keep]
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on='SEQN', how='inner')

# 4. 對應重新命名欄位
rename_dict = {
    'LBXSGTSI': 'GGT',
    'LBDSTPSI': 'TP',
    'SSUSG': 'SG',
    'LBXTT4': 'TT4',
    'BMXBMI': 'BMI',
    'LBDHDDSI': 'HDL',
    'LBXLYPCT': 'LYM%',
    'LBDNENO': 'NEU',
    'LBXHCT': 'HCT',
    'LBXRDW': 'RDW',
    'RIDAGEYR': 'age',
    'LBXGH': 'HBA1C',
    'URXUMS': 'ALB',
    'URXUCR': 'Cr',
    'RIAGENDR': 'Sex',
    'SEQN': 'SEQN',
    'LBXSCR': 'Scr'
}

df = merged_df[target_vars].rename(columns=rename_dict)

# 5. 刪除任一指標有空白或缺失值（NaN）的病人資料
df = df.dropna()

# 6. 清除不符合生理邏輯的明顯異常資料（如小於等於 0 的檢驗項目）
check_cols = ['GGT', 'TP', 'SG', 'TT4', 'BMI', 'HDL', 'LYM%', 'NEU', 'HCT', 'RDW',
              'age', 'HBA1C', 'ALB', 'Cr', 'Scr']
for col in check_cols:
    df = df[df[col] > 0]

# 7. TT4 數值轉換 (* 12.871)
df['TT4'] = df['TT4'] * 12.871

# 8. 計算 mALB_uCR = ALB / (Cr * 0.01)
df['mALB_uCR'] = df['ALB'] / (df['Cr'] * 0.01)


# 9. 計算 eGFR (CKD-EPI 2021 肌酸酐公式)
# Sex 說明：1 為男性，2 為女性
def compute_egfr(row):
    scr = row['Scr']
    age = row['age']
    sex = row['Sex']

    if sex == 2:  # 女性
        kappa = 0.7
        alpha = -0.241
        sex_factor = 1.012
    else:  # 男性 (sex == 1)
        kappa = 0.9
        alpha = -0.302
        sex_factor = 1.0

    scr_k = scr / kappa

    # 低值項與高值項判斷
    low_term = (scr_k ** alpha) if scr_k < 1 else 1.0
    high_term = (scr_k ** -1.200) if scr_k >= 1 else 1.0
    age_term = 0.9938 ** age

    return 142 * low_term * high_term * age_term * sex_factor


df['eGFR'] = df.apply(compute_egfr, axis=1)

# 10. 計算 CKD：若 eGFR < 60 或 mALB_uCR >= 30 則為 1，否則為 0
df['CKD'] = ((df['eGFR'] < 60) | (df['mALB_uCR'] >= 30)).astype(int)

# 11. 輸出結果為 CSV 檔案
output_path = r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\processed_data_output.csv"
df.to_csv(output_path, index=False)

print("數據處理完成！已成功導出 CSV 檔案至：", output_path)