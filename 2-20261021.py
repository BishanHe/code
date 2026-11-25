###1. 导入整理好的江川文件
### 设置程序工作目录
import os

# 设置工作目录为新文件夹路径
os.chdir(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021")
print("当前工作目录：", os.getcwd())

import pandas as pd
jiangchuan_data = pd.read_csv(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\processed_data_output.csv")

###2. 疾病组划分基线比较
continuous_vars = ['GGT', 'TP',	'SG',	'TT4',	'BMI',	'HDL',	'LYM%',	'NEU',	'HCT',
                   'RDW',	'age',	'HBA1C',	'ALB',	'Cr',	'Scr',
                   'mALB_uCR',	'eGFR']
categorical_vars = ['Sex']


import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact

def baseline_table(df, group_col, continuous_vars, categorical_vars, alpha=0.05,
                   output_excel=False, excel_name="baseline_table.xlsx"):
    """
    生成基线特征统计表（两组比较 + Total）

    参数：
        df : pandas.DataFrame
        group_col : str, 分组列名（二分类，取值 0/1 或 'A'/'B'）
        continuous_vars : list, 连续变量列名
        categorical_vars : list, 分类变量列名
        alpha : float, 显著性水平
        output_excel : bool, 是否导出 Excel
        excel_name : str, 导出文件名（需带 .xlsx 后缀）
    """
    results = []

    group_labels = df[group_col].unique()
    if len(group_labels) != 2:
        raise ValueError("⚠️ 当前函数只支持两组比较")

    g1, g0 = group_labels
    data1 = df[df[group_col] == g1]
    data0 = df[df[group_col] == g0]
    data_all = df.dropna(subset=[group_col])  # 总体（去掉分组缺失）

    # --- 连续变量 ---
    for var in continuous_vars:
        g1_vals = data1[var].dropna()
        g0_vals = data0[var].dropna()
        all_vals = data_all[var].dropna()

        # 正态性检验
        normal_g1 = shapiro(g1_vals)[1] > alpha if len(g1_vals) <= 5000 else True
        normal_g0 = shapiro(g0_vals)[1] > alpha if len(g0_vals) <= 5000 else True

        if normal_g1 and normal_g0:
            # Welch's t-test
            stat, p = ttest_ind(g1_vals, g0_vals, equal_var=False)
            desc1 = f"{g1_vals.mean():.2f} ± {g1_vals.std():.2f}"
            desc0 = f"{g0_vals.mean():.2f} ± {g0_vals.std():.2f}"
            desc_all = f"{all_vals.mean():.2f} ± {all_vals.std():.2f}"
            method = "Welch t-test"
        else:
            # Mann-Whitney U
            stat, p = mannwhitneyu(g1_vals, g0_vals, alternative="two-sided")
            desc1 = f"{np.median(g1_vals):.2f} ({np.percentile(g1_vals,25):.2f}–{np.percentile(g1_vals,75):.2f})"
            desc0 = f"{np.median(g0_vals):.2f} ({np.percentile(g0_vals,25):.2f}–{np.percentile(g0_vals,75):.2f})"
            desc_all = f"{np.median(all_vals):.2f} ({np.percentile(all_vals,25):.2f}–{np.percentile(all_vals,75):.2f})"
            method = "Mann-Whitney U"

        # P值显示格式
        p_display = "<0.001" if p < 0.001 else round(p, 4)

        results.append({
            "Variable": var,
            f"{g1} (n={len(g1_vals)})": desc1,
            f"{g0} (n={len(g0_vals)})": desc0,
            f"Total (n={len(all_vals)})": desc_all,
            "Test": method,
            "P-value": p_display
        })

    # --- 分类变量 ---
    for var in categorical_vars:
        cross_tab = pd.crosstab(df[var], df[group_col])
        if cross_tab.shape[1] != 2:
            raise ValueError(f"⚠️ 分类变量 {var} 在不同组中的水平异常，检查数据")

        # χ² 或 Fisher
        chi2, p, dof, ex = chi2_contingency(cross_tab)
        if (ex < 5).any():  # Fisher only works for 2x2
            if cross_tab.shape == (2,2):
                _, p = fisher_exact(cross_tab)
                method = "Fisher exact"
            else:
                method = "Chi-square (注意小样本可能不稳)"
        else:
            method = "Chi-square"

        p_display = "<0.001" if p < 0.001 else round(p, 4)

        # 格式化 n (%)
        desc_g1, desc_g0, desc_all = [], [], []
        for level in cross_tab.index:
            n1, n0 = cross_tab.loc[level, g1], cross_tab.loc[level, g0]
            n_all = n1 + n0
            perc1 = 100 * n1 / len(data1)
            perc0 = 100 * n0 / len(data0)
            perc_all = 100 * n_all / len(data_all)

            desc_g1.append(f"{level}: {n1} ({perc1:.2f}%)")
            desc_g0.append(f"{level}: {n0} ({perc0:.2f}%)")
            desc_all.append(f"{level}: {n_all} ({perc_all:.2f}%)")

        results.append({
            "Variable": var,
            f"{g1} (n={len(data1)})": "; ".join(desc_g1),
            f"{g0} (n={len(data0)})": "; ".join(desc_g0),
            f"Total (n={len(data_all)})": "; ".join(desc_all),
            "Test": method,
            "P-value": p_display
        })

    results_df = pd.DataFrame(results)

    # --- 导出 Excel ---
    if output_excel:
        results_df.to_excel(excel_name, index=False)
        print(f"✅ 基线表已保存: {excel_name}")

    return results_df

baseline_table(df=jiangchuan_data, group_col='CKD',
               continuous_vars=continuous_vars,categorical_vars=categorical_vars,
               output_excel=True,excel_name="疾病组对照组基线表.xlsx")

###3. 训练集测试集划分和基线表
## 备份原始数据
model_data = jiangchuan_data

#%%
### 划分训练集测试集
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    model_data,
    test_size=0.2,
    random_state=18,
    stratify=model_data['CKD']  # 保证分层抽样
)

print(f"Train size: {train_df.shape}, Test size: {test_df.shape}")

### 在model_data_3上增加训练集/测试集信息
model_data['cohort'] = model_data['SEQN'].apply(
    lambda x: 0 if x in train_df['SEQN'].values else 1
)

# 检查分布
print(model_data['cohort'].value_counts())

### 进行基线分析
baseline_table(df=model_data, group_col='cohort', continuous_vars=continuous_vars, categorical_vars=categorical_vars, output_excel=True,excel_name="训练集测试集基线表.xlsx")

###4. 保存训练集测试集
import pandas as pd

test_df.to_csv("test_df.csv", index=False)
train_df.to_csv("train_df.csv", index=False)

print("✅ 两个数据框已分别保存为 test_df.csv 和 train_df.csv")

###5. 特征选择
### 数据备份
feature_model = train_df

continuous_vars_feature = ['GGT', 'TP',	'SG',	'TT4',	'BMI',	'HDL',	'LYM%',	'NEU',	'HCT',
                   'RDW',	'age',	'HBA1C',	'ALB',	'Cr',	'Scr',
                   'mALB_uCR',	'eGFR']
categorical_vars_feature = ['Sex']
### 拆分X、Y
# 1. 定义因变量 y（结局变量）
y_train_feature = train_df["CKD"]

# 2. 定义自变量 X（去掉结局变量和ID列）
X_train_feature = train_df.drop(columns=["CKD", "SEQN"], errors="ignore")

# 同理对测试集也拆分
y_test_feature = test_df["CKD"]
X_test_feature = test_df.drop(columns=["CKD", "SEQN"], errors="ignore")

##5.1 方差筛选
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

# 1. 先分别处理数值和分类特征
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore",drop="first",sparse_output=False)

# 2. 分别fit_transform并获取列名
X_num = pd.DataFrame(num_transformer.fit_transform(X_train_feature[continuous_vars_feature]),
                     columns=continuous_vars_feature, index=X_train_feature.index)

X_cat = pd.DataFrame(cat_transformer.fit_transform(X_train_feature[categorical_vars_feature]),
                     columns=cat_transformer.get_feature_names_out(categorical_vars_feature),
                     index=X_train_feature.index)

# 3. 合并特征
X_train_encoded = pd.concat([X_num, X_cat], axis=1)

# 4. 进行方差选择
selector = VarianceThreshold(threshold=0.01)

X_train_var = selector.fit_transform(X_train_encoded)
feature_variances = pd.Series(selector.variances_, index=X_train_encoded.columns)
X_test_var = selector.transform(pd.concat([
    pd.DataFrame(num_transformer.transform(X_test_feature[continuous_vars_feature]), columns=continuous_vars_feature, index=X_test_feature.index),
    pd.DataFrame(cat_transformer.transform(X_test_feature[categorical_vars_feature]),
                 columns=cat_transformer.get_feature_names_out(categorical_vars_feature), index=X_test_feature.index)
], axis=1))

# 5. 查看被drop的特征
mask = selector.get_support()  # True表示保留，False表示丢掉
kept_features = X_train_encoded.columns[mask]
dropped_features = X_train_encoded.columns[~mask]

print("原始特征数:", X_train_encoded.shape[1])
print("筛选后特征数:", X_train_var.shape[1])
print("保留的特征:", list(kept_features))
print("丢掉的特征:", list(dropped_features))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

epsilon = 1e-6  # 防止log(0)
feature_variances_log = np.log10(feature_variances + epsilon)

# 将 mask 转为 'kept' / 'dropped'
feature_status_str = np.where(mask, 'kept', 'dropped')



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 预处理

epsilon = 1e-6
feature_variances_log = np.log10(feature_variances + epsilon)
feature_status_str = np.where(mask, 'kept', 'dropped')

# 创建 DataFrame

df = pd.DataFrame({
'Feature': feature_variances_log.index,
'LogVariance': feature_variances_log.values,
'Status': feature_status_str
})

# 按方差排序，便于阅读

df = df.sort_values(by='LogVariance', ascending=True)

# 设置风格



# 可视化2
category_cols = [col for col in X_train_encoded.columns if '_' in col]  # 简单判断分类变量
category_ratios = X_train_encoded[category_cols].mean()

plt.figure(figsize=(12,4))
sns.barplot(x=category_ratios.index, y=category_ratios.values)
plt.xticks(rotation=90)
plt.ylabel('Proportion of 1s')
plt.title('Category Proportion (0/1) - Low variance columns likely dropped')
plt.savefig('Category Proportion .svg', bbox_inches='tight')
plt.show()

##5.2 VIF验证
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

# 假设你已经有：
# X_train_var  —— 方差筛选后的 numpy 矩阵
# kept_features —— 保留的特征名列表

# 1. 转回 DataFrame
X_train_vif = pd.DataFrame(X_train_var, columns=kept_features, index=X_train_feature.index)

# 2. 计算 VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_train_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_train_vif.values, i)
    for i in range(X_train_vif.shape[1])
]

print(vif_data.sort_values("VIF", ascending=False))



### 相关性分析
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 计算相关系数矩阵
corr_matrix = X_train_vif.corr()

# 2. 绘制热图


### 自动找到相关性高的变量对
# 3. 找出高度相关的特征对
threshold = 0.8
high_corr = np.where(np.abs(corr_matrix) > threshold)

high_corr_pairs = [
    (corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
    for i, j in zip(*high_corr)
    if i < j   # 避免重复 & 自己和自己
]

print("高度相关的变量对（|r| > 0.8）：")
for f1, f2, corr in high_corr_pairs:
    print(f"{f1} - {f2} : {corr:.2f}")

### 删除相关性较高的部分变量
# 1. 把训练集和测试集都转为 DataFrame
X_train_df = pd.DataFrame(X_train_var, columns=kept_features, index=X_train_feature.index)
X_test_df  = pd.DataFrame(X_test_var,  columns=kept_features, index=X_test_feature.index)

# 2. 定义要删除的变量
drop_vars = ['ALB', 'Cr', 'Scr', 'mALB_uCR', 'eGFR', 'Sex']

# 3. 在 train 和 test 中同时删除这些变量
X_train_new = X_train_df.drop(columns=drop_vars, errors="ignore")
X_test_new  = X_test_df.drop(columns=drop_vars, errors="ignore")

#print("原始训练集形状:", X_train_df.shape)
#print("新训练集形状:", X_train_new.shape)
#print("原始测试集形状:", X_test_df.shape)
#print("新测试集形状:", X_test_new.shape)

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 重新计算 VIF
vif_data_new = pd.DataFrame()
vif_data_new["feature"] = X_train_new.columns
vif_data_new["VIF"] = [
    variance_inflation_factor(X_train_new.values, i)
    for i in range(X_train_new.shape[1])
]

# 2. 输出 VIF 排序表
#print(vif_data_new.sort_values("VIF", ascending=False))

# 3. 画条形图


def calculate_vif(df, thresh=10.0):
    """
    迭代删除 VIF > thresh 的变量
    """
    variables = list(df.columns)
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(df[variables].values, i) for i in range(len(variables))]
        max_vif = max(vif)
        if max_vif > thresh:
            maxloc = vif.index(max_vif)
            print(f"Drop {variables[maxloc]} with VIF={max_vif:.2f}")
            variables.pop(maxloc)
            dropped = True
    return df[variables]

# 在 X_train_new 上应用
X_train_vif_filtered = calculate_vif(X_train_new, thresh=10)
X_test_vif_filtered  = X_test_new[X_train_vif_filtered.columns]  # 保证列一致

#print("最终保留的变量数:", X_train_vif_filtered.shape[1])

# 1. 定义你的连续变量和分类变量
continuous_vars = ['GGT', 'TP', 'SG', 'TT4', 'BMI', 'HDL', 'LYM%', 'NEU', 'HCT',
                   'RDW', 'age', 'HBA1C', 'ALB', 'Cr', 'Scr',
                   'mALB_uCR', 'eGFR']
categorical_vars = ['Sex']
# 2. 定义之前方差+VIF筛选保留下来的34个变量
selected_features = ['GGT', 'TP', 'SG', 'TT4', 'BMI', 'HDL', 'LYM%', 'NEU', 'HCT',
                   'RDW', 'age', 'HBA1C']

# 3. 对训练集和测试集进行编码
cat_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# 编码训练集分类变量
X_train_cat = pd.DataFrame(
    cat_encoder.fit_transform(train_df[categorical_vars]),
    columns=cat_encoder.get_feature_names_out(categorical_vars),
    index=train_df.index
)

# 编码测试集分类变量
X_test_cat = pd.DataFrame(
    cat_encoder.transform(test_df[categorical_vars]),
    columns=cat_encoder.get_feature_names_out(categorical_vars),
    index=test_df.index
)

# 标准化连续变量
scaler = StandardScaler()
X_train_num = pd.DataFrame(scaler.fit_transform(train_df[continuous_vars]),
                           columns=continuous_vars, index=train_df.index)
X_test_num = pd.DataFrame(scaler.transform(test_df[continuous_vars]),
                          columns=continuous_vars, index=test_df.index)

# 4. 合并编码后的连续和分类特征
X_train_encoded = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_encoded  = pd.concat([X_test_num, X_test_cat], axis=1)

# 5. 根据之前选出的34个变量抽列
X_train_final = X_train_encoded[selected_features].copy()
X_test_final  = X_test_encoded[selected_features].copy()

print("最终训练集形状:", X_train_final.shape)
print("最终测试集形状:", X_test_final.shape)

print("原始categorical_vars:", categorical_vars)
print("train_df实际的列:", train_df.columns.tolist())
print("编码后列:", cat_encoder.get_feature_names_out(categorical_vars))

y_train_final = train_df["CKD"].copy()
y_test_final  = test_df["CKD"].copy()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, ElasticNet, enet_path
# =============================
# 1. 用交叉验证寻找最佳 alpha 和 l1_ratio
# =============================

# alpha 范围
alphas = np.logspace(-4, 1, 300)

# ElasticNetCV, l1_ratio 可调整 0~1
# 这里选择 0.5 为例（L1/L2 权重比例）
enet_cv = ElasticNetCV(l1_ratio=0.9, alphas=alphas, cv=10, max_iter=20000, random_state=42)
enet_cv.fit(X_train_final, y_train_final)

best_alpha = enet_cv.alpha_
best_l1_ratio = enet_cv.l1_ratio_
print("最佳 alpha:", best_alpha)
print("L1_ratio:", best_l1_ratio)

# =============================
# 2. 计算 1-SE alpha（可选，类似 Lasso）
# =============================
mse_mean = np.mean(enet_cv.mse_path_, axis=1)
mse_std = np.std(enet_cv.mse_path_, axis=1) / np.sqrt(enet_cv.mse_path_.shape[1])

idx_best = np.where(enet_cv.alphas_ == best_alpha)[0][0]
min_mse = mse_mean[idx_best]
min_mse_plus_se = min_mse + mse_std[idx_best]

alpha_1se = enet_cv.alphas_[mse_mean <= min_mse_plus_se].max()
print("1-SE alpha:", alpha_1se)

# =============================
alphas_enet, coefs, _ = enet_path(X_train_final, y_train_final, l1_ratio=best_l1_ratio, alphas=alphas)
alphas_enet = alphas_enet[::-1]
coefs = coefs[:, ::-1]

# =============================
# 5. 输出最佳 alpha 下的变量
# =============================
coef_best = pd.Series(enet_cv.coef_, index=X_train_final.columns)
selected_features_best = coef_best[coef_best != 0].index.tolist()
print(f"Best alpha 非零系数数量: {len(selected_features_best)}")
print("Best alpha 选择的变量:")
print(selected_features_best)

# =============================
# 6. 输出 1-SE alpha 下的变量
# =============================
enet_1se = ElasticNet(alpha=alpha_1se, l1_ratio=best_l1_ratio, max_iter=20000, random_state=42)
enet_1se.fit(X_train_final, y_train_final)

coef_1se = pd.Series(enet_1se.coef_, index=X_train_final.columns)
selected_features_1se = coef_1se[coef_1se != 0].index.tolist()
print(f"1-SE alpha 非零系数数量: {len(selected_features_1se)}")
print("1-SE alpha 选择的变量:")
print(selected_features_1se)


###6. 模型构建
#%%
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

# =========================
# 1. 定义原始变量
# =========================
continuous_vars = ['GGT', 'TP', 'SG', 'TT4', 'BMI', 'HDL', 'LYM%', 'NEU', 'HCT',
                   'RDW', 'age', 'HBA1C', 'ALB', 'Cr', 'Scr',
                   'mALB_uCR', 'eGFR']
categorical_vars = ['Sex']
# =========================
# 2. 之前选出的变量（最终建模用）
# =========================
selected_features = ['GGT', 'TP', 'SG', 'TT4', 'BMI', 'HDL', 'LYM%', 'NEU', 'HCT',
                   'RDW', 'age', 'HBA1C']   # 包括编码后的列

# =========================
# 3. 识别 selected_features 中涉及的连续 & 分类变量
# =========================
# 先把还在 continuous_vars 里的挑出来
cont_for_model = [col for col in selected_features if col in continuous_vars]

# 再识别出 selected_features 中的分类变量编码（比如 sex_1 -> sex）
cat_for_model = [col for col in categorical_vars if any(col in feat for feat in selected_features)]

print("本次建模使用的连续变量:", cont_for_model)
print("本次建模使用的分类变量:", cat_for_model)

# =========================
# 4. 编码 + 标准化
# =========================
cat_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

# --- 分类变量编码 ---
X_train_cat = pd.DataFrame(
    cat_encoder.fit_transform(train_df[cat_for_model]),
    columns=cat_encoder.get_feature_names_out(cat_for_model),
    index=train_df.index
)

X_test_cat = pd.DataFrame(
    cat_encoder.transform(test_df[cat_for_model]),
    columns=cat_encoder.get_feature_names_out(cat_for_model),
    index=test_df.index
)

# --- 连续变量标准化 ---
scaler = StandardScaler()
X_train_num = pd.DataFrame(
    scaler.fit_transform(train_df[cont_for_model]),
    columns=cont_for_model,
    index=train_df.index
)
X_test_num = pd.DataFrame(
    scaler.transform(test_df[cont_for_model]),
    columns=cont_for_model,
    index=test_df.index
)

# =========================
# 5. 合并，并只保留 selected_features
# =========================
X_train_encoded = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_encoded  = pd.concat([X_test_num, X_test_cat], axis=1)

X_train_model = X_train_encoded[selected_features].copy()
X_test_model  = X_test_encoded[selected_features].copy()

print("最终训练集形状:", X_train_model.shape)
print("最终测试集形状:", X_test_model.shape)

#%%
## 保存数据标准化参数用于之后外部队列
data_dict = {
    "scaler": scaler,
    "encoder": cat_encoder,
    "selected_features": selected_features,
    "cont_for_model": cont_for_model,
    "cat_for_model": cat_for_model
}
#%%
import joblib
joblib.dump(data_dict, "preprocessing.pkl")
#%%
y_train_model = train_df["CKD"].copy()
y_test_model  = test_df["CKD"].copy()
#%%
## 定义通用表征输出函数
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

def evaluate_model(y_true, y_pred_proba, model_name="Model", n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)

    # 确保输入是 numpy array，避免 pandas 索引问题
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    # === 1. 计算 AUC 和 bootstrap 95% CI ===
    auc = roc_auc_score(y_true, y_pred_proba)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred_proba), len(y_pred_proba))
        if len(np.unique(y_true[indices])) < 2:  # 避免全是一个类别
            continue
        score = roc_auc_score(y_true[indices], y_pred_proba[indices])
        bootstrapped_scores.append(score)

    ci_lower = np.percentile(bootstrapped_scores, 2.5)
    ci_upper = np.percentile(bootstrapped_scores, 97.5)

    # === 2. 寻找最佳 cutoff（Youden’s J statistic） ===
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]

    # === 3. 计算在最佳 cutoff 下的性能指标 ===
    y_pred_opt = (y_pred_proba >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred_opt)
    sensitivity = recall_score(y_true, y_pred_opt)   # = TP / (TP+FN)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_pred_opt)
    f1 = f1_score(y_true, y_pred_opt)

    # === 4. 汇总结果 ===
    results = pd.DataFrame({
        "Model": [model_name],
        "AUC": [auc],
        "AUC 95% CI": [f"[{ci_lower:.3f}, {ci_upper:.3f}]"],
        "Optimal Cutoff": [best_thresh],
        "Accuracy": [accuracy],
        "Sensitivity": [sensitivity],
        "Specificity": [specificity],
        "Precision": [precision],
        "F1 Score": [f1]
    })

    return results
#%%
X_train_model.to_csv("X_train.csv", index=False)
X_test_model.to_csv("X_test.csv", index=False)
y_train_model.to_csv("y_train.csv", index=False)
y_test_model.to_csv("y_test.csv", index=False)

### 6.1 logistic regression
#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, brier_score_loss
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

# --- 1. 标准化 ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_model)
X_test_scaled = scaler.transform(X_test_model)

# --- 2. 定义逻辑回归 + 网格搜索参数 ---
param_grid = {
    "penalty": ["l1", "l2"],
    "C": np.logspace(-3, 2, 10),
    "solver": ["liblinear", "saga"],
    "class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_reg = LogisticRegression(max_iter=5000, random_state=42)

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

# --- 3. 拟合 ---
grid_search.fit(X_train_scaled, y_train_model)

print("最佳参数:", grid_search.best_params_)
print("最佳 AUC (CV):", grid_search.best_score_)

# --- 4. 拿到最佳模型 ---
best_lr = grid_search.best_estimator_

# --- 5. 概率校准（Platt scaling / isotonic regression） ---
calibrated_lr = CalibratedClassifierCV(best_lr, method="isotonic", cv=5)
calibrated_lr.fit(X_train_scaled, y_train_model)

# --- 6. 测试集预测 ---
y_pred_proba = calibrated_lr.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_pred_proba > 0.5).astype(int)

auc = roc_auc_score(y_test_model, y_pred_proba)
acc = accuracy_score(y_test_model, y_test_pred)
brier = brier_score_loss(y_test_model, y_pred_proba)
print(f"测试集 Accuracy: {acc:.4f}")
print(f"测试集 AUC: {auc:.4f}")
print(f"测试集 Brier: {brier:.4f}")


#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# 设置风格
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12

# ---------------------
# 1. ROC 曲线
# ---------------------
fpr, tpr, thresholds = roc_curve(y_test_model, y_pred_proba)
roc_auc = auc(fpr, tpr)



# ---------------------
# 2. 校准曲线（折线）
# ---------------------
prob_true, prob_pred = calibration_curve(y_test_model, y_pred_proba, n_bins=10)


#%%
# 5. 用最佳模型预测测试集
best_log_reg = grid_search.best_estimator_
y_pred_lr = best_log_reg.predict(X_test_model)
y_pred_proba_lr = best_log_reg.predict_proba(X_test_model)[:, 1]
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# AUC
auc_test = roc_auc_score(y_test_model, y_pred_proba_lr)
print("测试集 AUC:", auc_test)

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_test_model, y_pred_proba_lr)


#%%
results_lr = evaluate_model(y_test_model, y_pred_proba_lr, model_name=best_log_reg)

### 6.2 Random Forest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 8, 12, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5]
}

rf = RandomForestClassifier(random_state=42, n_jobs=1)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=1
)
grid_search.fit(X_train_model, y_train_model)

print("最佳参数:", grid_search.best_params_)
print("最佳AUC:", grid_search.best_score_)

# 1. 获取调参后的最佳模型
best_rf = grid_search.best_estimator_
#%%
# 2. 预测概率
y_test_pred_rf = best_rf.predict_proba(X_test_model)[:, 1]

from sklearn.metrics import roc_auc_score
test_auc = roc_auc_score(y_test_model, y_test_pred_rf)
print("测试集 AUC:", test_auc)
#%%
results_rf = evaluate_model(y_test_model, y_test_pred_rf, model_name=best_rf)
#%% md
### 6.3 XGBoost
#%%
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# 1. 定义参数搜索空间
param_grid = {
    "n_estimators": [200, 500, 800],   # 树的数量
    "max_depth": [3, 5, 7],            # 树的最大深度
    "learning_rate": [0.01, 0.05, 0.1],# 学习率
    "subsample": [0.7, 0.8, 1.0],      # 每棵树的样本采样比例
    "colsample_bytree": [0.7, 0.8, 1.0]# 每棵树的特征采样比例
}

# 2. 初始化模型
xgb_clf = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss",
)

# 3. 网格搜索
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=1
)

grid_search.fit(X_train_model, y_train_model)

# 4. 打印最优参数和最优分数
print("最佳参数:", grid_search.best_params_)
print("最佳AUC:", grid_search.best_score_)

# 5. 拿到最优模型
best_xgb = grid_search.best_estimator_
#%%
# 6. 预测
y_test_pred_xgb  = best_xgb.predict_proba(X_test_model)[:, 1]
from sklearn.metrics import roc_auc_score
auc_test = roc_auc_score(y_test_model, y_test_pred_xgb)
print("测试集 AUC:", auc_test)
#%%
results_xgb = evaluate_model(y_test_model, y_test_pred_xgb, model_name=best_xgb)
#%% md
### 6.4 SVC
#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# =========================
# 1. 构建 Pipeline（关键）
# =========================
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        probability=True,
        class_weight='balanced',
        random_state=42
    ))
])

# =========================
# 2. 合理的参数网格
# =========================
param_grid = [
    # Linear SVM（医学筛查常常更好）
    {
        'svm__kernel': ['linear'],
        'svm__C': [0.01, 0.1, 1, 10]
    },
    # RBF SVM
    {
        'svm__kernel': ['rbf'],
        'svm__C': [0.1, 1, 10],
        'svm__gamma': [0.001, 0.01, 0.1, 'scale']
    }
]

# =========================
# 3. GridSearch（AUC）
# =========================
grid = GridSearchCV(
    pipe,
    param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

grid.fit(X_train_model, y_train_model)

print("最佳参数:", grid.best_params_)
print("最佳 AUC:", grid.best_score_)

# =========================
# 4. 测试集 AUC
# =========================
best_svm = grid.best_estimator_

y_test_pred_svm = best_svm.predict_proba(X_test_model)[:, 1]
test_auc_svm = roc_auc_score(y_test_model, y_test_pred_svm)

print("SVM 测试集 AUC:", test_auc_svm)

results_svc = evaluate_model(y_test_model, y_test_pred_svm, model_name=best_svm)
#%% md
### 6.5 朴素贝叶斯
# --- 6.5 Naive Bayes (with hyperparameter tuning) ---

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np

# 1️⃣ 定义参数搜索范围
param_grid_nb = {
    "var_smoothing": np.logspace(-12, -6, 7)
}

# 2️⃣ 初始化模型
nb = GaussianNB()

# 3️⃣ Grid Search + 10-fold CV
grid_nb = GridSearchCV(
    estimator=nb,
    param_grid=param_grid_nb,
    cv=10,
    scoring="roc_auc",
    n_jobs=-1
)

# 4️⃣ 在训练集上拟合
grid_nb.fit(X_train_model, y_train_model)

# 5️⃣ 输出最佳参数
print("最佳参数:", grid_nb.best_params_)
print("最佳AUC:", grid_nb.best_score_)

# 6️⃣ 取最佳模型
nb_model = grid_nb.best_estimator_

# -----------------------
# 用最佳模型进行预测
# -----------------------

# 预测
y_pred_nb = nb_model.predict(X_test_model)
y_prob_nb = nb_model.predict_proba(X_test_model)[:, 1]  # 概率，用于 ROC
# ---- 2. 训练集预测 ----
y_train_prob_nb = nb_model.predict_proba(X_train_model)[:, 1]
y_train_auc = roc_auc_score(y_train_model, y_train_prob_nb)
print(f"训练集 AUC: {y_train_auc:.4f}")

# ---- 3. 测试集预测 ----
y_test_prob_nb = nb_model.predict_proba(X_test_model)[:, 1]
y_test_auc = roc_auc_score(y_test_model, y_test_prob_nb)
print(f"测试集 AUC: {y_test_auc:.4f}")
#%%
results_nb = evaluate_model(y_test_model, y_prob_nb, model_name=nb_model)

#%% md
### 6.6 神经网络
#%%
# ============================================
# 6.6 神经网络（含超参数搜索 + 类别不平衡处理）
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np

# 转换为 tensor
X_train_tensor = torch.tensor(X_train_model.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_model.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_model.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_model.values, dtype=torch.float32).view(-1, 1)

# 划分训练集 & 验证集
dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义改进版模型
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # 不要 sigmoid，这里输出 logits
        return x

nn_model = Net(X_train_model.shape[1])

# 损失函数 + 优化器 + 学习率调度
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# 提前停止参数
best_val_loss = float("inf")
patience, trials = 10, 0

# 训练
for epoch in range(200):  # 最多 200 epoch
    nn_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = nn_model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    # 验证
    nn_model.eval()
    val_losses = []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            y_val_pred = nn_model(X_val)
            val_loss = criterion(y_val_pred, y_val)
            val_losses.append(val_loss.item())
    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trials = 0
        best_model_state = nn_model.state_dict()
    else:
        trials += 1
        if trials >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 恢复最佳模型
nn_model.load_state_dict(best_model_state)
#%%
# 在测试集上评估
nn_model.eval()
with torch.no_grad():
    logits = nn_model(X_test_tensor)
    y_prob_nn = torch.sigmoid(logits).numpy().ravel()
    y_test_pred = (y_prob_nn > 0.5).astype(int)

acc = accuracy_score(y_test_model, y_test_pred)
auc = roc_auc_score(y_test_model, y_prob_nn)
brier = brier_score_loss(y_test_model, y_prob_nn)

print(f"测试集准确率: {acc:.4f}")
print(f"测试集AUC: {auc:.4f}")
print(f"测试集Brier分数: {brier:.4f}")

# --- 校准曲线 ---
prob_true, prob_pred = calibration_curve(y_test_model, y_prob_nn, n_bins=10)


# =========================
# 9️⃣ 保存模型
# 保存最佳模型的权重
torch.save(nn_model.state_dict(), "nn_model.pth")

# 保存输入维度、特征名等元信息
import joblib
nn_info = {
    "input_dim": X_train_model.shape[1],
    "feature_names": list(X_train_model.columns)
}
joblib.dump(nn_info, "nn_info.pkl")
results_nn = evaluate_model(y_test_model, y_prob_nn, model_name= nn_model)
#%% md
## 7. 保存模型
#%%
# 保存模型
import joblib
joblib.dump(best_log_reg, "lr_model.pkl")   # best_model是你训练好的模型
#%%
joblib.dump(best_rf, "rf_model.pkl")
#%%
joblib.dump(best_xgb, "xgb_model.pkl")
joblib.dump(best_svm, "svm_model.pkl")
joblib.dump(nb_model, "nb_model.pkl")
#%%
torch.save(nn_model.state_dict(), "nn_model.pth")




#%% md
### 8.2 AUC曲线
#%%
import torch
import torch.nn.functional as F
import numpy as np

# 1️⃣ 定义模型列表
sklearn_models = [best_log_reg, best_rf, best_xgb, best_svm, nb_model]
pytorch_models = [nn_model]  # 如果没有 PyTorch 模型可以写 []

# 模型对应名字
model_names = ["LR", "RF", "XGB", "SVM", "NB", "NN"]  # 顺序和上面列表对应

# 2️⃣ 初始化字典存放预测概率
y_probs = dict()

# 3️⃣ 处理 sklearn 模型
for skl_model, name in zip(sklearn_models, model_names[:len(sklearn_models)]):
    probs = skl_model.predict_proba(X_test_model)[:, 1]  # 二分类取正类概率
    y_probs[name] = probs

# 4️⃣ 处理 PyTorch 模型
for nn_model, name in zip(pytorch_models, model_names[len(sklearn_models):]):
    nn_model.eval()
    with torch.no_grad():
        logits = nn_model(X_test_tensor)  # 输出 logits
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)  # 二分类
        # 如果是多分类:
        # probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    y_probs[name] = probs

# 5️⃣ 现在 y_probs 字典里包含所有模型预测概率
print("预测概率示例：")
for name, probs in y_probs.items():
    print(f"{name}: {probs[:5]}")  # 打印前5个

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(8, 6))

# 循环所有模型
for name, probs in y_probs.items():
    fpr, tpr, _ = roc_curve(y_test_model, probs)  # 计算 FPR 和 TPR
    model_auc = auc(fpr, tpr)                     # 计算 AUC
    plt.plot(fpr, tpr, label=f"{name} (AUC = {model_auc:.3f})")

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')

# ✅ 统一标题格式
plt.title(
    'ROC Curves for All Models (6 Variables) on the Test Cohort',
    fontsize=14, fontweight='bold', pad=15, loc='center'
)

plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig("roc_curves_test.png", dpi=300, bbox_inches='tight')
plt.show()
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import numpy as np

plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# 调色板（最多支持 10 种模型，Set2 很柔和）
colors = sns.color_palette("Set2", n_colors=len(y_probs))

# 循环绘制每个模型
for i, (name, probs) in enumerate(y_probs.items()):
    fpr, tpr, _ = roc_curve(y_test_model, probs)
    model_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label=f"{name} (AUC = {model_auc:.3f})",
             color=colors[i % len(colors)],
             linewidth=2.2,
             linestyle='-',
             alpha=0.9,
             marker='o',
             markevery=int(len(fpr)/10))  # 每 10% 取一个点显示 marker

# 对角线
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.8, label='Random')

# 坐标轴美化
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')

# 标题
plt.title(
    'ROC Curves for All Models (6 Variables) on the Test Cohort',
    fontsize=14, fontweight='bold', pad=15, loc='center'
)

plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)

# 保存图像
plt.savefig("roc_curves_test_modified.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
y_probs = {
    "LR": y_pred_proba_lr,
    "RF": y_test_pred_rf,
    "XGBoost": y_test_pred_xgb,
    "SVM": y_test_pred_svm,
    "NB": y_prob_nb ,
    "NN": y_prob_nn
}


#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import gaussian_kde

def plot_calibration(y_true, y_pred, n_bins=10, smooth_frac=0.3, n_bootstraps=200,
                     bin_strategy="quantile", plot=True, save_path=None, title="Calibration plot"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- 1. 基础指标 ---
    brier = brier_score_loss(y_true, y_pred)
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=bin_strategy)

    eps = 1e-15
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    logit_pred = np.log(y_pred_clip / (1 - y_pred_clip))
    lr = LogisticRegression(fit_intercept=True, solver='lbfgs')
    lr.fit(logit_pred.reshape(-1,1), y_true)
    intercept = lr.intercept_[0]
    slope = lr.coef_[0][0]

    observed = np.sum(y_true)
    expected = np.sum(y_pred)
    oe_ratio = observed / expected

    smooth_points = np.linspace(0,1,100)
    smoothed = lowess(y_true, y_pred, frac=smooth_frac, return_sorted=True)
    smooth_mean = np.interp(smooth_points, smoothed[:,0], smoothed[:,1])
    ici = np.mean(np.abs(smooth_mean - smooth_points))

    # --- 2. Bootstrap CI ---
    metrics_boot = {"intercept": [], "slope": [], "oe_ratio": [], "ici": []}
    smooth_vals = np.zeros((n_bootstraps, len(smooth_points)))
    for i in range(n_bootstraps):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        yt_bs = y_true[idx]
        yp_bs = y_pred[idx]

        yp_bs_clip = np.clip(yp_bs, eps, 1 - eps)
        logit_bs = np.log(yp_bs_clip / (1 - yp_bs_clip))
        lr_bs = LogisticRegression(fit_intercept=True, solver='lbfgs')
        lr_bs.fit(logit_bs.reshape(-1,1), yt_bs)
        metrics_boot["intercept"].append(lr_bs.intercept_[0])
        metrics_boot["slope"].append(lr_bs.coef_[0][0])
        metrics_boot["oe_ratio"].append(np.sum(yt_bs)/np.sum(yp_bs))

        smoothed_bs = lowess(yt_bs, yp_bs, frac=smooth_frac, return_sorted=True)
        smooth_bs_vals = np.interp(smooth_points, smoothed_bs[:,0], smoothed_bs[:,1])
        smooth_vals[i,:] = smooth_bs_vals
        metrics_boot["ici"].append(np.mean(np.abs(smooth_bs_vals - smooth_points)))

    def ci(vals):
        return np.percentile(vals, [2.5, 97.5])
    metrics_ci = {k: ci(v) for k,v in metrics_boot.items()}

    smooth_lower = np.percentile(smooth_vals, 2.5, axis=0)
    smooth_upper = np.percentile(smooth_vals, 97.5, axis=0)

### log reg
y_pred_log = best_log_reg.predict_proba(X_test_model)[:, 1]
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test_model, y_pred_log)
print("内部验证 AUC:", auc)
#%%
plot_calibration(y_test_model, y_pred_log,save_path="logit_calibration.svg",title=" calibration curve of the logistic regression model in the test cohort")
#%%
### rf
y_pred_rf = best_rf.predict_proba(X_test_model)[:, 1]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test_model, y_pred_rf)
print("内部验证 AUC:", auc)
#%%
plot_calibration(y_test_model, y_pred_rf,save_path="rf_calibration.svg",title="calibration curve of the random forest model in the test cohort")
#%%
### xgb
y_pred_xgb = best_xgb.predict_proba(X_test_model)[:, 1]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test_model, y_pred_xgb)
print("内部验证 AUC:", auc)
#%%
plot_calibration(y_test_model, y_pred_xgb,save_path="xgb_calibration.svg",title="calibration curve of the XGBoost model in the test cohort")
#%%
### svc
y_pred_svc = best_svm.predict_proba(X_test_model)[:, 1]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test_model, y_pred_svc)
print("内部验证 AUC:", auc)
#%%
plot_calibration(y_test_model, y_pred_svc,save_path="svc_calibration.svg",title="calibration curve of the SVM model in the test cohort")
#%%
### nb
y_pred_nb = nb_model.predict_proba(X_test_model)[:, 1]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test_model, y_pred_svc)
print("内部验证 AUC:", auc)
#%%
plot_calibration(y_test_model, y_pred_nb,save_path="nb_calibration.svg",title="calibration curve of the Naive Bayes model in the test cohort")
#%%
### nn
import torch.nn.functional as F
def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X.values, dtype=torch.float32))
        probs = torch.sigmoid(outputs).numpy().ravel()  # 概率
    return probs
#%%
y_pred_nn = predict_proba(nn_model, X_test_model)  # 返回正类的概率
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test_model, y_pred_nn)
print("AUC:", auc)
#%%
plot_calibration(y_test_model, y_pred_nn,save_path="nn_calibration.svg",title="calibration curve of the neural network model in the test cohort")
#%% md

# =============================
###9. 简易模型构建
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

# =========================
# 1. 定义原始变量
# =========================
continuous_vars = ['GGT', 'TP', 'SG', 'TT4', 'BMI', 'HDL', 'LYM%', 'NEU', 'HCT',
                   'RDW', 'age', 'HBA1C', 'ALB', 'Cr', 'Scr',
                   'mALB_uCR', 'eGFR']
categorical_vars = ['Sex']
# =========================
# 2. 之前选出的变量（最终建模用）
# =========================
selected_features = ['GGT', 'TP', 'SG', 'TT4', 'BMI', 'HDL', 'LYM%', 'NEU', 'HCT',
                   'RDW', 'age', 'HBA1C']   # 包括编码后的列

# =========================
# 3. 识别 selected_features 中涉及的连续 & 分类变量
# =========================
# 先把还在 continuous_vars 里的挑出来
cont_for_model = [col for col in selected_features if col in continuous_vars]

# 再识别出 selected_features 中的分类变量编码（比如 sex_1 -> sex）
cat_for_model = [col for col in categorical_vars if any(col in feat for feat in selected_features)]

print("本次建模使用的连续变量:", cont_for_model)
print("本次建模使用的分类变量:", cat_for_model)

# =========================
# 4. 编码 + 标准化
# =========================
cat_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

# --- 分类变量编码 ---
X_train_cat = pd.DataFrame(
    cat_encoder.fit_transform(train_df[cat_for_model]),
    columns=cat_encoder.get_feature_names_out(cat_for_model),
    index=train_df.index
)

X_test_cat = pd.DataFrame(
    cat_encoder.transform(test_df[cat_for_model]),
    columns=cat_encoder.get_feature_names_out(cat_for_model),
    index=test_df.index
)

# --- 连续变量标准化 ---
scaler = StandardScaler()
X_train_num = pd.DataFrame(
    scaler.fit_transform(train_df[cont_for_model]),
    columns=cont_for_model,
    index=train_df.index
)
X_test_num = pd.DataFrame(
    scaler.transform(test_df[cont_for_model]),
    columns=cont_for_model,
    index=test_df.index
)

# =========================
# 5. 合并，并只保留 selected_features
# =========================
X_train_encoded = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_encoded  = pd.concat([X_test_num, X_test_cat], axis=1)

X_train_model_simple = X_train_encoded[selected_features].copy()
X_test_model_simple  = X_test_encoded[selected_features].copy()

print("最终训练集形状:", X_train_model_simple.shape)
print("最终测试集形状:", X_test_model_simple.shape)
#%%
y_train_model_simple = train_df["CKD"].copy()
y_test_model_simple = test_df["CKD"].copy()
#%%
## 保存数据标准化参数用于之后外部队列
data_dict_simple = {
    "scaler": scaler,
    "encoder": cat_encoder,
    "selected_features": selected_features,
    "cont_for_model": cont_for_model,
    "cat_for_model": cat_for_model
}
#%%
joblib.dump(data_dict_simple, "preprocessing_simple.pkl")
#%%
X_train_model_simple.to_csv("X_train_simple.csv", index=False)
X_test_model_simple.to_csv("X_test_simple.csv", index=False)
y_train_model_simple.to_csv("y_train_simple.csv", index=False)
y_test_model_simple.to_csv("y_test_simple.csv", index=False)
#%% md
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def decision_curve_analysis(y_true, y_pred_prob, thresholds=np.linspace(0.01, 0.7, 100)):
    """
    决策曲线分析 (Decision Curve Analysis, DCA)
    """
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    n = len(y_true)
    prevalence = np.mean(y_true)

    net_benefits = []
    for pt in thresholds:
        y_pred = (y_pred_prob >= pt).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        NB_model = (TP / n) - (FP / n) * (pt / (1 - pt))
        NB_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        NB_none = 0
        net_benefits.append([pt, NB_model, NB_all, NB_none])

    results = pd.DataFrame(net_benefits, columns=["threshold", "NB_model", "NB_all", "NB_none"])
    return results


def plot_dca_multi(results_dict, title="Decision Curve Analysis", save_path=None):
    """
    绘制并保存多模型 DCA 曲线（所有曲线实线 + 不同颜色区分）
    """
    plt.figure(figsize=(8, 6))

    # 使用 matplotlib 自带的调色板
    colors = plt.cm.tab20.colors  # tab20 可以支持最多 20 种颜色

    # 画每个模型的曲线
    for i, (model_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        plt.plot(results["threshold"], results["NB_model"],
                 label=model_name, color=color, linestyle="-", linewidth=2)

    # 基准线（也用实线，不同颜色）
    sample_results = list(results_dict.values())[0]
    plt.plot(sample_results["threshold"], sample_results["NB_all"],
             label="Treat All", color="red", linestyle="-", linewidth=2)
    plt.plot(sample_results["threshold"], sample_results["NB_none"],
             label="Treat None", color="black", linestyle="-", linewidth=2)

    # 美化
    plt.xlabel("Threshold Probability", fontsize=12)
    plt.ylabel("Net Benefit", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图已保存到: {save_path}")

    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_dca_multi(results_dict, title="Decision Curve Analysis new", save_path=None):
    """
    绘制多模型 DCA 曲线（优化颜色、线型和标记）
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(9, 6))

    # 调色板
    colors = sns.color_palette("Set2", n_colors=len(results_dict))

    # 绘制每个模型
    for i, (model_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        plt.plot(results["threshold"], results["NB_model"],
                 label=model_name,
                 color=color,
                 linewidth=2.2,
                 linestyle='-',
                 marker='o',
                 markersize=5,
                 alpha=0.85)

        # 可选：为每条曲线增加轻微透明填充区域（上下 5%）
        plt.fill_between(
            results["threshold"],
            results["NB_model"] * 0.95,
            results["NB_model"] * 1.05,
            color=color,
            alpha=0.1
        )

    # 基准线：Treat All / Treat None
    sample_results = list(results_dict.values())[0]
    plt.plot(sample_results["threshold"], sample_results["NB_all"],
             label="Treat All", color="red", linestyle='--', linewidth=2.0)
    plt.plot(sample_results["threshold"], sample_results["NB_none"],
             label="Treat None", color="black", linestyle='--', linewidth=2.0)

    # 美化坐标轴和标题
    plt.xlabel("Threshold Probability", fontsize=13)
    plt.ylabel("Net Benefit", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=11)
    plt.ylim(bottom=min(sample_results["NB_none"].min(), 0))  # 从0或更低开始

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ DCA 图已保存到: {save_path}")

    plt.show()
### 计算结果
results_decision_lr = decision_curve_analysis(y_test_model, y_pred_log)
results_decision_rf = decision_curve_analysis(y_test_model, y_pred_rf)
results_decision_xgb = decision_curve_analysis(y_test_model, y_pred_xgb)
results_decision_svc = decision_curve_analysis(y_test_model, y_pred_svc)
results_decision_nb = decision_curve_analysis(y_test_model, y_pred_nb)
results_decision_nn = decision_curve_analysis(y_test_model, y_pred_nn)
#%%
results_dict = {
    "Logistic Regression": results_decision_lr,
    "Random Forest": results_decision_rf,
    "XGBoost": results_decision_xgb,
    "SVM": results_decision_svc,
    "Naive Bayes": results_decision_nb,
    "Neural Network": results_decision_nn
}

plot_dca_multi(results_dict,
               title="DCA curve of multiple models (6 variables) on test cohort",
               save_path="multi_model_dca_test.png")
### 9.1 逻辑回归
#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, classification_report
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt

# 1. 定义参数网格
param_grid = {
    "penalty": ["l1", "l2"],             # 正则化方式
    "C": np.logspace(-3, 2, 10),         # 正则化强度 (0.001 ~ 100)
    "solver": ["liblinear", "saga"],     # 支持 L1 和 L2 的 solver
    "class_weight": [None, "balanced"]   # 类别不平衡处理
}

# 2. 使用 StratifiedKFold 保证类别平衡
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. GridSearchCV
log_reg = LogisticRegression(max_iter=5000, random_state=42)
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring="roc_auc",   # 以 AUC 作为优化目标
    cv=cv,
    n_jobs=1,
    verbose=2
)

# 4. 拟合
grid_search.fit(X_train_model_simple, y_train_model_simple)

print("最佳参数:", grid_search.best_params_)
print("最佳 AUC (CV):", grid_search.best_score_)

# 5. 用最佳模型预测测试集
best_log_reg_simple = grid_search.best_estimator_
y_pred_simple_lr= best_log_reg_simple.predict(X_test_model_simple)
y_pred_proba_simple_lr = best_log_reg_simple.predict_proba(X_test_model_simple)[:, 1]
#%%
results_lr = evaluate_model(y_test_model_simple, y_pred_proba_simple_lr, model_name=best_log_reg_simple)
#%% md
### 9.2 Random Forest
#%%
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 8, 12, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5]
}

rf = RandomForestClassifier(random_state=42, n_jobs=1)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=1
)
grid_search.fit(X_train_model_simple, y_train_model_simple)

print("最佳参数:", grid_search.best_params_)
print("最佳AUC:", grid_search.best_score_)
#%%
# 1. 获取调参后的最佳模型
best_rf_simple = grid_search.best_estimator_

# 2. 预测概率
y_pred_proba_simple_rf = best_rf_simple.predict_proba(X_test_model_simple)[:, 1]
#%%
results_rf = evaluate_model(y_test_model_simple, y_pred_proba_simple_rf, model_name= best_rf_simple)
#%% md
### 9.3 XGBoost
#%%
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# 1. 定义参数搜索空间
param_grid = {
    "n_estimators": [200, 500, 800],   # 树的数量
    "max_depth": [3, 5, 7],            # 树的最大深度
    "learning_rate": [0.01, 0.05, 0.1],# 学习率
    "subsample": [0.7, 0.8, 1.0],      # 每棵树的样本采样比例
    "colsample_bytree": [0.7, 0.8, 1.0]# 每棵树的特征采样比例
}

# 2. 初始化模型
xgb_clf = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss",
)

# 3. 网格搜索
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=1
)

grid_search.fit(X_train_model, y_train_model)

# 4. 打印最优参数和最优分数
print("最佳参数:", grid_search.best_params_)
print("最佳AUC:", grid_search.best_score_)
#%%
# 5. 拿到最优模型
best_xgb_simple = grid_search.best_estimator_

# 6. 预测
y_pred_proba_simple_xgb  = best_xgb_simple.predict_proba(X_test_model_simple)[:, 1]
from sklearn.metrics import roc_auc_score
auc_test = roc_auc_score(y_test_model_simple, y_test_pred_xgb)
print("测试集 AUC:", auc_test)
#%%
results_xgb = evaluate_model(y_test_model_simple, y_pred_proba_simple_xgb, model_name= best_xgb_simple)
#%% md
### 9.4 SVC
#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# =========================
# 1. 构建 Pipeline（关键）
# =========================
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        probability=True,
        class_weight='balanced',
        random_state=42
    ))
])

# =========================
# 2. 合理的参数网格
# =========================
param_grid = [
    # Linear SVM（医学筛查常常更好）
    {
        'svm__kernel': ['linear'],
        'svm__C': [0.01, 0.1, 1, 10]
    },
    # RBF SVM
    {
        'svm__kernel': ['rbf'],
        'svm__C': [0.1, 1, 10],
        'svm__gamma': [0.001, 0.01, 0.1, 'scale']
    }
]

# =========================
# 3. GridSearch（AUC）
# =========================
grid = GridSearchCV(
    pipe,
    param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

grid.fit(X_train_model, y_train_model)

print("最佳参数:", grid.best_params_)
print("最佳 AUC:", grid.best_score_)

best_svm_simple = grid.best_estimator_

y_pred_proba_simple_svm  = best_svm_simple.predict_proba(X_test_model_simple)[:, 1]
#%%
results_svm = evaluate_model(y_test_model_simple, y_pred_proba_simple_svm, model_name= best_svm_simple)
#%% md
### 9.5 nb
#%%
# 导入库
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 构建 Naive Bayes 模型
nb_model_simple = GaussianNB()
nb_model_simple.fit(X_train_model_simple, y_train_model_simple)

# 预测
y_pred_nb = nb_model_simple.predict(X_test_model_simple)
y_prob_nb_simple = nb_model_simple.predict_proba(X_test_model_simple)[:, 1]  # 概率，用于 ROC
#%%
results_nb = evaluate_model(y_test_model_simple, y_prob_nb_simple, model_name=nb_model_simple)
#%% md
### 9.6 神经网络
### 6.5 朴素贝叶斯
# --- 6.5 Naive Bayes (with hyperparameter tuning) ---

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np

# 1️⃣ 定义参数搜索范围
param_grid_nb = {
    "var_smoothing": np.logspace(-12, -6, 7)
}

# 2️⃣ 初始化模型
nb = GaussianNB()

# 3️⃣ Grid Search + 10-fold CV
grid_nb = GridSearchCV(
    estimator=nb,
    param_grid=param_grid_nb,
    cv=10,
    scoring="roc_auc",
    n_jobs=-1
)

# 4️⃣ 在训练集上拟合
grid_nb.fit(X_train_model, y_train_model)

# 5️⃣ 输出最佳参数
print("最佳参数:", grid_nb.best_params_)
print("最佳AUC:", grid_nb.best_score_)

# 6️⃣ 取最佳模型
nb_model_simple = grid_nb.best_estimator_

# -----------------------
# 用最佳模型进行预测
# -----------------------

# 预测
y_pred_nb = nb_model_simple.predict(X_test_model_simple)
y_prob_nb_simple = nb_model_simple.predict_proba(X_test_model_simple)[:, 1]  # 概率，用于 ROC

#%%
results_nb = evaluate_model(y_test_model_simple, y_prob_nb_simple, model_name=nb_model_simple)

#%% md
### 9.6 神经网络
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np

# 转换为 tensor
X_train_tensor = torch.tensor(X_train_model.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_model.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_model.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_model.values, dtype=torch.float32).view(-1, 1)

# 划分训练集 & 验证集
dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义改进版模型
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # 不要 sigmoid，这里输出 logits
        return x

nn_model_simple = Net(X_train_model.shape[1])

# 损失函数 + 优化器 + 学习率调度
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# 提前停止参数
best_val_loss = float("inf")
patience, trials = 10, 0

# 训练
for epoch in range(200):  # 最多 200 epoch
    nn_model_simple.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = nn_model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    # 验证
    nn_model_simple.eval()
    val_losses = []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            y_val_pred = nn_model(X_val)
            val_loss = criterion(y_val_pred, y_val)
            val_losses.append(val_loss.item())
    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trials = 0
        best_model_state = nn_model.state_dict()
    else:
        trials += 1
        if trials >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 恢复最佳模型
nn_model_simple.load_state_dict(best_model_state)
#%%
# 在测试集上评估
# --- 测试集评估 ---
nn_model_simple.eval()
with torch.no_grad():
    logits = nn_model_simple(X_test_tensor)
    # 使用概率而不是硬分类
    y_pred_proba_simple_nn = torch.sigmoid(logits).numpy().ravel()
    y_test_pred_simple = (y_pred_proba_simple_nn > 0.5).astype(int)

# 指标计算
acc = accuracy_score(y_test_model, y_test_pred_simple)               # Accuracy 用硬分类
auc = roc_auc_score(y_test_model, y_pred_proba_simple_nn)            # AUC 用概率
brier = brier_score_loss(y_test_model, y_pred_proba_simple_nn)       # Brier score 用概率

print(f"测试集准确率: {acc:.4f}")
print(f"测试集AUC: {auc:.4f}")
print(f"测试集Brier分数: {brier:.4f}")

# --- 校准曲线 ---
prob_true, prob_pred = calibration_curve(y_test_model, y_pred_proba_simple_nn, n_bins=10)

#%%
# 保存最佳模型的权重
torch.save(nn_model_simple.state_dict(), "nn_model_simple.pth")

# 保存输入维度、特征名等元信息
import joblib
nn_info_simple = {
    "input_dim": X_train_model_simple.shape[1],
    "feature_names": list(X_train_model_simple.columns)
}
joblib.dump(nn_info_simple, "nn_info_simple.pkl")
#%%
results_nn = evaluate_model(y_test_model_simple,  y_pred_proba_simple_nn, model_name= nn_model_simple)
#%% md
## 10. 保存简易模型
#%%
import joblib

joblib.dump(best_log_reg_simple, "lr_model_simple.pkl")  # best_model是你训练好的模型
joblib.dump(best_rf_simple, "rf_model_simple.pkl")
joblib.dump(best_xgb_simple, "xgb_model_simple.pkl")
joblib.dump(best_svm_simple, "svm_model_simple.pkl")
joblib.dump(nb_model_simple, "nb_model_simple.pkl")
#%%
torch.save(nn_model_simple.state_dict(), "nn_model_simple.pth")
#%% md
## 11. 简易模型验证
#%% md
### 11.1 auc
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 假设已经有以下变量
# y_lr, y_rf, y_xgb, y_svc, y_nb, y_nn
# y_ext 是外部验证集的真实标签

# 把各模型的预测概率放进字典
y_probs = {
    "LR": y_pred_proba_simple_lr,
    "RF": y_pred_proba_simple_rf,
    "XGB": y_pred_proba_simple_xgb,
    "SVM": y_pred_proba_simple_svm,
    "NB": y_prob_nb_simple,
    "NN": y_pred_proba_simple_nn
}

# 开始画 ROC

#%% md
### 11.2 calibration
#%%
### 原来你训练好的 best_log_reg 已存在
### 在此基础上做概率校准（prefit）

from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# 假设 X_train_model, y_train_model, X_test_model, y_test_model_simple, best_log_reg, plot_calibration
# 变量和函数已在代码运行环境中定义
calibrated_lr = CalibratedClassifierCV(
    estimator=best_log_reg,
    method='sigmoid',  # 或 'isotonic'
    cv='prefit'
)

calibrated_lr.fit(X_train_model, y_train_model)   # 用训练集进行校准

# 关键：使用校准后的概率，而不是旧概率
y_pred_cal = calibrated_lr.predict_proba(X_test_model)[:, 1]

# 关键：使用 y_pred_cal，而不是 y_pred_proba_simple_lr


from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 校準模型



# 修改 plot_calibration 函數示例（SCI風格）
def plot_calibration_1(y_true, y_pred, n_bins=10, smooth_frac=0.3, n_bootstraps=200,
                     bin_strategy="quantile", plot=True, save_path=None, title="Calibration plot"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- 1. 基础指标 ---
    brier = brier_score_loss(y_true, y_pred)
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=bin_strategy)

    eps = 1e-15
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    logit_pred = np.log(y_pred_clip / (1 - y_pred_clip))
    lr = LogisticRegression(fit_intercept=True, solver='lbfgs')
    lr.fit(logit_pred.reshape(-1,1), y_true)
    intercept = lr.intercept_[0]
    slope = lr.coef_[0][0]

    observed = np.sum(y_true)
    expected = np.sum(y_pred)
    oe_ratio = observed / expected

    smooth_points = np.linspace(0,1,100)
    smoothed = lowess(y_true, y_pred, frac=smooth_frac, return_sorted=True)
    smooth_mean = np.interp(smooth_points, smoothed[:,0], smoothed[:,1])
    ici = np.mean(np.abs(smooth_mean - smooth_points))

    # --- 2. Bootstrap CI ---
    metrics_boot = {"intercept": [], "slope": [], "oe_ratio": [], "ici": []}
    smooth_vals = np.zeros((n_bootstraps, len(smooth_points)))
    for i in range(n_bootstraps):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        yt_bs = y_true[idx]
        yp_bs = y_pred[idx]

        yp_bs_clip = np.clip(yp_bs, eps, 1 - eps)
        logit_bs = np.log(yp_bs_clip / (1 - yp_bs_clip))
        lr_bs = LogisticRegression(fit_intercept=True, solver='lbfgs')
        lr_bs.fit(logit_bs.reshape(-1,1), yt_bs)
        metrics_boot["intercept"].append(lr_bs.intercept_[0])
        metrics_boot["slope"].append(lr_bs.coef_[0][0])
        metrics_boot["oe_ratio"].append(np.sum(yt_bs)/np.sum(yp_bs))

        smoothed_bs = lowess(yt_bs, yp_bs, frac=smooth_frac, return_sorted=True)
        smooth_bs_vals = np.interp(smooth_points, smoothed_bs[:,0], smoothed_bs[:,1])
        smooth_vals[i,:] = smooth_bs_vals
        metrics_boot["ici"].append(np.mean(np.abs(smooth_bs_vals - smooth_points)))

    def ci(vals):
        return np.percentile(vals, [2.5, 97.5])
    metrics_ci = {k: ci(v) for k,v in metrics_boot.items()}

    smooth_lower = np.percentile(smooth_vals, 2.5, axis=0)
    smooth_upper = np.percentile(smooth_vals, 97.5, axis=0)

    # --- 3. 绘图（SCI顶刊风格） ---
    if plot:
        import matplotlib as mpl

        # === SCI风格参数 ===
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        mpl.rcParams['font.size'] = 11

        main_color = "#2C3E50"  # 深蓝灰（主色）
        accent_color = "#E64B35"  # 红橙（平滑曲线）
        bar_color = "#4DBBD5"  # 蓝青（分箱）
        scatter_color = "#7F7F7F"  # 灰色（散点）
        ci_color = "#E64B35"  # CI阴影

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(8, 10),
            gridspec_kw={'height_ratios': [3, 1]}
        )

        # ===== 上半部分：Calibration curve =====

        # 分箱（改颜色）
        bin_width = 0.04
        ax1.bar(prob_pred, prob_true,
                width=bin_width,
                color=bar_color,
                alpha=0.6,
                label='Binned')

        # 平滑曲线（改为红橙色）
        ax1.plot(smooth_points, smooth_mean,
                 color=accent_color,
                 linewidth=2.5,
                 label='Smoothed')

        # CI阴影（更淡）
        ax1.fill_between(smooth_points, smooth_lower, smooth_upper,
                         color=ci_color,
                         alpha=0.15)

        # 理想线（更淡）
        ax1.plot([0, 1], [0, 1],
                 linestyle='--',
                 color='black',
                 linewidth=1,
                 alpha=0.7,
                 label='Ideal')

        # 散点（灰色、低透明度）
        ax1.scatter(y_pred, y_true,
                    alpha=0.2,
                    c=scatter_color,
                    s=15,
                    label='Predictions')

        ax1.set_xlabel('Predicted probability')
        ax1.set_ylabel('Observed frequency')
        ax1.set_title(title, fontweight='bold', pad=10)

        # 网格（SCI风格：虚线）
        ax1.grid(True, linestyle=':', alpha=0.5)

        ax1.legend(loc='upper left', frameon=False)

        # ===== 右侧指标（保留 + 优化）=====
        textstr = '\n'.join((
            f"Brier = {brier:.3f}",
            f"CITL = {intercept:.2f} [{metrics_ci['intercept'][0]:.2f}, {metrics_ci['intercept'][1]:.2f}]",
            f"O/E = {oe_ratio:.2f} [{metrics_ci['oe_ratio'][0]:.2f}, {metrics_ci['oe_ratio'][1]:.2f}]",
            f"Slope = {slope:.2f} [{metrics_ci['slope'][0]:.2f}, {metrics_ci['slope'][1]:.2f}]",
            f"ICI = {ici:.3f} [{metrics_ci['ici'][0]:.3f}, {metrics_ci['ici'][1]:.3f}]"
        ))

        ax1.text(
            1.02, 0.5, textstr,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment='center',
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#F7F7F7",  # 淡灰背景（SCI常见）
                edgecolor="#B0B0B0",
                alpha=0.9
            )
        )

        # ===== 下半部分：密度 =====

        kde = gaussian_kde(y_pred)
        x_grid = np.linspace(0, 1, 200)

        ax2.plot(x_grid, kde(x_grid),
                 color=main_color,
                 linewidth=2)

        ax2.fill_between(x_grid, 0, kde(x_grid),
                         color=main_color,
                         alpha=0.2)

        ax2.set_xlabel('Predicted probability')
        ax2.set_ylabel('Density')

        ax2.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=600)  # ⭐ 从300 → 600

        plt.show()



# 使用新的 SCI 風格函數畫圖
plot_calibration_1(
    y_test_model_simple,
    y_pred_cal,
    save_path="logit_calibration_simple.svg",
    title="Calibration curve of the LR model in the cohort"
)

# 【重要修复区域】
# 1. 访问列表中的第一个校准后的模型对象
final_calibrated_object = calibrated_lr.calibrated_classifiers_[0]

# 2. 访问原始模型的 Beta 和 Intercept (这部分是安全的)
# 这将提取出用于计算 Log-Odds_raw 的系数，即您之前手动验证时使用的那组系数。
raw_beta = final_calibrated_object.estimator.coef_[0]
raw_intercept = final_calibrated_object.estimator.intercept_[0]

# 3. 计算校准器的 A 和 B (通过 Log-Odds 倒推)
# 这一步是关键：我们通过计算训练集上的原始Log-Odds和校准后的Log-Odds，来拟合出 A 和 B。

# 计算训练集上的原始 Log-Odds (Log-Odds_raw)
raw_log_odds_train = raw_intercept + np.dot(X_train_model, raw_beta)

# 拟合一个新的线性回归模型来找到 A 和 B：
# Logit(P_calibrated) = A * Logit(P_raw) + B
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer

# P_raw: 原始模型的预测概率
P_raw_train = best_log_reg.predict_proba(X_train_model)[:, 1]

# P_calibrated: 理论上，Logit(P_calibrated) 应该近似等于 A * Logit(P_raw) + B
# 由于我们没有直接的 P_calibrated，我们使用目标变量 y_train_model 来拟合 A 和 B
# 这是 Platt Scaling (Sigmoid) 的本质。

# 重新训练一个校准修正模型 (Sigmoid)
calibrator_fit = LogisticRegression(C=1.0, solver='lbfgs') # 这是一个LogReg，它的 coef_ 就是A，intercept_ 就是B
calibrator_fit.fit(raw_log_odds_train.reshape(-1, 1), y_train_model)

A_slope = calibrator_fit.coef_[0][0]
B_intercept = calibrator_fit.intercept_[0]


print("--- 关键参数提取 ---")
print(f"原始模型的 Beta 系数 (raw_beta): \n{raw_beta}")
print(f"原始模型的截距 (raw_intercept): {raw_intercept}")
print(f"倒推校准器的斜率 A (A_slope): {A_slope}")
print(f"倒推校准器的截距 B (B_intercept): {B_intercept}")

import joblib

# 假设您的校准代码已经成功运行，并且 calibrated_lr 对象已存在于内存中

# 设定保存路径和文件名
save_path_calibrated = r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_lr_model_simple.pkl"

# 使用 joblib.dump() 将内存中的对象保存到磁盘
joblib.dump(calibrated_lr, save_path_calibrated)

print(f"✅ 校准后的 LR 模型已成功保存到: {save_path_calibrated}")
# 【请注意】
# 重新运行 SHAP 应该基于这个 Log-Odds_calibrated = A * Log-Odds_raw + B 的复合模型。
# ERI 简化评分的开发，应该基于这套完整的 A, B, raw_beta, raw_intercept 参数进行。

#未校准的LR calibration curve在8.3
#验证β
import numpy as np

# ----------------------------------------------------
# 步骤 1: 参数和数据
# ----------------------------------------------------
# 原始参数 (已验证 100% 正确)
raw_beta = np.array([0.08192441,  0.12241402, -0.05004919,  0.0678664,  -0.01938552,
                     -0.04418611, -0.07657063,  0.03769081, -0.13579451,  0.16301067,
                     0.4094881,   0.26937778])
#顺序：'age', 'WHR', 'BMI', 'MMSE', 'ADL', 'HCT', 'RDW', 'NEU', 'LYM%', 'TT4', 'HDL',
# 'HBAIC', 'TP', 'GGT', 'SG'
raw_intercept = -1.6916371695067673

# 校准参数 (倒推获得)
A_slope = 1.357032011938865
B_intercept = 0.4940720521140823

# ----------------------------------------------------
# 步骤 2: 定义 Logit 和 Sigmoid 转换函数
# ----------------------------------------------------
def logit(p):
    # Logit(P) = log(P / (1-P))
    # 确保 P 不为 0 或 1，避免 inf
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))

def sigmoid(logit_val):
    # P = 1 / (1 + exp(-Logit))
    return 1 / (1 + np.exp(-logit_val))

# ----------------------------------------------------
# 步骤 3: 最终验证运行 (在概率空间修正)
# ----------------------------------------------------
num_samples_to_verify = 3
X_verify = X_test_model[:num_samples_to_verify] # 假设 X_test_model 已定义

# A. 计算原始模型的 Log-Odds 和概率
raw_log_odds = raw_intercept + np.dot(X_verify, raw_beta)
P_raw = sigmoid(raw_log_odds)

# B. 转换到 Logit 空间并应用 A*x + B 修正
Logit_P_raw = logit(P_raw)
Logit_P_cal = A_slope * Logit_P_raw + B_intercept

# C. 转换回概率
manual_P_cal = sigmoid(Logit_P_cal)

# D. 对比
model_predicted_probability = calibrated_lr.predict_proba(X_verify)[:, 1][:num_samples_to_verify]
difference = np.abs(model_predicted_probability - manual_P_cal)

print("\n--- 最终概率空间修正验证结果 ---")
for i in range(num_samples_to_verify):
    print(f"--- 样本 {i+1} ---")
    print(f"模型预测概率 (calibrated_lr): {model_predicted_probability[i]:.8f}")
    print(f"手动计算概率 (P_raw修正): {manual_P_cal[i]:.8f}")
    print(f"绝对差异: {difference[i]:.8f}")

if np.all(difference < 0.0005):
    print("\n✅ 验证成功! 校准器 A 和 B 参数在概率空间修正中提取正确。")
    print("结论: ERI 赋分不能使用简单的复合 Beta'，必须使用 **P_cal = Sigmoid(A * Logit(P_raw) + B)** 公式。")
else:
    print("\n❌ 验证失败! 验证程序仍存在问题。")
#%%
plot_calibration(y_test_model_simple, y_pred_proba_simple_rf,save_path="rf_calibration_simple.svg",
                 title="calibration curve of the random forest model (15 variables) in the test cohort")
from sklearn.calibration import CalibratedClassifierCV
# 确保导入了您的 XGBoost 模型 (假设您已经训练好了 best_xgb_clf)
# import xgboost as xgb

# ----------------------------------------------------
# 关键步骤：使用相同的 CalibratedClassifierCV 流程
# ----------------------------------------------------
calibrated_rf = CalibratedClassifierCV(
    estimator=best_rf_simple,
    method='sigmoid',  # 或 'isotonic'
    cv='prefit'
)
# 用训练集进行校准 (这一步是训练校准器 A 和 B)
calibrated_rf.fit(X_train_model_simple, y_train_model_simple)

# 关键：使用校准后的概率
y_pred_rf_cal = calibrated_rf.predict_proba(X_test_model_simple)[:, 1]

# 绘制校准曲线


from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 校準模型



# 修改 plot_calibration 函數示例（SCI風格）


# 使用新的 SCI 風格函數畫圖
plot_calibration_1(
    y_test_model_simple,
    y_pred_cal,
    save_path="rf_simple.svg",
    title="Calibration curve of the RF model in the cohort"
)

# ----------------------------------------------------
# 步骤 3: 保存校准模型 (以便 SHAP 使用)
# ----------------------------------------------------
joblib.dump(calibrated_rf, "calibrated_rf_model_simple.pkl")
#%%
plot_calibration(y_test_model_simple, y_pred_proba_simple_xgb,save_path="xgb_calibration_simple.svg",
                 title="calibration curve of the XGBoost model (15 variables) in the test cohort")
from sklearn.calibration import CalibratedClassifierCV
# 确保导入了您的 XGBoost 模型 (假设您已经训练好了 best_xgb_clf)
# import xgboost as xgb

# ----------------------------------------------------
# 关键步骤：使用相同的 CalibratedClassifierCV 流程
# ----------------------------------------------------
calibrated_xgb = CalibratedClassifierCV(
    estimator=best_xgb_simple,
    method='sigmoid',
    cv='prefit'
)
# 用训练集进行校准 (这一步是训练校准器 A 和 B)
calibrated_xgb.fit(X_train_model_simple, y_train_model_simple)

# 关键：使用校准后的概率
y_pred_xgb_cal = calibrated_xgb.predict_proba(X_test_model_simple)[:, 1]

# 绘制校准曲线

from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 校準模型
# 修改 plot_calibration 函數示例（SCI風格）


# 使用新的 SCI 風格函數畫圖
plot_calibration_1(
    y_test_model_simple,
    y_pred_cal,
    save_path="xgb_simple.svg",
    title="Calibration curve of the XGBoost model in the cohort"
)
# ----------------------------------------------------
# 步骤 3: 保存校准模型 (以便 SHAP 使用)
# ----------------------------------------------------
joblib.dump(calibrated_xgb, "calibrated_xgb_model_simple.pkl")
#%%
plot_calibration(y_test_model_simple, y_pred_proba_simple_svm,save_path="svm_calibration_simple.svg",
                 title="calibration curve of the SVM model (15 variables) in the test cohort")
from sklearn.calibration import CalibratedClassifierCV
# 确保导入了您的 XGBoost 模型 (假设您已经训练好了 best_xgb_clf)
# import xgboost as xgb

# ----------------------------------------------------
# 关键步骤：使用相同的 CalibratedClassifierCV 流程
# ----------------------------------------------------
calibrated_svm = CalibratedClassifierCV(
    estimator=best_svm_simple,
    method='isotonic',
    cv='prefit'
)
# 用训练集进行校准 (这一步是训练校准器 A 和 B)
calibrated_svm.fit(X_train_model_simple, y_train_model_simple)

# 关键：使用校准后的概率
y_pred_svm_cal = calibrated_svm.predict_proba(X_test_model_simple)[:, 1]

from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 校準模型
# 修改 plot_calibration 函數示例（SCI風格）


# 绘制校准曲线
plot_calibration_1(
    y_test_model_simple,
    y_pred_svm_cal,
    save_path="svm_calibration_simple_calibrated.svg", # <--- 修正文件名
    title="calibration curve of the SVM model in the test cohort" # <--- 修正标题
)

# ----------------------------------------------------
# 步骤 3: 保存校准模型 (以便 SHAP 使用)
# ----------------------------------------------------
joblib.dump(calibrated_svm, "calibrated_svm_model_simple.pkl")
#%%
plot_calibration(y_test_model_simple, y_pred_nb,save_path="nb_calibration_simple.svg",
                 title="calibration curve of the Naive Bayes model (15 variables) in the test cohort")
from sklearn.calibration import CalibratedClassifierCV
# 确保导入了您的 XGBoost 模型 (假设您已经训练好了 best_xgb_clf)
# import xgboost as xgb

# ----------------------------------------------------
# 关键步骤：使用相同的 CalibratedClassifierCV 流程
# ----------------------------------------------------
calibrated_nb = CalibratedClassifierCV(
    estimator=nb_model_simple,
    method='isotonic',
    cv='prefit'
)
# 用训练集进行校准 (这一步是训练校准器 A 和 B)
calibrated_nb.fit(X_train_model_simple, y_train_model_simple)

# 关键：使用校准后的概率
y_pred_nb_cal = calibrated_nb.predict_proba(X_test_model_simple)[:, 1]

from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 校準模型
# 修改 plot_calibration 函數示例（SCI風格）


# 绘制校准曲线
plot_calibration_1(
    y_test_model_simple,
    y_pred_nb_cal,
    save_path="nb_calibration_simple_calibrated.svg", # <--- 修正文件名
    title="calibration curve of the Naive Bayes model in the test cohort" # <--- 修正标题
)

# ----------------------------------------------------
# 步骤 3: 保存校准模型 (以便 SHAP 使用)
# ----------------------------------------------------
joblib.dump(calibrated_nb, "calibrated_nb_model_simple.pkl")
#%%
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import numpy as np
import joblib

# ----------------------------------------------------
# ❗ 核心修正：手动 Platt Scaling ❗
# ----------------------------------------------------

# 1. 确保 X/Y 数据是 NumPy/Pandas 格式 (float32, int64)
# 假设 X_train_model, y_train_model, X_test_model, y_test_model 已经被转换为正确的 numpy 类型

# 2. 提取 PyTorch 模型的原始 Logits (raw_log_odds)
nn_model_simple.eval() # 确保在评估模式
with torch.no_grad():
    # 确保输入是 PyTorch tensor
    X_train_tensor = torch.tensor(X_train_model.values.astype(np.float32), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_model.values.astype(np.float32), dtype=torch.float32)

    # 提取训练集上的 Logits
    raw_log_odds_train = nn_model_simple(X_train_tensor).cpu().numpy().flatten()
    # 提取测试集上的 Logits
    raw_log_odds_test = nn_model_simple(X_test_tensor).cpu().numpy().flatten()


# 3. 训练 Platt Scaling 校准器 (一个简单的 LogReg)
# X 是原始 Logits，Y 是真实的类别标签
platt_scaler = LogisticRegression(C=1.0, solver='lbfgs')

# 训练校准器：使用 Logits 修正目标 Y (y_train_model)
# 注意：X 需要是 (n_samples, 1) 形状
platt_scaler.fit(raw_log_odds_train.reshape(-1, 1), y_train_model.values.astype(np.int64).ravel())

# 提取 A 和 B
A_slope = platt_scaler.coef_[0][0]
B_intercept = platt_scaler.intercept_[0]
print(f"✅ Platt Scaling 参数提取成功: A={A_slope:.4f}, B={B_intercept:.4f}")


# 4. 计算校准后的概率 (应用于测试集)
# Logit_cal = A * Logit_raw + B
cal_log_odds_test = A_slope * raw_log_odds_test + B_intercept

# P_cal = Sigmoid(Logit_cal)
y_pred_nn_cal = 1 / (1 + np.exp(-cal_log_odds_test))


# 5. 评估和绘图 (使用 y_pred_nn_cal)

# Brier Score 校验
calibrated_brier = brier_score_loss(y_test_model_simple, y_pred_nn_cal)
print(f"✅ 校准后的 Brier Score: {calibrated_brier:.4f}")

plot_calibration_1(
    y_test_model_simple,
    y_pred_nn_cal,
    save_path="nn_calibration_simple_calibrated.svg",
    title="Calibration curve of the NN model in the test cohort"
)


# 6. 保存校准器 (platt_scaler) 而不是 CalibratedClassifierCV
joblib.dump(platt_scaler, "nn_platt_scaler_simple.pkl")

print("✅ 神经网络手动校准成功并已保存校准器。")

save_path_nn = r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_nn_info_simple.pkl"
import joblib

# 假设 X_train_model 仍然在内存中
input_dim = X_train_model.shape[1]

nn_info = {
    "input_dim": input_dim,
    "feature_count": input_dim,
    # 如果需要，可以添加更多信息，例如: "Platt_A": A_slope, "Platt_B": B_intercept
}

info_path = r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_nn_model_simple.pkl"
joblib.dump(nn_info, info_path)
# 假设 best_model_state 存储了最佳模型的权重
save_path_weights = r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_nn_model_simple.pth"
torch.save(best_model_state, save_path_weights)

#%% md
### 11.3 DCA
#%%
### 计算结果
results_decision_lr = decision_curve_analysis(y_test_model, y_pred_proba_simple_lr)
results_decision_rf = decision_curve_analysis(y_test_model, y_pred_proba_simple_rf)
results_decision_xgb = decision_curve_analysis(y_test_model, y_pred_proba_simple_xgb)
results_decision_svc = decision_curve_analysis(y_test_model, y_pred_proba_simple_svm)
results_decision_nb = decision_curve_analysis(y_test_model, y_prob_nb_simple)
results_decision_nn = decision_curve_analysis(y_test_model, y_pred_proba_simple_nn)
#%%
results_dict = {
    "LR": results_decision_lr,
    "RF": results_decision_rf,
    "XGB": results_decision_xgb,
    "SVM": results_decision_svc,
    "NB": results_decision_nb,
    "NN": results_decision_nn
}

plot_dca_multi(results_dict,
               title="DCA curves of six models (14 variables) on the test cohort",
               save_path="multi_model_dca_test.png")



#%%
import numpy as np
import matplotlib.pyplot as plt

# 假设你的六个变量
features = ['GGT', 'TP', 'SG', 'TT4', 'BMI', 'HDL', 'LYM%', 'NEU', 'HCT','RDW', 'age', 'HBA1C']
# 替换为你实际训练出的 LR 模型的系数（beta值）
coefficients = np.array([ 0.08192441*1.357032011938865,  0.12241402*1.357032011938865,
                          -0.05004919*1.357032011938865,  0.0678664*1.357032011938865,
                          -0.01938552*1.357032011938865, -0.04418611*1.357032011938865,
                          -0.07657063*1.357032011938865,  0.03769081*1.357032011938865,
                          -0.13579451*1.357032011938865,  0.16301067*1.357032011938865,
                          0.4094881*1.357032011938865,   0.26937778*1.357032011938865])
# 替换为你实际训练出的 LR 模型的截距（Intercept）
intercept = 0.4940720521140823

# 修复后的 ranges 字典（键已改为小写）
ranges = {
    'BMI': (12, 74),
    'age': (0, 81),
    'HDL': (0.1, 4.1),
    'HCT': (22, 57),
    'LYM%': (4.8, 86),
    'TT4': (19, 356),
    'HBA1C': (2, 16),
    'TP': (34, 104),
    'SG': (1, 1.2),
    'RDW': (6, 38),
    'NEU': (0.2, 21),
    'GGT': (4, 892),
}

# --- 1. 计算每个变量的得分尺度 (Point Scale) ---
linear_contributions = {}
for i, feature in enumerate(features):
    min_val = ranges[feature][0]
    max_val = ranges[feature][1]

    contribution_min = coefficients[i] * min_val
    contribution_max = coefficients[i] * max_val

    linear_contributions[feature] = (contribution_min, contribution_max)

# 找到所有变量贡献度的总跨度，用于统一 Nomogram 的 Point 尺度
all_min = min(cont[0] for cont in linear_contributions.values())
all_max = max(cont[1] for cont in linear_contributions.values())
range_span = all_max - all_min

max_point = 100

def get_point_score(feature, value):
    """根据变量值和系数计算得分 (0-100)"""
    linear_val = coefficients[features.index(feature)] * value
    point_score = ((linear_val - all_min) / range_span) * max_point
    return np.clip(point_score, 0, max_point)

# --- 2. 绘制 Nomogram 框架（SCI风格） ---

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_facecolor('white')

# === SCI配色（核心修改）===
main_color = "#2F4F4F"      # 深灰蓝（替代 black / darkgreen）
accent_color = "#D55E00"    # 强调色（橙红，用于points）
grid_color = "#BFBFBF"      # 浅灰辅助线

plt.rcParams['text.color'] = main_color
plt.rcParams['axes.labelcolor'] = main_color
plt.rcParams['xtick.color'] = main_color
plt.rcParams['ytick.color'] = main_color

ax.set_yticks([])

y_positions = np.linspace(0.8, 0.1, len(features) + 3)

# ===== 1. Points axis =====
ax.hlines(y_positions[0], 0, 100, color=main_color, linewidth=1.8)
ax.text(-5, y_positions[0], 'Points',
        fontsize=13, fontweight='bold', color=main_color)

ax.set_xticks(np.arange(0, 101, 10))
ax.set_xlim(-10, 110)

# ===== 2. 特征变量 =====
for i, feature in enumerate(features):
    y_pos = y_positions[i + 1]

    # 更柔和的辅助线（SCI常见）
    ax.hlines(y_pos, 0, 100,
              color=grid_color,
              linestyle=':',
              linewidth=0.8)

    ax.text(-5, y_pos, feature.upper(),
            fontsize=11, fontweight='semibold',
            color=main_color)

    min_val, max_val = ranges[feature]
    ticks = np.linspace(min_val, max_val, 5)

    for val in ticks:
        point = get_point_score(feature, val)

        # 刻度线（细一点）
        ax.plot(point, y_pos,
                marker='|',
                color=main_color,
                markersize=8,
                markeredgewidth=1.2)

        ax.text(point, y_pos - 0.015,
                f'{val:.1f}',
                fontsize=8,
                ha='center',
                color=main_color)

        # ⭐ 顶部 Points 用强调色（区别师兄的红色）
        ax.text(point, y_positions[0] + 0.005,
                f'{int(point)}',
                fontsize=8,
                ha='center',
                color=accent_color)

# ===== 3. Total Points =====
total_points_pos = y_positions[-2]
max_total_point = len(features) * max_point

ax.hlines(total_points_pos, 0, 100,
          color=main_color,
          linewidth=1.8)

ax.text(-5, total_points_pos,
        'Total Points',
        fontsize=13,
        fontweight='bold',
        color=main_color)

total_point_ticks = np.arange(0, max_total_point + 1, 100)
mapped_tp_ticks = (total_point_ticks / max_total_point) * 100

for tp, x_pos in zip(total_point_ticks, mapped_tp_ticks):
    ax.plot(x_pos, total_points_pos,
            marker='|',
            color=main_color,
            markersize=8,
            markeredgewidth=1.2)

    ax.text(x_pos, total_points_pos - 0.02,
            f'{int(tp)}',
            fontsize=9,
            ha='center',
            color=main_color)

# ===== 4. Risk Probability =====
risk_pos = y_positions[-1]

risk_probs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
corresponding_total_points = np.array([100, 150, 250, 300, 400, 500])
mapped_risk_x = (corresponding_total_points / max_total_point) * 100

# ⭐ 主线改成深灰蓝（不是darkgreen）
ax.hlines(risk_pos, 0, 100,
          color=main_color,
          linewidth=1.8)

ax.text(-5, risk_pos,
        'Risk Probability',
        fontsize=13,
        fontweight='bold',
        color=main_color)

for x_pos, rp in zip(mapped_risk_x, risk_probs):

    # ⭐ 空心圆（SCI常见）
    ax.plot(x_pos, risk_pos,
            marker='o',
            markerfacecolor='white',
            markeredgecolor=main_color,
            markersize=6,
            markeredgewidth=1.2)

    ax.text(x_pos, risk_pos - 0.03,
            f'{rp:.2f}',
            fontsize=9,
            ha='center',
            color=main_color)

# ===== 标题 =====
plt.title(
    'Nomogram for Predicting Risk of CKD ',
    fontsize=15,
    fontweight='bold',
    color=main_color,
    pad=15
)

plt.axis('off')

# ===== 保存（SCI关键）=====
save_path = "Nomogram_Malnutrition_Risk_SCI.png"

plt.savefig(
    save_path,
    dpi=600,   # ⭐ 从300 → 600
    bbox_inches='tight',
    facecolor='white'
)

print(f"Nomogram 已保存至: {save_path}")

plt.show()

#%% md
#13 Bootstrap
#%%
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

n_iterations = 1000
auc_scores = []

print(f"开始进行 {n_iterations} 次 Bootstrap 抽样...")
print(f"验证的Y-True变量: y_test_model_simple")
print(f"验证的Y-Pred变量: y_pred_proba_simple_lr")

for i in range(n_iterations):
    indices = resample(
        np.arange(len(y_test_model_simple)),
        random_state=i
    )

    y_true_boot = y_test_model_simple.iloc[indices]
    y_pred_boot = y_pred_proba_simple_lr[indices]
    if len(np.unique(y_true_boot)) < 2:
        continue
    boot_auc = roc_auc_score(
        y_true_boot,
        y_pred_boot
    )

    auc_scores.append(boot_auc)

mean_auc = np.mean(auc_scores)

lower_ci = np.percentile(auc_scores, 2.5)
upper_ci = np.percentile(auc_scores, 97.5)

print("\n--- Bootstrap 验证结果 ---")
print(f"模型: 校准后的逻辑回归 (calibrated_lr_model_simple)")
print(f"原始测试集AUC: {roc_auc_score(y_test_model_simple, y_pred_proba_simple_lr):.4f}")
print(f"Bootstrap 平均 AUC: {mean_auc:.4f}")
print(f"Bootstrap 95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")



#====================================