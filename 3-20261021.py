###设置工作目录
import os

# 设置工作目录为新文件夹路径
os.chdir(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\SHAP")
print("当前工作目录：", os.getcwd())

###1. 加载模型

import shap
print(shap.__version__)

#%%
## 加载之前拟合的模型
import joblib

# 逐个加载
lr_model = joblib.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_lr_model_simple.pkl")
rf_model = joblib.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_rf_model_simple.pkl")
xgb_model  = joblib.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_xgb_model_simple.pkl")
svm_model = joblib.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_svm_model_simple.pkl")
nb_model = joblib.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_nb_model_simple.pkl")

import torch
import joblib
import torch.nn as nn

# 重新定义模型结构（要和训练时完全一致）
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
        x = self.fc4(x)
        return x



# 加载模型信息
nn_info = joblib.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_nn_model_simple.pkl")
input_dim = nn_info["input_dim"]

# 初始化模型并加载参数
loaded_model = Net(input_dim)
loaded_model.load_state_dict(torch.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_nn_model_simple.pth"))
loaded_model.eval()

import joblib

# 读回来
data_dict_simple = joblib.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\preprocessing_simple.pkl")

# 直接取出
scaler = data_dict_simple["scaler"]
cat_encoder = data_dict_simple["encoder"]
selected_features = data_dict_simple["selected_features"]
cont_for_model = data_dict_simple["cont_for_model"]
cat_for_model = data_dict_simple["cat_for_model"]

###2. 训练模型
##2.1 导入测试集训练集相关参数
#%%
import pandas as pd

X_train = pd.read_csv(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\X_train_simple.csv")
X_test = pd.read_csv(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\X_test_simple.csv")
y_train = pd.read_csv(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\y_train_simple.csv").squeeze()
y_test = pd.read_csv(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\y_test_simple.csv").squeeze()

##2.2 计算SHAP值

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- 1️⃣ 加载校准后的逻辑回归模型 ---
lr_model = joblib.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_lr_model_simple.pkl")

# --- 2️⃣ 定义可调用函数返回正类概率 ---
def model_predict_proba(X):
    return lr_model.predict_proba(X)[:, 1]

# --- 3️⃣ 使用训练集抽样作为背景分布 ---
background = shap.sample(X_train, 100, random_state=42)

# --- 4️⃣ 创建 KernelExplainer ---
explainer = shap.KernelExplainer(model_predict_proba, background)

# --- 5️⃣ 计算 SHAP 值（测试集样本可适量减少，加快速度） ---
shap_values = explainer.shap_values(X_test, nsamples=200)

# --- 6️⃣ 计算每个特征平均绝对 SHAP 值 ---
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# --- 7️⃣ 构建 DataFrame 并按重要性排序 ---
importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean |SHAP value|': mean_abs_shap,
    'Mean feature value': X_test.mean().values
}).sort_values('Mean |SHAP value|', ascending=True)

# --- 8️⃣ 绘制柱状图（更新颜色和视觉效果） ---
fig, ax = plt.subplots(figsize=(9, 6))

# 使用渐变颜色
colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(importance_df)))
bars = ax.barh(importance_df['Feature'], importance_df['Mean |SHAP value|'], color=colors, edgecolor='black', alpha=0.9)

# 右侧标注 SHAP 值
for i, v in enumerate(importance_df['Mean |SHAP value|']):
    ax.text(v + 0.001, i, f"{v:.3f}", va='center', ha='left', fontsize=9, color='black')

# 左侧标注特征均值
max_shap = importance_df['Mean |SHAP value|'].max()
for i, v in enumerate(importance_df['Mean feature value']):
    ax.text(-0.12 * max_shap, i, f"{v:.2f}", va='center', ha='right', fontsize=9, color='dimgray')

# 图形美化
ax.set_xlabel("Mean |SHAP value|", fontsize=12, fontweight='bold')
ax.set_ylabel("Feature", fontsize=12, fontweight='bold')
ax.set_title("Feature Importance based on SHAP values (Logistic Regression)", fontsize=14, fontweight='bold')
plt.subplots_adjust(left=0.35, right=0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(axis='x', linestyle='--', alpha=0.4)

# 保存图像
plt.tight_layout()
plt.savefig("feature_importance_shap_calibrated_colored.svg", dpi=300, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
import shap

# --- 1️⃣ 创建 figure ---
plt.figure(figsize=(10, 6))

# --- 2️⃣ 绘制 summary_plot，不自动显示 ---
shap.summary_plot(
    shap_values,
    X_test,
    show=False,
    plot_type="dot",          # 保持默认点图，也可改为 "bar" 如果想柱状
    color=plt.cm.viridis       # 改用 viridis 渐变色
)

# --- 3️⃣ 添加标题 ---
plt.title(
    "SHAP Summary Plot (Logistic Regression)",
    fontsize=14,
    fontweight='bold',
    pad=20
)

# --- 4️⃣ 调整布局并保存 ---
plt.tight_layout()
plt.savefig("shap_summary_plot_colored.svg", dpi=300, bbox_inches='tight')
plt.show()

#%% md
## 2.3 总体特征重要性条状图
#%%
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- 1️⃣ 加载模型 ---
lr_model = joblib.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_lr_model_simple.pkl")

# --- 2️⃣ 定义概率函数 ---
def model_predict_proba(X):
    return lr_model.predict_proba(X)[:, 1]

# --- 3️⃣ 背景分布 ---
background = shap.sample(X_train, 100, random_state=42)

# --- 4️⃣ 创建 KernelExplainer ---
explainer = shap.KernelExplainer(model_predict_proba, background)

# --- 5️⃣ 计算 SHAP 值 ---
shap_values = explainer.shap_values(X_test, nsamples=200)

# --- 6️⃣ 计算 mean|SHAP| ---
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# --- 7️⃣ 排序（从大到小） ---
importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean |SHAP value|': mean_abs_shap,
}).sort_values('Mean |SHAP value|', ascending=False)

# --- 8️⃣ 全新风格：竖向单色图 ---
plt.figure(figsize=(7, 6))
bars = plt.bar(
    importance_df['Feature'],
    importance_df['Mean |SHAP value|'],
    color="steelblue",            # 单色
    edgecolor="black",            # 黑色边框（更学术）
    alpha=0.85
)

# 添加数值
for idx, val in enumerate(importance_df['Mean |SHAP value|']):
    plt.text(idx, val + 0.002, f"{val:.3f}",
             ha='center', va='bottom', fontsize=9)

plt.ylabel("Mean |SHAP value|", fontsize=12, fontweight='bold')
plt.xlabel("Feature", fontsize=12, fontweight='bold')
plt.title("Feature Importance (mean |SHAP|) — Logistic Regression", fontsize=14)

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("shap_importance_vertical_clean.png", dpi=300)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 排序 SHAP 值
order = np.argsort(mean_abs_shap)[::-1]
sorted_features = X_test.columns[order]
sorted_shap = shap_values[:, order]

plt.figure(figsize=(7, 5))

for i in range(len(sorted_features)):
    y = np.full(sorted_shap.shape[0], i)
    jitter = np.random.normal(0, 0.08, size=y.shape)   # 垂直抖动
    plt.scatter(
        sorted_shap[:, i],
        y + jitter,
        color="grey",        # 单色
        alpha=0.35,
        s=12,
        edgecolors="none"
    )

plt.yticks(range(len(sorted_features)), sorted_features)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)

plt.title("SHAP Value Distribution (Customized Visualization)", fontsize=14)
plt.xlabel("SHAP value (impact on model output)")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("shap_summary_nocolorbar_clean.png", dpi=300)
plt.show()

#%%
import seaborn as sns

shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
shap_long = shap_df.melt(var_name="Feature", value_name="SHAP value")

plt.figure(figsize=(8, 6))
sns.violinplot(
    data=shap_long,
    x="SHAP value",
    y="Feature",
    scale="width",
    inner="quartile",
    color="lightblue"
)

plt.title("Feature Importance (Violin Based on SHAP Distribution)")
plt.tight_layout()
plt.savefig("shap_violin.png", dpi=300)
plt.show()

#%%
plt.figure(figsize=(7, 6))

plt.boxplot(
    [sorted_shap[:, i] for i in range(len(sorted_features))],
    vert=False,
    labels=sorted_features,
    patch_artist=True,
    boxprops=dict(facecolor="lightgray")
)

plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("SHAP value")
plt.title("SHAP Value Distribution (Boxplot)")
plt.tight_layout()
plt.savefig("shap_boxplot.png", dpi=300)
plt.show()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 1️⃣ 计算每个特征的均值 ===
# 连续变量均值（反归一化）
cont_means = scaler.mean_

# 分类变量均值（One-hot 后）
encoded_cat_cols = cat_encoder.get_feature_names_out(cat_for_model)
cat_means = X_test[encoded_cat_cols].mean().values

# 合并均值
mean_feature_values = np.concatenate([cont_means, cat_means])

# === 2️⃣ 计算平均 SHAP 绝对值 ===
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# === 3️⃣ 构建 DataFrame 并按重要性排序 ===
importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean |SHAP value|': mean_abs_shap,
    'Mean feature value': mean_feature_values
}).sort_values('Mean |SHAP value|', ascending=True)

# === 4️⃣ 绘制双标签 SHAP 重要性图 ===
fig, ax = plt.subplots(figsize=(9, 6))

# 主条形图
bars = ax.barh(importance_df['Feature'], importance_df['Mean |SHAP value|'], color='#4C72B0')

# 右侧标注 SHAP
# 右侧标注 SHAP 值
for i, v in enumerate(importance_df['Mean |SHAP value|']):
    ax.text(v + 0.001, i, f"{v:.3f}", va='center', ha='left', fontsize=9, color='black')

# 左侧标注均值（避免与特征名重叠）
max_shap = importance_df['Mean |SHAP value|'].max()
for i, v in enumerate(importance_df['Mean feature value']):
    ax.text(-0.1 * max_shap, i, f"{v:.2f}", va='center', ha='right', fontsize=9, color='gray')

# 调整边距
ax.set_xlabel("Mean |SHAP value|")
ax.set_ylabel("Feature")
ax.set_title("Feature Importance based on SHAP values (Logistic Regression)")
plt.subplots_adjust(left=0.35, right=0.85)
plt.tight_layout()
plt.savefig("feature_inportance_shap.svg", dpi=300, bbox_inches='tight')
plt.show()
#%% md
## 2.4 10折交叉验证
#%%
# === 1️⃣ 导入库 ===
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV # 导入校准器

# 假设你已有训练好的模型对象
# 特征数据：X_train, y_train
X = X_train
y = y_train

# === 2️⃣ 模型列表 ===
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgboost
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 初始化原始模型
lr_model = LogisticRegression(max_iter=5000, random_state=42)
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
xgb_model = xgboost.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
svm_model = SVC(C=0.1, gamma=0.001, probability=True, random_state=42)
nb_model = GaussianNB()

# 定义校准方法字典
calibration_methods = {
    "LR": 'sigmoid',
    "RF": 'sigmoid',
    "XGB": 'sigmoid',
    "SVM": 'isotonic', # SVM和NB推荐Isotonic
    "NB": 'isotonic',
}

# 组合模型和校准方法
models = {
    "LR": lr_model,
    "RF": rf_model,
    "XGB": xgb_model,
    "SVM": svm_model,
    "NB": nb_model,
}


# === 3️⃣ 10折交叉验证设置 ===
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# === 4️⃣ 记录指标 ===
cv_results = {name: {"AUC": [], "Accuracy": [], "Brier": []} for name in models.keys()}

# === 5️⃣ 交叉验证循环 (修正区域) ===
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    for name, model in models.items():
        # 获取该模型对应的校准方法
        method = calibration_methods[name]

        # 核心：在每一折上，重新创建 CalibratedClassifierCV
        # cv=2 表示校准器自身使用 2 折内部CV，以确保其鲁棒性。
        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv=2 # 使用内部CV，更稳定
        )

        # 拟合校准模型 (这一步同时训练了 model 和校准器)
        # 注意：这里我们使用 X_tr, y_tr 来拟合，避免数据泄露
        calibrated_model.fit(X_tr, y_tr)

        # 使用校准后的模型进行预测
        y_pred_proba = calibrated_model.predict_proba(X_te)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # 记录结果
        cv_results[name]["AUC"].append(roc_auc_score(y_te, y_pred_proba))
        cv_results[name]["Accuracy"].append(accuracy_score(y_te, y_pred))
        cv_results[name]["Brier"].append(brier_score_loss(y_te, y_pred_proba))

    print(f"完成第 {fold+1} 折交叉验证的校准评估。")

# === 6️⃣ 汇总结果 ===
summary = []
n_folds = kf.get_n_splits()

# ... (汇总和保存代码保持不变) ...

for name, metrics in cv_results.items():
    auc_mean, auc_std = np.mean(metrics["AUC"]), np.std(metrics["AUC"])
    acc_mean, acc_std = np.mean(metrics["Accuracy"]), np.std(metrics["Accuracy"])
    brier_mean, brier_std = np.mean(metrics["Brier"]), np.std(metrics["Brier"])

    summary.append({
        "Model": name,
        "AUC_mean": auc_mean,
        "AUC_std": auc_std,
        "AUC_95CI_lower": auc_mean - 1.96 * auc_std / np.sqrt(n_folds),
        "AUC_95CI_upper": auc_mean + 1.96 * auc_std / np.sqrt(n_folds),
        "Accuracy_mean": acc_mean,
        "Accuracy_std": acc_std,
        "Accuracy_95CI_lower": acc_mean - 1.96 * acc_std / np.sqrt(n_folds),
        "Accuracy_95CI_upper": acc_mean + 1.96 * acc_std / np.sqrt(n_folds),
        "Brier_mean": brier_mean,
        "Brier_std": brier_std,
        "Brier_95CI_lower": brier_mean - 1.96 * brier_std / np.sqrt(n_folds),
        "Brier_95CI_upper": brier_mean + 1.96 * brier_std / np.sqrt(n_folds),
    })

summary_df = pd.DataFrame(summary)
print(summary_df)

# === 7️⃣ 保存结果 ===
summary_df.to_csv("5models_calibrated_cv_results_10fold.csv", index=False)


# === 8️⃣ 神经网络交叉验证 ===
# ❗ 警告：NN 的 CV 评估非常复杂，建议单独进行，不在此处修改 ❗
# ... (NN 评估代码保持不变) ...
# === 8️⃣ 神经网络交叉验证 ===
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
import numpy as np
import pandas as pd

# === 模型定义 ===
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
        x = self.fc4(x)
        return x

# === 数据 ===
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

# === CV 设置 ===
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {"AUC": [], "Accuracy": [], "Brier": []}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor, y_train)):
    X_tr, X_val = X_tensor[train_idx], X_tensor[val_idx]
    y_tr, y_val = y_tensor[train_idx], y_tensor[val_idx]

    model = Net(input_dim=X_tensor.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):  # 可以调节 epoch
        model.train()
        optimizer.zero_grad()
        output = model(X_tr)
        loss = criterion(output, y_tr)
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        probs = torch.sigmoid(logits).numpy().flatten()
        y_true = y_val.numpy().flatten()

    auc = roc_auc_score(y_true, probs)
    acc = accuracy_score(y_true, (probs >= 0.5).astype(int))
    brier = brier_score_loss(y_true, probs)

    results["AUC"].append(auc)
    results["Accuracy"].append(acc)
    results["Brier"].append(brier)
    print(f"Fold {fold+1}: AUC={auc:.3f}, Acc={acc:.3f}, Brier={brier:.3f}")

# === 汇总神经网络结果 ===
n_folds = kf.get_n_splits()

summary_nn = pd.DataFrame({
    "Metric": ["AUC", "Accuracy", "Brier"],
    "Mean": [np.mean(results["AUC"]), np.mean(results["Accuracy"]), np.mean(results["Brier"])],
    "Std": [np.std(results["AUC"]), np.std(results["Accuracy"]), np.std(results["Brier"])],
})

# === 加入95%置信区间 ===
summary_nn["Lower_95CI"] = summary_nn["Mean"] - 1.96 * summary_nn["Std"] / np.sqrt(n_folds)
summary_nn["Upper_95CI"] = summary_nn["Mean"] + 1.96 * summary_nn["Std"] / np.sqrt(n_folds)

print(summary_nn)
summary_nn.to_csv("nn_cv_10fold.csv", index=False)

#%%
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, roc_auc_score

# ====================================================
# 第一部分：数据加载、模型加载与概率重新计算
# ====================================================

# --- 1. 加载数据 (请确保路径和文件名正确) ---
# --- 修正后的加载数据代码 ---

# 1. 定义正确的绝对路径
absolute_path_X = r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\X_test_simple.csv"
absolute_path_y = r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\y_test_simple.csv"

try:
    # 🚨 修正：移除 index_col=0 参数！强制读取所有列作为特征。
    X_test_model = pd.read_csv(absolute_path_X)
    y_test_model_simple = pd.read_csv(absolute_path_y).squeeze()

    # 2. 检查是否有未命名的序号列（通常是第一列）并删除
    # 如果您的 CSV 包含一个未命名的序号列，它会作为第一列出现
    if X_test_model.columns[0].startswith('Unnamed: '):
        X_test_model = X_test_model.drop(columns=X_test_model.columns[0])

    # 3. 再次调试以确认特征数和名称
    print("\n--- 修正后调试结果 ---")
    print(f"X_test_model 新的特征数量: {X_test_model.shape[1]}")
    print(f"X_test_model 的特征名称: {X_test_model.columns.tolist()}")
    print("--------------------------")

    # 4. 确保 y_test_model_simple 是 1D array for y_true
    y_true = y_test_model_simple.to_numpy().ravel()

    if X_test_model.shape[1] != 12:
        print("🚨 警告：修正后特征数仍不是 12，请手动检查 CSV 文件结构。")

    # ... (如果成功，继续运行模型加载代码)

except Exception as e:
    # ... (错误处理不变)
    pass
# --- 2. 重新定义 NN 模型架构 (必须与训练时一致) ---
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
        x = self.fc4(x)
        return x

# --- 3. 加载模型和重新计算概率 ---
input_dim = X_test_model.shape[1]
y_probs_dict = {}

# a) NN 模型概率计算
try:
    nn_model = Net(input_dim)
    # 假设权重文件名为 'nn_model_simple.pth'
    nn_model.load_state_dict(torch.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_nn_model_simple.pth"))
    nn_model.eval()
    # 假设 Platt Scaler 文件名为 'nn_platt_scaler_simple.pkl'
    platt_scaler = joblib.load(r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\nn_platt_scaler_simple.pkl")

    # 计算 NN 校准后概率
    X_test_tensor = torch.tensor(X_test_model.values.astype(np.float32), dtype=torch.float32)
    with torch.no_grad():
        raw_log_odds_test = nn_model(X_test_tensor).cpu().numpy().flatten()

    # 应用 Platt Scaling
    y_pred_proba_simple_nn = platt_scaler.predict_proba(raw_log_odds_test.reshape(-1, 1))[:, 1]
    y_probs_dict["NN"] = y_pred_proba_simple_nn.ravel()
    print("✅ NN 模型概率计算成功。")

except Exception as e:
    print(f"🚨 错误：NN 模型加载或计算失败。请检查 NN 相关文件路径。错误信息: {e}")


# b) Sklearn 模型概率计算
sklearn_model_files = {
    "LR": r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_lr_model_simple.pkl",
    "RF": r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_rf_model_simple.pkl",
    "XGB": r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_xgb_model_simple.pkl",
    "SVM": r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_svm_model_simple.pkl",
    "NB": r"C:\Users\user\Desktop\返修\2次\外部驗證\數據\2-meta-12-261021\calibrated_nb_model_simple.pkl"
}

for name, filename in sklearn_model_files.items():
    try:
        calibrated_model = joblib.load(filename)
        # 假设 calibrated_model 是 CalibratedClassifierCV 实例
        y_probs = calibrated_model.predict_proba(X_test_model)[:, 1]
        y_probs_dict[name] = y_probs.ravel()
    except Exception as e:
        print(f"🚨 警告：加载或计算模型 {name} 失败。请检查 {filename}。错误信息: {e}")

print("\n✅ 所有成功加载的模型概率已存储在 y_probs_dict 中。")


# ====================================================
# 第二部分：通用 Cutoff 计算逻辑
# ====================================================

# --- 2. 定义寻找最优阈值的函数 (基于 Youden Index) ---
def find_optimal_cutoff(y_true, y_probs):
    """使用 Youden Index (Sensitivity + Specificity - 1) 寻找最佳阈值。"""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]

# --- 3. 通用计算和结果汇总 ---

summary_list = []

for name, y_probs in y_probs_dict.items():
    # 找到最优阈值
    optimal_thresh = find_optimal_cutoff(y_true, y_probs)

    # 使用最优阈值进行二元分类预测
    y_pred_optimal = (y_probs >= optimal_thresh).astype(int)

    # 计算混淆矩阵: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()

    # 计算分类指标
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # 处理除以零情况 (如果 tp+fp=0, precision=0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn)
    accuracy = accuracy_score(y_true, y_pred_optimal)
    # 处理除以零情况 (如果 precision+sensitivity=0, F1=0)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    auc_score = roc_auc_score(y_true, y_probs)

    # 汇总结果
    summary_list.append({
        "Model": name,
        "AUC": f"{auc_score:.3f}",
        "Cutoff (Optimal)": f"{optimal_thresh:.4f}",
        "Accuracy": f"{accuracy:.3f}",
        "Sensitivity": f"{sensitivity:.3f}",
        "Specificity": f"{specificity:.3f}",
        "Precision (PPV)": f"{precision:.3f}",
        "NPV": f"{npv:.3f}",
        "F1-score": f"{f1_score:.3f}"
    })

# --- 4. 结果展示与保存 ---

results_df = pd.DataFrame(summary_list)
print("\n" + "="*50)
print("✅ 所有模型在测试集上的最优 Cutoff 分类指标 (基于 Youden Index):")
print(results_df.to_string(index=False))
print("="*50)

# 如果需要保存到 CSV
results_df.to_csv("15models_optimal_cutoff_metrics.csv", index=False)
