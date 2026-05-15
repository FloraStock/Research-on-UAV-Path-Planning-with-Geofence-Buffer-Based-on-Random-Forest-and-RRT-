# ml_buffer_decider.py
"""
Random Forest 缓冲区动态决策模型训练与评估
处理类别不平衡 + 生成论文所需图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 设置中文字体（论文用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # 1. 加载数据
    print("📥 加载数据集...")
    df = pd.read_csv("results/ml_data/buffer_decision_dataset.csv")

    # 特征列（去掉目标和无关列）
    feature_cols = ['min_clearance', 'local_density', 'length_increase_ratio',
                    'num_nodes', 'avg_deviation', 'sim_battery',
                    'path_length', 'straight_dist']
    X = df[feature_cols]
    y = df['need_buffer']

    print(f"特征维度: {X.shape}")
    print(f"正样本比例: {y.mean():.1%}")

    # 2. 划分训练集/测试集（分层抽样保持比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 训练 Random Forest（处理不平衡）
    print("\n🌲 训练 Random Forest（class_weight='balanced'）...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',  # 自动处理不平衡
        random_state=42,
        n_jobs=-1
    )

    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"5折交叉验证 ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 训练最终模型
    rf.fit(X_train, y_train)

    # 4. 测试集评估
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\n📊 测试集性能:")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}   ← 重点关注（安全优先）")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['不需要缓冲', '需要缓冲']))

    # 5. 保存模型
    os.makedirs("results/ml_models", exist_ok=True)
    joblib.dump(rf, "results/ml_models/buffer_rf_model.pkl")
    print("✅ 模型已保存: results/ml_models/buffer_rf_model.pkl")

    # 6. 生成论文图表
    os.makedirs("results/ml_figures", exist_ok=True)

    # === 图1: Feature Importance ===
    plt.figure(figsize=(10, 6))
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.8)
    plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=45, ha='right')
    plt.title('Random Forest 特征重要性排序', fontsize=14, fontweight='bold')
    plt.ylabel('重要性得分')
    plt.tight_layout()
    plt.savefig("results/ml_figures/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("📈 特征重要性图已保存")

    # === 图2: Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['不需要缓冲', '需要缓冲'],
                yticklabels=['不需要缓冲', '需要缓冲'])
    plt.title('混淆矩阵（测试集）', fontsize=14, fontweight='bold')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig("results/ml_figures/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 混淆矩阵已保存")

    # === 图3: ROC Curve ===
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5,
             label=f'ROC 曲线 (AUC = {roc_auc_score(y_test, y_prob):.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR / Recall)')
    plt.title('ROC 曲线（测试集）', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/ml_figures/roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("📉 ROC 曲线已保存")

    # 7. 保存详细结果到 CSV（方便写论文）
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Value': [accuracy_score(y_test, y_pred),
                  precision_score(y_test, y_pred),
                  recall_score(y_test, y_pred),
                  f1_score(y_test, y_pred),
                  roc_auc_score(y_test, y_prob)]
    })
    results_df.to_csv("results/ml_results/performance_metrics.csv", index=False)
    print("📋 性能指标已保存: results/ml_results/performance_metrics.csv")

    print("\n🎉 全部完成！请检查 results/ml_figures/ 下的三张图和模型文件。")


if __name__ == "__main__":
    main()