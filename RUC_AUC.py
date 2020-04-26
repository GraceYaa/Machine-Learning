import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 混淆矩阵
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
warnings.filterwarnings("ignore")

iris = load_iris()
X, y = iris.data, iris.target
# 选定鸢尾花的后两个特征，并移除类别0
X = X[y != 0, 2:]
y = y[y != 0]
# 此时y的标签为1，2， 修改y的标签为0，1
y[y == 1] = 0
y[y == 2] = 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_hat = lr.predict(X_test)
# 根据传入的真实值与预测值，创建混淆矩阵
matrix = confusion_matrix(y_true=y_test, y_pred=y_hat)
print(matrix)

mat = plt.matshow(matrix, cmap=plt.cm.Blues, alpha=0.4)
label = ["负例", "正例"]
ax = plt.gca()
ax.set(xticks=np.arange(matrix.shape[1]), yticks=np.arange(
    matrix.shape[0]), xticklabels=label, yticklabels=label, title="混淆矩阵可视化")
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(x=j, y=i, s=matrix[i, j], va="center", ha="center")
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("正确率：", accuracy_score(y_test, y_hat))
# 默认将1类别视为正例，可以通过pos_label参数指定。
print("精准率：", precision_score(y_test, y_hat))
print("召回率：", recall_score(y_test, y_hat))
print("F1调和平均值：", f1_score(y_test, y_hat))
# 我们也可以调用逻辑回归模型对象的score方法，也能获取正确率。
# 但是需要注意，score方法与f1_score函数的参数是不同的。
print("score方法计算正确率：", lr.score(X_test, y_test))


# 使用classification_report函数查看模型分类统计信息，该方法会返回字符串类型，给出相关的分类指标估值
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred=y_hat))


# roc_curve函数
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
y = np.array([0, 1, 0, 1, 1])
scores = np.array([0.1, 0.3, 0.2, 0.35, 0.8])
# 返回ROC曲线相关值。返回FPR，TPR与阈值。当分值达到阈值时，将样本判定为正类，
# 否则判定为负类。
# y_true：二分类的标签值（真实值）。
# y_score：每个标签（数据）的分值或概率值。当该值达到阈值时，判定为正例，否则判定为负例。
# 在实际模型评估时，该值往往通过决策函数（decision_function）或者概率函数（predict_proba）获得。
# pos_label：指定正例的标签值。
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
print(f"fpr：{fpr}")
print(f"tpr：{tpr}")
print(f"thresholds：{thresholds}")
# auc与roc_auc_score函数都可以返回AUC面积值，但是注意，两个函数的参数是不同的。
print("AUC面积值：", auc(fpr, tpr))
print("AUC面积得分：", roc_auc_score(y_true=y, y_score=scores))


iris = load_iris()
X, y = iris.data, iris.target
X = X[y != 0, 2:]
y = y[y != 0]
y[y == 1] = 0
y[y == 2] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
lr = LogisticRegression()
lr.fit(X_train, y_train)
# 使用概率来作为每个样本数据的分值。
probo = lr.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=probo[:, 1], pos_label=1)
print(probo[:, 1])
# 从概率中，选择若干元素作为阈值，每个阈值下，都可以确定一个tpr与fpr，
# 每个tpr与fpr对应ROC曲线上的一个点，将这些点进行连接，就可以绘制ROC曲线。
print(thresholds)

# 随着阈值的不断降低，tpr与fpr都在不断增大
tpr,fpr


# 绘制ROC曲线

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, marker="o",lw=3, label="ROC曲线")
plt.plot([0,1], [0,1], lw=2, ls="--", label="随机猜测")
plt.plot([0, 0, 1], [0, 1, 1], lw=2, ls="-.", label="完美预测")
plt.xlim(-0.01, 1.02)
plt.ylim(-0.01, 1.02)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel("False Positive Rate(FPR)")
plt.ylabel("True Positive Rate(TPR)")
plt.grid()
plt.title(f"ROC曲线-AUC值为{auc(fpr, tpr):.2f}")
plt.legend()
plt.show()




