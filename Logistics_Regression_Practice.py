from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import warnings
# %matplotlib inline

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

# 绘制sifmoid函数


def sigmoid(z):
    return 1/(1 + np.exp(-z))


z = np.linspace(-10, 10, 250)

# 绘制sigmoid函数图像
plt.plot(z, sigmoid(z))
# 绘制水平与垂直线
plt.axvline(x=0, ls="--", c="k")
plt.axhline(ls=":", c="k")
plt.axhline(y=0.5, ls=":", c="k")
plt.axhline(y=1, ls=":", c="k")

plt.xlabel("z值")
plt.ylabel("sigmoid(z)值")
# Text(0, 0.5, "sigmoid(z)")
plt.show()


# 损失函数可视化
s = np.linspace(0, 1, 200)
for y in [0, 1]:
    loss = -y * np.log(s) - (1-y)*np.log(1-s)
    plt.plot(s, loss, label=f"y={y}")
plt.legend()
plt.xlabel("sigmoid(z)")
plt.ylabel("J(w)")
plt.title("损失函数J（w）与sigmoid（z）的关系")
plt.show()


# 逻辑回归实现二分类

warnings.filterwarnings("ignore")

iris = load_iris()
iris
X, y = iris.data, iris.target
X
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
print("权重：", lr.coef_)
print("偏置：", lr.intercept_)
print("真实值：", y_test)
print("预测值：", y_hat)


# 结果可视化
# 绘制鸢尾花数据的分布图
c1 = X[y == 0]
c2 = X[y == 1]
c1
plt.scatter(x=c1[:, 0], y=c1[:, 1], c="m", label="类别0")
plt.scatter(x=c2[:, 0], y=c2[:, 1], c="c", label="类别1")
plt.xlabel("花瓣长度")
plt.ylabel("花瓣宽度")
plt.title("鸢尾花样本分布")
plt.legend(loc="best", bbox_to_anchor=(1, 1))
plt.show()

# 绘制在测试集中，样本的真实类别和预测类别
plt.figure(figsize=(12, 5))
plt.plot(y_test, "co", ms=15, label="真实类别")
plt.plot(y_hat, "m*", ms=15, label="预测类别")
plt.legend(loc="best", bbox_to_anchor=(1, 1))
plt.xlabel("样本序号")
plt.ylabel("类别")
plt.title("逻辑回归分类预测结果")
plt.show()

# 获取预测的概率值，包含数据属于哪个类别的概率
probability = lr.predict_proba(X_test)
probability[:5]
np.argmax(probability, axis=1)
# 产生序列号，用于可视化的横坐标
index = np.arange(len(X_test))
pro_0 = probability[:, 0]
# pro_0
pro_1 = probability[:, 1]
tick_label = np.where(y_test == y_hat, "o", "x")
plt.figure(figsize=(12, 5))
# 绘制堆叠图
plt.bar(index, height=pro_0, color="b", label="类别o概率值")
# bottom = x，表示从x的值起往上堆叠
# tick_label 设置标签刻度的文本内容
plt.bar(index, height=pro_1, bottom=pro_0, color="c",
        label="类别1概率值", tick_label=tick_label)
plt.legend(loc="best", bbox_to_anchor=(1, 1))
plt.xlabel("样本序号")
plt.ylabel("各个类别的概率")
plt.title("逻辑回归分类概率")
plt.show()


# 绘制决策边界
# 我们可以绘制决策边界，将分类效果进行可视化显示
np.max(X, axis=0)


# 定义函数，用于绘制决策边界

def plot_decision_boundary(model, X, y):
    color = ["c", "m", "b"]
    marker = ["o", "*", "x"]
    class_label = np.unique(y)
    cmap = ListedColormap(color[: len(class_label)])
    x1_min, x2_min = np.min(X, axis=0)
    x1_max, x2_max = np.max(X, axis=0)
    x1 = np.arange(x1_min - 0.5, x1_max+0.5, 0.01)
    x2 = np.arange(x2_min - 0.5, x2_max+0.5, 0.01)
    X1, X2 = np.meshgrid(x1, x2)
    Z = model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    # 绘制使用颜色填充的登高线
    plt.contourf(X1, X2, Z, cmap=cmap, alpha=0.35)
    for i, class_ in enumerate(class_label):
        plt.scatter(x=X[y == class_, 0], y=X[y == class_, 1], lw=3,
                    c=cmap.colors[i], label=class_, marker=marker[i])
    plt.legend()
    plt.show()


plot_decision_boundary(lr, X_train, y_train)
plot_decision_boundary(lr, X_test, y_test)


# 逻辑回归实现多分类
# 建模与可视化
iris = load_iris()
X, y = iris.data, iris.target
# 仅使用其中的两个特征
X = X[:, 2:]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_hat = lr.predict(X_test)
print("分类正确率：", np.sum(y_test == y_hat) / len(y_test))

# 训练集决策边界
plot_decision_boundary(lr, X_train, y_train)
plot_decision_boundary(lr, X_test, y_test)



#说明：这是基于二分类做的
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import warnings
import numpy as np
warnings.filterwarnings('ignore')

iris=load_iris()
X,y=iris.data,iris.target
X=X[y!=0,2:]
y=y[y!=0]
y[y==1]=0
y[y==2]=1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2)
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_hat=lr.predict(X_test)
print('分类正确率:',np.sum(y_test==y_hat)/len(y_test))

# 使用四个特征进行分类
iris=load_iris()
X,y=iris.data,iris.target
X=X[y!=0] #将原来的2个特征改为全部
y=y[y!=0]
y[y==1]=0
y[y==2]=1
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.25,random_state=2)
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_hat=lr.predict(X_test)
print('分类正确率:',np.sum(y_test==y_hat)/len(y_test))

# 使用全部特征进行多分类
iris = load_iris()
X, y = iris.data, iris.target
# 仅使用其中的两个特征
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_hat = lr.predict(X_test)
print("分类正确率：", np.sum(y_test == y_hat) / len(y_test))