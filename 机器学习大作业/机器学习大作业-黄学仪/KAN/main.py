import torch


torch.set_default_dtype(torch.float64)  # 设置默认张量数据类型为双精度浮点数
from sklearn.model_selection import StratifiedKFold
from kan import KAN
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 加载并预处理数据
data = pd.read_csv('heart.csv')

# 分割特征和标签
X = data.drop(['target'], axis=1).values
y = data['target'].values

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 使用Stratified K-Fold交叉验证
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

train_accuracies = []
test_accuracies = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 将数据转换为 PyTorch 张量
    dataset = {}
    dataset['train_input'] = torch.from_numpy(X_train).double()
    dataset['test_input'] = torch.from_numpy(X_test).double()
    dataset['train_label'] = torch.from_numpy(y_train[:, None]).double()
    dataset['test_label'] = torch.from_numpy(y_test[:, None]).double()

    # 定义模型
    model = KAN(width=[X_train.shape[1], 5, 1], grid=5, k=3)


    # 定义准确率计算函数
    def train_acc():
        return torch.mean((torch.round(model(dataset['train_input'])[:, 0]) == dataset['train_label'][:, 0]).float())


    def test_acc():
        return torch.mean((torch.round(model(dataset['test_input'])[:, 0]) == dataset['test_label'][:, 0]).float())


    # 训练模型并获取结果
    results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc))

    # 输出每个fold的训练准确率和测试准确率
    train_accuracy = results['train_acc'][-1]
    test_accuracy = results['test_acc'][-1]
    print(f"Fold {fold + 1}:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # 设置激活函数为符号公式
    mode = "auto"  # 自动模式

    if mode == "manual":
        # 手动模式
        model.fix_symbolic(0, 0, 0, 'sin')
        model.fix_symbolic(0, 1, 0, 'x^2')
        model.fix_symbolic(1, 0, 0, 'exp')
    elif mode == "auto":
        # 自动模式
        lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
        model.auto_symbolic(lib=lib)

    # 获取符号公式
    symbolic_formulas = model.symbolic_formula()
    if symbolic_formulas is not None:
        for formula in symbolic_formulas:
            print(formula[0])
    else:
        print("No symbolic formula available. Make sure all activations are converted to symbolic formulas first.")

    # 绘制模型图像
    plt.figure()
    model.plot()
    # plt.savefig(f'model_plot_fold_{fold + 1}.png')
    plt.show()

# 输出平均训练准确率和测试准确率
print(f"Average Train Accuracy: {np.mean(train_accuracies):.4f}")
print(f"Average Test Accuracy: {np.mean(test_accuracies):.4f}")
