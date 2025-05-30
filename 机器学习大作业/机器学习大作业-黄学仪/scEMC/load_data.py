import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")

# 数据集字典，包含数据集的基本信息（名称 数据集路径 样本数量 聚类数 视图数量 输入特征维度 隐藏层维度 输出维度）
ALL_data = dict(
    BMNC={1: 'BMNC', 2: 'd1', 'N': 30672, 'K': 27, 'V': 2, 'n_input': [1000, 25], 'n_hid': [10, 256], 'n_output': 64},
    # 可以添加更多数据集信息，例如：
    # SL111= {1: 'SL111', 2: 'd1', 'N': 30672, 'K': 27, 'V': 2, 'n_input': [1000,25], 'n_hid': [10,256], 'n_output': 64},
)

# 数据集路径
path = './datasets/'



def load_data(dataset):
    """
    加载指定的数据集，并进行预处理。

    参数：
    - dataset: 数据集的配置信息

    返回：
    - X: 预处理后的输入特征列表
    - Y: 标签列表
    """
    # 打开指定路径的 .mat 文件，读取数据
    data = h5py.File(path + dataset[1] + ".mat")
    X = []  # 用于存储预处理后的输入特征
    Y = []  # 用于存储标签
    Label = np.array(data['Y']).T  # 读取标签
    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()  # 创建 MinMaxScaler 对象，用于特征缩放

    # 遍历数据的每个视图
    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]  # 读取第 i 个视图的数据
        diff_view = np.array(diff_view, dtype=np.float32).T  # 转换为 numpy 数组并转置
        std_view = mm.fit_transform(diff_view)  # 对数据进行归一化处理
        X.append(std_view)  # 将预处理后的数据添加到输入特征列表中
        Y.append(Label)  # 将标签添加到标签列表中

    size = len(Y[0])  # 样本数量
    view_num = len(X)  # 视图数量

    # 打乱数据顺序
    index = [i for i in range(size)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]

    # 将输入特征转换为 PyTorch 张量
    for v in range(view_num):
        X[v] = torch.from_numpy(X[v])

    return X, Y
