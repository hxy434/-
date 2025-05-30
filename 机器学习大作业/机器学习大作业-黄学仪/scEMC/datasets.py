import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler


class TrainDataset(torch.utils.data.Dataset):
    """
    自定义数据集类，用于处理多个视图的数据。

    参数：
    - X_list: 包含多个视图输入特征的列表。
    - Y_list: 包含多个视图标签的列表。
    """

    def __init__(self, X_list, Y_list):
        self.X_list = X_list  # 输入特征列表
        self.Y_list = Y_list  # 标签列表
        self.view_size = len(X_list)  # 视图的数量

    def __getitem__(self, index):
        """
        获取指定索引的数据，包括所有视图的输入特征和标签。

        参数：
        - index: 数据索引
        

        返回：
        - current_x_list: 包含所有视图输入特征的列表
        - current_y_list: 包含所有视图标签的列表
        """
        current_x_list = []
        current_y_list = []
        for v in range(self.view_size):
            current_x = self.X_list[v][index]  # 获取第v个视图的输入特征
            current_x_list.append(current_x)  # 添加到输入特征列表中
            current_y = self.Y_list[v][index]  # 获取第v个视图的标签
            current_y_list.append(current_y)  # 添加到标签列表中
        return current_x_list, current_y_list

    def __len__(self):
        """
        返回数据集的大小，即第一个视图的样本数量。
        """
        return self.X_list[0].shape[0]


class Data_Sampler(object):
    """
    自定义采样器，用于生成数据批次。

    参数：
    - pairs: 数据对（索引）
    - shuffle: 是否打乱数据
    - batch_size: 批次大小
    - drop_last: 是否丢弃最后一个不完整的批次
    """

    def __init__(self, pairs, shuffle=False, batch_size=1, drop_last=False):
        if shuffle:
            self.sampler = RandomSampler(pairs)  # 使用随机采样器
        else:
            self.sampler = SequentialSampler(pairs)  # 使用顺序采样器
        self.batch_size = batch_size  # 批次大小
        self.drop_last = drop_last  # 是否丢弃最后一个批次

    def __iter__(self):
        """
        生成数据批次的迭代器。
        """
        batch = []
        for idx in self.sampler:
            batch.append(idx)  # 添加索引到当前批次
            if len(batch) == self.batch_size:
                batch = [batch]  # 将批次包装成列表
                yield batch  # 生成当前批次
                batch = []  # 重置批次
        if len(batch) > 0 and not self.drop_last:
            batch = [batch]
            yield batch

    def __len__(self):
        """
        返回批次数。
        """
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
