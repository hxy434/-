import torch
import torch.nn as nn
import torch.nn.functional as F


class NBLoss(nn.Module):
    """
    负二项分布（Negative Binomial）损失函数类。
    """

    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=1.0):
        """
        前向计算函数，计算负二项分布损失。

        参数：
        - x: 输入张量
        - mean: 预测均值
        - disp: 离散度
        - scale_factor: 缩放因子

        返回：
        - result: 计算的负二项分布损失
        """
        eps = 1e-10  # 防止数值问题的极小值
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        result = t1 + t2

        result = torch.mean(result)
        return result



class ZINBLoss(nn.Module):
    """
    零膨胀负二项分布（Zero-Inflated Negative Binomial）损失函数类。
    """

    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        """
        前向计算函数，计算零膨胀负二项分布损失。

        参数：
        - x: 输入张量
        - mean: 预测均值
        - disp: 离散度
        - pi: 零膨胀概率
        - scale_factor: 缩放因子
        - ridge_lambda: 岭回归系数

        返回：
        - result: 计算的零膨胀负二项分布损失
        """
        eps = 1e-10  # 防止数值问题的极小值
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class GaussianNoise(nn.Module):
    """
    高斯噪声层，在训练期间添加高斯噪声。
    """

    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        """
        前向计算函数，在训练期间添加高斯噪声。

        参数：
        - x: 输入张量

        返回：
        - 添加高斯噪声后的张量
        """
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


class MeanAct(nn.Module):
    """
    自定义激活函数类，用于输出的均值。
    """

    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        """
        前向计算函数，对输入张量应用激活函数。

        参数：
        - x: 输入张量

        返回：
        - 激活后的张量
        """
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """
    自定义激活函数类，用于输出的离散度。
    """

    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        """
        前向计算函数，对输入张量应用激活函数。

        参数：
        - x: 输入张量

        返回：
        - 激活后的张量
        """
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
