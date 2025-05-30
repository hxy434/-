import argparse
import os
import numpy as np
import torch
from time import time
import load_data as loader
from network import scEMC
from preprocess import read_dataset, normalize
from utils import *


def parse_arguments(data_para):
    """
    解析命令行参数。


    参数：
    - data_para: 数据集参数字典

    返回：
    - args: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='scEMC')
    parser.add_argument('--n_clusters', default=data_para['K'], type=int, help="聚类的数量")
    parser.add_argument('--lr', default=1, type=float, help="学习率")
    parser.add_argument('-el1', '--encodeLayer1', nargs='+', default=[256, 64, 32, 8], help="第一个视图的编码层大小")
    parser.add_argument('-el2', '--encodeLayer2', nargs='+', default=[256, 64, 32, 8], help="第二个视图的编码层大小")
    parser.add_argument('-dl1', '--decodeLayer1', nargs='+', default=[24, 64, 256], help="第一个视图的解码层大小")
    parser.add_argument('-dl2', '--decodeLayer2', nargs='+', default=[24, 20], help="第二个视图的解码层大小")
    parser.add_argument('--dataset', default=data_para, help="数据集参数")
    parser.add_argument("--view_dims", default=data_para['n_input'], help="视图的输入维度")
    parser.add_argument('--name', type=str, default=data_para[1], help="数据集名称")
    parser.add_argument('--cutoff', default=0.5, type=float, help='在什么比例的epoch后开始训练组合层')
    parser.add_argument('--batch_size', default=256, type=int, help="批量大小")
    parser.add_argument('--maxiter', default=500, type=int, help="最大迭代次数")
    parser.add_argument('--pretrain_epochs', default=400, type=int, help="预训练epoch数")
    parser.add_argument('--gamma', default=.1, type=float, help='聚类损失的系数')
    parser.add_argument('--tau', default=1., type=float, help='聚类损失的模糊度')
    parser.add_argument('--phi1', default=0.001, type=float, help='KL损失的预系数')
    parser.add_argument('--phi2', default=0.001, type=float, help='KL损失的系数')
    parser.add_argument('--update_interval', default=1, type=int, help='更新间隔')
    parser.add_argument('--tol', default=0.001, type=float, help='容忍度')
    parser.add_argument('--ae_weights', default=None, help="预训练自编码器的权重文件")
    parser.add_argument('--save_dir', default='result/', help="结果保存目录")
    parser.add_argument('--ae_weight_file', default='model.pth.tar', help="自编码器权重文件名")
    parser.add_argument('--resolution', default=0.2, type=float, help="聚类的分辨率")
    parser.add_argument('--n_neighbors', default=30, type=int, help="聚类的邻居数量")
    parser.add_argument('--embedding_file', action='store_true', default=True, help="是否保存嵌入文件")
    parser.add_argument('--prediction_file', action='store_true', default=True, help="是否保存预测文件")
    parser.add_argument('--sigma1', default=2.5, type=float, help="第一个视图的噪声标准差")
    parser.add_argument('--sigma2', default=1.5, type=float, help="第二个视图的噪声标准差")
    parser.add_argument('--f1', default=2000, type=float, help='特征选择后的mRNA数量')
    parser.add_argument('--f2', default=2000, type=float, help='特征选择后的ADT/ATAC数量')
    parser.add_argument('--run', default=1, type=int, help="运行次数")
    parser.add_argument('--device', default='cuda', help="设备类型")
    parser.add_argument('--lam1', default=1, type=float, help="聚类损失的权重")
    parser.add_argument('--lam2', default=1, type=float, help="重构损失的权重")
    return parser.parse_args()


def prepare_data(dataset, size_factors=True, normalize_input=True, logtrans_input=True):
    """
    准备数据用于模型输入。

    参数：
    - dataset: 数据集
    - size_factors: 是否计算大小因子
    - normalize_input: 是否归一化输入
    - logtrans_input: 是否对输入进行对数转换

    返回：
    - 预处理后的 AnnData 对象
    """
    data = sc.AnnData(np.array(dataset))
    data = read_dataset(data, transpose=False, test_split=False, copy=True)
    data = normalize(data, size_factors=size_factors, normalize_input=normalize_input, logtrans_input=logtrans_input)
    return data


def main():
    # 获取数据集参数字典
    my_data_dic = loader.ALL_data
    for i_d in my_data_dic:
        data_para = my_data_dic[i_d]
    # 解析命令行参数
    args = parse_arguments(data_para)
    # 加载数据
    X, Y = loader.load_data(args.dataset)
    labels = Y[0].copy().astype(np.int32)  # 获取标签
    # 准备数据1
    adata1 = prepare_data(X[0])
    # 准备数据2
    adata2 = prepare_data(X[1], size_factors=False, normalize_input=False, logtrans_input=False)
    y = labels
    input_size1 = adata1.n_vars  # 输入尺寸1
    input_size2 = adata2.n_vars  # 输入尺寸2

    # 编码和解码层大小
    encodeLayer1 = list(map(int, args.encodeLayer1))
    encodeLayer2 = list(map(int, args.encodeLayer2))
    decodeLayer1 = list(map(int, args.decodeLayer1))
    decodeLayer2 = list(map(int, args.decodeLayer2))

    # 初始化scEMC模型
    model = scEMC(input_dim1=input_size1, input_dim2=input_size2, tau=args.tau,
                  encodeLayer1=encodeLayer1,
                  encodeLayer2=encodeLayer2,
                  decodeLayer1=decodeLayer1, decodeLayer2=decodeLayer2,
                  activation='elu', sigma1=args.sigma1, sigma2=args.sigma2, gamma=args.gamma,
                  cutoff=args.cutoff, phi1=args.phi1, phi2=args.phi2, device=args.device).to(args.device)

    # 创建结果保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    t0 = time()
    if args.ae_weights is None:
        # 预训练自编码器
        model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
                                   X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors,
                                   batch_size=args.batch_size,
                                   epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        # 加载预训练的自编码器权重
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError

    print('Pretraining time: %d seconds.' % int(time() - t0))

    # 编码数据
    latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device))
    latent = latent.cpu().numpy()
    if args.n_clusters == -1:
        # 估计聚类数量
        n_clusters = GetCluster(latent, res=args.resolution, n=args.n_neighbors)
    else:
        print("n_cluster is defined as " + str(args.n_clusters))
        n_clusters = args.n_clusters

    # 训练模型
    y_pred, _, _, _ = model.fit(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
                                X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, y=y,
                                n_clusters=n_clusters, batch_size=args.batch_size, num_epochs=args.maxiter,
                                update_interval=args.update_interval, tol=args.tol, lr=args.lr,
                                save_dir=args.save_dir, lam1=args.lam1, lam2=args.lam2)
    print('Total time: %d seconds.' % int(time() - t0))

    # 保存预测结果
    if args.prediction_file:
        y_pred_ = best_map(y, y_pred) - 1
        np.savetxt(args.save_dir + "/" + str(args.run) + "_pred.csv", y_pred_, delimiter=",")

    # 保存嵌入结果
    if args.embedding_file:
        final_latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device),
                                         torch.tensor(adata2.X).to(args.device))
        final_latent = final_latent.cpu().numpy()
        np.savetxt(args.save_dir + "/" + str(args.run) + "_embedding.csv", final_latent, delimiter=",")

    y_pred_ = best_map(y, y_pred)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 4)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 4)
    print('Final: ARI= %.4f, NMI= %.4f' % (ari, nmi))

    my_dic2 = dict({'View': 'multi', 'ARI': ari, 'NMI': nmi})  #

    # 保存结果到文件
    with open("./result/{}.txt".format(args.name), "a+") as f:
        f.write(str(args))
        f.write("\n")
        f.write(str(my_dic2) + '\r')


if __name__ == "__main__":
    main()
