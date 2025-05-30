# Copyright 2017 Goekcen Eraslan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pickle
import os
import numbers
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import scipy


class AnnSequence:
    """
    一个用于批处理数据的类。

    参数：
    - matrix: 数据矩阵
    - batch_size: 批量大小
    - sf: 大小因子（size factors）
    """

    def __init__(self, matrix, batch_size, sf=None):
        self.matrix = matrix
        if sf is None:
            self.size_factors = np.ones((self.matrix.shape[0], 1), dtype=np.float32)
        else:
            self.size_factors = sf
        self.batch_size = batch_size

    def __len__(self):
        """
        返回批次数。
        """
        return len(self.matrix) // self.batch_size

    def __getitem__(self, idx):
        """
        获取指定批次的数据。

        参数：
        - idx: 批次索引

        返回：
        - 包含数据和大小因子的字典，以及数据
        """
        batch = self.matrix[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sf = self.size_factors[idx * self.batch_size:(idx + 1) * self.batch_size]

        return {'count': batch, 'size_factors': batch_sf}, batch


def read_dataset(adata, transpose=False, test_split=False, copy=False):
    """
    读取并预处理数据集。

    参数：
    - adata: AnnData 对象或文件路径
    - transpose: 是否转置数据矩阵
    - test_split: 是否拆分训练和测试数据
    - copy: 是否复制数据

    返回：
    - 预处理后的 AnnData 对象
    """
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    if adata.X.size < 50e6:  # 如果数组较小，检查是否为整数
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            pass
            # assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose:
        adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


def clr_normalize_each_cell(adata):
    """
    对每个细胞进行 CLR 归一化。

    参数：
    - adata: AnnData 对象

    返回：
    - 归一化后的 AnnData 对象
    """

    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    adata.raw = adata.copy()
    sc.pp.normalize_per_cell(adata)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.raw.X.A if scipy.sparse.issparse(adata.raw.X) else adata.raw.X)
    )
    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    """
    归一化数据集。

    参数：
    - adata: AnnData 对象
    - filter_min_counts: 是否过滤最小计数
    - size_factors: 是否计算大小因子
    - normalize_input: 是否归一化输入
    - logtrans_input: 是否对输入进行对数转换

    返回：
    - 归一化后的 AnnData 对象
    """
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def read_genelist(filename):
    """
    读取基因列表。

    参数：
    - filename: 基因列表文件路径

    返回：
    - 基因列表
    """
    genelist = list(set(open(filename, 'rt').read().strip().split('\n')))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    print('### Autoencoder: Subset of {} genes will be denoised.'.format(len(genelist)))

    return genelist


def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    """
    将矩阵写入文本文件。

    参数：
    - matrix: 要写入的矩阵
    - filename: 文件名
    - rownames: 行名
    - colnames: 列名
    - transpose: 是否转置矩阵
    """
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')


def read_pickle(inputfile):
    """
    读取 pickle 文件。

    参数：
    - inputfile: pickle 文件路径

    返回：
    - 读取的对象
    """
    return pickle.load(open(inputfile, "rb"))
