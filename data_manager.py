import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn.preprocessing as sk_prep
from itertools import cycle
from sklearn.decomposition import PCA

class HSIDataManager:
    """高光谱图像数据管理类"""
    def __init__(self, 
                 hsi_path_train, 
                 label_path_train,
                 hsi_path_test,
                 label_path_test,
                 patch_size=7,
                 batch_size=32,
                 num_workers=4,
                 bg_num=1000,
                 val_ratio=0.1):
        """
        初始化数据管理器
        
        参数:
            hsi_path_train: 训练用高光谱图像路径
            label_path_train: 训练用标签路径
            hsi_path_test: 测试用高光谱图像路径
            label_path_test: 测试用标签路径
            patch_size: patch大小
            batch_size: 批次大小
            num_workers: 数据加载线程数
            bg_num: 选择的背景点数量
            val_ratio: 验证集占训练数据的比例，默认0.1
        """
        self.hsi_path_train = hsi_path_train
        self.label_path_train = label_path_train
        self.hsi_path_test = hsi_path_test
        self.label_path_test = label_path_test
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bg_num = bg_num
        self.val_ratio = val_ratio
        
        # 验证参数
        if not 0 <= val_ratio < 1:
            raise ValueError("验证集比例必须在0到1之间")
        
    def load_hsi_data(self, hsi_path, label_path):
        """
        加载并预处理高光谱图像数据和标签
        
        参数:
            hsi_path: 高光谱图像mat文件路径
            label_path: 标签mat文件路径
        
        返回:
            hsi_data: 标准化后的高光谱图像数据
            label_data: 标签数据
        """
        # 读取高光谱数据
        hsi_mat = sio.loadmat(hsi_path)
        hsi_keys = [k for k in hsi_mat.keys() if not k.startswith('__')]
        if not hsi_keys:
            raise ValueError("在高光谱mat文件中未找到有效数据")
        hsi_data = hsi_mat[hsi_keys[0]]
        
        # 读取标签数据
        label_mat = sio.loadmat(label_path)
        label_keys = [k for k in label_mat.keys() if not k.startswith('__')]
        if not label_keys:
            raise ValueError("在标签mat文件中未找到有效数据")
        label_data = label_mat[label_keys[0]]
        
        # 对高光谱数据进行标准化处理
        H, W, D = hsi_data.shape
        hsi_data_reshaped = hsi_data.reshape(-1, D)  # 将空间维度展平
        if D == 102:
            hsi_data_scaled = sk_prep.minmax_scale(hsi_data_reshaped)  # 标准化
        else:
            hsi_data_scaled = sk_prep.scale(hsi_data_reshaped)
        
        hsi_data = hsi_data_scaled.reshape(H, W, D)  # 恢复原始形状
        
        print(f"HSI数据形状: {hsi_data.shape}")
        print(f"标签数据形状: {label_data.shape}")
        print("数据标准化完成")
        
        return hsi_data, label_data
    
    def process_hsi_labels(self, label_data):
        """处理HSI标签数据"""
        if label_data.ndim == 3:
            label_data = label_data.squeeze()
        
        unique_labels = np.unique(label_data)
        background_label = unique_labels.min()
        
        label_mapping = {}
        new_label = 1
        for old_label in unique_labels:
            if old_label == background_label:
                label_mapping[old_label] = 0
            else:
                label_mapping[old_label] = new_label
                new_label += 1
        
        processed_labels = np.zeros_like(label_data)
        for old_label, new_label in label_mapping.items():
            processed_labels[label_data == old_label] = new_label
        
        known_indices = np.where(processed_labels > 0)
        known_indices = np.array(known_indices).T
        
        return processed_labels, known_indices
    
    def extract_hsi_patches(self, hsi_data, processed_labels, known_indices):
        """提取HSI patches"""
        H, W, D = hsi_data.shape
        pad_size = self.patch_size // 2
        
        padded_data = np.pad(hsi_data,
                            ((pad_size, pad_size),
                             (pad_size, pad_size),
                             (0, 0)),
                            mode='reflect')
        
        num_samples = len(known_indices)
        patches = np.zeros((num_samples, self.patch_size, self.patch_size, D))
        patch_labels = np.zeros(num_samples, dtype=int)
        
        for i, (row, col) in enumerate(known_indices):
            r_start = row
            r_end = row + self.patch_size
            c_start = col
            c_end = col + self.patch_size
            patch = padded_data[r_start:r_end, c_start:c_end, :]
            
            patches[i] = patch
            patch_labels[i] = processed_labels[row, col]
        
        return patches, patch_labels-1
    
    def process_background(self, label_data, processed_labels,class_index):
        """
        处理背景点
        
        参数:
            label_data: 原始标签数据
            processed_labels: 处理后的标签数据
        
        返回:
            processed_labels: 修改后的标签数据
        """
        # 深拷贝标签数据
        processed_labels = processed_labels.copy()
        
        # 获取背景点（标签值为0的点）
        background_indices = np.where(processed_labels == 0)
        
        # 打印处理前的信息
        print("\n背景点处理前:")
        print(f"标签唯一值: {np.unique(processed_labels)}")
        print(f"总背景点数量: {len(background_indices[0])}")
        
        if len(background_indices[0]) > self.bg_num:
            # 随机选择指定数量的背景点
            selected_indices = np.random.choice(
                len(background_indices[0]), 
                self.bg_num, 
                replace=False
            )
            # 将选中的背景点标记为第6类
            selected_bg = (
                background_indices[0][selected_indices],
                background_indices[1][selected_indices]
            )
            processed_labels[selected_bg] = class_index*np.ones_like(processed_labels[selected_bg])
        else:
            # 如果背景点数量不足，使用所有背景点
            processed_labels[background_indices] = class_index*np.ones_like(processed_labels[selected_bg])
        
        # 打印处理后的信息
        print("\n背景点处理后:")
        print(f"标签唯一值: {np.unique(processed_labels)}")
        print(f"类别分布: {np.bincount(processed_labels.flatten())}")
        
        return processed_labels
    
    def create_data_loaders(self):
        """创建训练、验证和测试数据加载器"""
        try:
            # 加载数据
            hsi_data_train, label_data_train = self.load_hsi_data(
                self.hsi_path_train, self.label_path_train)
            hsi_data_test, label_data_test = self.load_hsi_data(
                self.hsi_path_test, self.label_path_test)
                
            # 保存原始标签形状，用于后续绘制分类图
            self.train_label_shape = label_data_train.shape
            self.test_label_shape = label_data_test.shape
            
            # 如果测试数据是145维，对训练数据进行PCA降维
            if hsi_data_train.shape[-1] == 144:
                # 获取训练数据的形状
                H_train, W_train, D_train = hsi_data_train.shape
                # 获取目标维度
                target_dim = hsi_data_test.shape[-1]
                
                # 重塑训练数据以进行PCA
                hsi_data_train_reshaped = hsi_data_train.reshape(-1, D_train)
                
                # 创建PCA对象并拟合转换
                pca = PCA(n_components=target_dim)
                hsi_data_train_pca = pca.fit_transform(hsi_data_train_reshaped)
                
                # 将降维后的数据重塑回原始空间维度
                hsi_data_train = hsi_data_train_pca.reshape(H_train, W_train, target_dim)
                
                print(f"已将训练数据从{D_train}维降至{target_dim}维，与测试数据维度匹配")
                
            if hsi_data_test.shape[-1]==48:
                ind_1=np.where(label_data_test==6)
                label_data_test[ind_1[0],ind_1[1]]=0*np.ones_like(label_data_test[ind_1[0],ind_1[1]])
                ind_2=np.where(label_data_train==7)
                label_data_train[ind_2[0],ind_2[1]]=np.zeros_like(label_data_train[ind_2[0],ind_2[1]])
                ind_3=np.where(label_data_train==6)
                label_data_train[ind_3[0],ind_3[1]]=np.zeros_like(label_data_train[ind_3[0],ind_3[1]])
            if hsi_data_test.shape[-1]==102:
                ind_1=np.where(label_data_test==7)
                label_data_test[ind_1[0],ind_1[1]]=6*np.ones_like(label_data_test[ind_1[0],ind_1[1]])
                ind_2=np.where(label_data_train==7)
                label_data_train[ind_2[0],ind_2[1]]=np.zeros_like(label_data_train[ind_2[0],ind_2[1]])
                ind_3=np.where(label_data_train==6)
                label_data_train[ind_3[0],ind_3[1]]=np.zeros_like(label_data_train[ind_3[0],ind_3[1]])
            if hsi_data_test.shape[-1]==145:
                ind_1=np.where(label_data_test==9)
                label_data_test[ind_1[0],ind_1[1]]=6*np.ones_like(label_data_test[ind_1[0],ind_1[1]])
                ind_2=np.where(label_data_test==8)
                label_data_test[ind_2[0],ind_2[1]]=6*np.ones_like(label_data_test[ind_2[0],ind_2[1]])
                ind_3=np.where(label_data_test==7)
                label_data_test[ind_3[0],ind_3[1]]=6*np.ones_like(label_data_test[ind_3[0],ind_3[1]])
                ind_4=np.where(label_data_train==9)
                label_data_train[ind_4[0],ind_4[1]]=np.zeros_like(label_data_train[ind_4[0],ind_4[1]])
                ind_5=np.where(label_data_train==8)
                label_data_train[ind_5[0],ind_5[1]]=np.zeros_like(label_data_train[ind_5[0],ind_5[1]])
                ind_6=np.where(label_data_train==7)
                label_data_train[ind_6[0],ind_6[1]]=np.zeros_like(label_data_train[ind_6[0],ind_6[1]])
                ind_7=np.where(label_data_train==6)
                label_data_train[ind_7[0],ind_7[1]]=np.zeros_like(label_data_train[ind_7[0],ind_7[1]])
            if hsi_data_test.shape[-1]==50:
                #调整目标域标签 [未知类包含：第3，4，5，6，9，11，12，13，16，17，18，19，20]
                unk_class=[3,4,5,6,9,11,12,13,16,17,18,19,20]
                for i  in unk_class:
                    ind=np.where(label_data_test==i)
                    label_data_test[ind[0],ind[1]]=8*np.ones_like(label_data_test[ind[0],ind[1]]) 
                #重新排序剩余类别标签              
                ind_1=np.where(label_data_test==7)
                label_data_test[ind_1[0],ind_1[1]]=3*np.ones_like(label_data_test[ind_1[0],ind_1[1]])
                ind_2=np.where(label_data_test==8)
                label_data_test[ind_2[0],ind_2[1]]=4*np.ones_like(label_data_test[ind_2[0],ind_2[1]])
                ind_3=np.where(label_data_test==10)
                label_data_test[ind_3[0],ind_3[1]]=5*np.ones_like(label_data_test[ind_3[0],ind_3[1]])
                ind_4=np.where(label_data_test==14)
                label_data_test[ind_4[0],ind_4[1]]=6*np.ones_like(label_data_test[ind_4[0],ind_4[1]])
                ind_5=np.where(label_data_test==15)
                label_data_test[ind_5[0],ind_5[1]]=7*np.ones_like(label_data_test[ind_5[0],ind_5[1]])

                #调整源域标签，去除第[3,4,5,8,12,13,14,15]类
                delet_class=[3,4,5,8,12,13,14,15]
                for i  in  delet_class:
                    ind=np.where(label_data_train==i)
                    label_data_train[ind[0],ind[1]]=np.zeros_like(label_data_test[ind[0],ind[1]]) 
                ind_6=np.where(label_data_train==6)
                label_data_train[ind_6[0],ind_6[1]]=3*np.ones_like(label_data_train[ind_6[0],ind_6[1]])
                ind_7=np.where(label_data_train==7)
                label_data_train[ind_7[0],ind_7[1]]=4*np.ones_like(label_data_train[ind_7[0],ind_7[1]])
                ind_8=np.where(label_data_train==9)
                label_data_train[ind_8[0],ind_8[1]]=5*np.ones_like(label_data_train[ind_8[0],ind_8[1]])
                ind_9=np.where(label_data_train==10)
                label_data_train[ind_9[0],ind_9[1]]=6*np.ones_like(label_data_train[ind_9[0],ind_9[1]])
                ind_10=np.where(label_data_train==11)
                label_data_train[ind_10[0],ind_10[1]]=7*np.ones_like(label_data_train[ind_10[0],ind_10[1]])
              
            
            
            # 处理标签
            processed_labels_train, _ = self.process_hsi_labels(label_data_train)
            processed_labels_test, _ = self.process_hsi_labels(label_data_test)
            
            # 处理背景点
            if hsi_data_test.shape[-1]==50:
              processed_labels_train = self.process_background(
                label_data_train, processed_labels_train,8)
            else:
              processed_labels_train = self.process_background(
                label_data_train, processed_labels_train,6)

            
            
            # 获取所有非零标签的索引（包括背景类6）
            known_indices_train = np.where(processed_labels_train > 0)
            known_indices_train = np.array(known_indices_train).T
            
            known_indices_test = np.where(processed_labels_test > 0)
            known_indices_test = np.array(known_indices_test).T
            
            # 保存测试集索引映射，用于后续绘制分类图
            self.test_indices = list(zip(known_indices_test[:, 0], known_indices_test[:, 1]))
            
            # 提取patches
            patches_train, patch_labels_train = self.extract_hsi_patches(
                hsi_data_train, processed_labels_train, known_indices_train)
            patches_test, patch_labels_test = self.extract_hsi_patches(
                hsi_data_test, processed_labels_test, known_indices_test)
            
            # 打印数据统计信息
            print("\n数据集标签分布:")
            print("训练集:", np.bincount(patch_labels_train))
            print("测试集:", np.bincount(patch_labels_test))
            
            # 随机划分训练集和验证集
            num_samples = len(patches_train)
            num_val = int(num_samples * self.val_ratio)
            
            # 生成随机索引
            indices = np.random.permutation(num_samples)
            val_indices = indices[:num_val]
            train_indices = indices[num_val:]
            
            # 保存验证集索引映射，用于后续绘制分类图
            self.val_indices = [
                (known_indices_train[val_indices[i], 0], known_indices_train[val_indices[i], 1]) 
                for i in range(len(val_indices))
            ]
            
            # 划分训练集和验证集
            patches_val = patches_train[val_indices]
            patch_labels_val = patch_labels_train[val_indices]
            patches_train_final = patches_train[train_indices]
            patch_labels_train_final = patch_labels_train[train_indices]
            
            # 验证数据集大小
            print("\n数据集大小:")
            print(f"训练集: {len(patches_train_final)} 样本")
            print(f"验证集: {len(patches_val)} 样本")
            print(f"测试集: {len(patches_test)} 样本")
            
            # 验证标签分布
            print("\n类别分布:")
            print("训练集:", np.bincount(patch_labels_train_final))
            print("验证集:", np.bincount(patch_labels_val))
            print("测试集:", np.bincount(patch_labels_test))
            
            # 创建数据集
            train_dataset = HSIDataset(patches_train_final, patch_labels_train_final)
            val_dataset = HSIDataset(patches_val, patch_labels_val)
            test_dataset = HSIDataset(patches_test, patch_labels_test)
            
            # 计算每个类别的样本数量
            train_class_counts = {label: len(indices) for label, indices in train_dataset.class_indices.items()}
            print(f"训练集各类别样本数量: {train_class_counts}")
            
            # 确保batch_size能被类别数整除
            num_classes = len(train_dataset.class_indices)
            if self.batch_size % num_classes != 0:
                adjusted_batch_size = (self.batch_size // num_classes) * num_classes
                print(f"警告: batch_size ({self.batch_size}) 不能被类别数 ({num_classes}) 整除")
                print(f"调整batch_size为: {adjusted_batch_size}")
                self.batch_size = adjusted_batch_size
            
            # 在创建采样器之前添加验证
            min_samples_per_class = min(len(indices) for indices in train_dataset.class_indices.values())
            samples_per_class_per_batch = self.batch_size // train_dataset.num_classes
            
            if min_samples_per_class < samples_per_class_per_batch:
                print(f"警告：某些类别的样本数 ({min_samples_per_class}) 少于每批次所需的样本数 ({samples_per_class_per_batch})")
                print("将通过重复采样来填充批次")
            
            # 创建平衡的批次采样器
            train_batch_sampler = BalancedBatchSampler(train_dataset, self.batch_size)
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=patches_test.shape[0],
                shuffle=False,
                num_workers=1,
                pin_memory=True
            )
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"创建数据加载器时发生错误: {str(e)}")
            raise

class HSIDataset(Dataset):
    """高光谱图像数据集类"""
    def __init__(self, patches, labels, transform=None):
        self.patches = torch.from_numpy(patches).float()
        self.patches = self.patches.permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
        
        # 按类别索引样本
        self.class_indices = {}
        for i in range(len(self.labels)):
            label = self.labels[i].item()
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)
        
        self.num_classes = len(self.class_indices)
        self.samples_per_class = min(len(indices) for indices in self.class_indices.values())
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch, label

class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    每个batch中包含相同数量的各类样本的采样器
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 确保batch_size能被类别数整除
        if batch_size % dataset.num_classes != 0:
            raise ValueError(f"batch_size ({batch_size}) 必须能被类别数 ({dataset.num_classes}) 整除")
        
        self.samples_per_class_per_batch = batch_size // dataset.num_classes
        
        # 计算可以生成的最大批次数
        min_samples_per_class = min(len(indices) for indices in dataset.class_indices.values())
        max_batches_per_class = min_samples_per_class // self.samples_per_class_per_batch
        
        # 确保至少有一个批次
        self.num_batches = max(1, max_batches_per_class)
        
        print(f"每个类别的最小样本数: {min_samples_per_class}")
        print(f"每批次每类样本数: {self.samples_per_class_per_batch}")
        print(f"总批次数: {self.num_batches}")
    
    def __iter__(self):
        # 为每个类别创建索引迭代器
        class_iterators = {}
        for class_idx, indices in self.dataset.class_indices.items():
            indices = np.array(indices)
            # 无限循环采样该类别的索引
            class_iterators[class_idx] = cycle_indices(indices)
        
        # 生成batch_size个样本的批次
        for _ in range(self.num_batches):
            batch_indices = []
            # 从每个类别中抽取相同数量的样本
            for class_idx in sorted(self.dataset.class_indices.keys()):
                class_iter = class_iterators[class_idx]
                batch_indices.extend([next(class_iter) for _ in range(self.samples_per_class_per_batch)])
            
            # 打乱batch内的顺序
            np.random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return self.num_batches

def cycle_indices(indices):
    """
    无限循环生成索引
    """
    indices = indices.copy()
    while True:
        np.random.shuffle(indices)
        for idx in indices:
            yield idx

# 使用示例
# if __name__ == "__main__":
#     # 初始化数据管理器
#     data_manager = HSIDataManager(
#         hsi_path_train="dataset/Pavia/paviaC.mat",
#         label_path_train="dataset/Pavia/paviaC_7gt.mat",
#         hsi_path_test="dataset/Pavia/paviaU.mat",
#         label_path_test="dataset/Pavia/paviaU_7gt.mat",
#         patch_size=7,
#         batch_size=32,
#         num_workers=4,
#         bg_num=1000,
#         val_ratio=0.1
#     )
    
#     try:
#         # 创建数据加载器
#         train_loader, val_loader, test_loader = data_manager.create_data_loaders()
        
#         print("数据加载器创建成功！")
        
#         # 测试数据加载器
#         for name, loader in [("训练", train_loader), 
#                            ("验证", val_loader), 
#                            ("测试", test_loader)]:
#             for batch_idx, (data, target) in enumerate(loader):
#                 print(f"{name}批次 {batch_idx}:")
#                 print(f"数据形状: {data.shape}")
#                 print(f"标签形状: {target.shape}")
#                 break
            
#     except Exception as e:
#         print(f"发生错误: {str(e)}") 