import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import time
from thop import profile 
import logging
from data_manager import HSIDataManager
from model import create_model_and_optimizer
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，必须在导入plt之前
import matplotlib.pyplot as plt
import os

def get_params():
    parser = argparse.ArgumentParser(description="M3Net")
    parser.add_argument("--seed", type=int, default=9809243, help='seed')
    parser.add_argument("--patch_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=200, help='batch_size')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bg_num", type=int, default=2000)
    parser.add_argument('--val_ratio', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--in_channels', type=int, default=48)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--num_refine_steps', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='HS')

    args, _  = parser.parse_known_args()
    return args
def merge_parameter(base_params, override_params):
    """
    Update the parameters in ``base_params`` with ``override_params``.
    Can be useful to override parsed command line arguments.

    Parameters
    ----------
    base_params : namespace or dict
        Base parameters. A key-value mapping.
    override_params : dict or None
        Parameters to override. Usually the parameters got from ``get_next_parameters()``.
        When it is none, nothing will happen.

    Returns
    -------
    namespace or dict
        The updated ``base_params``. Note that ``base_params`` will be updated inplace. The return value is
        only for convenience.
    """
    if override_params is None:
        return base_params
    is_dict = isinstance(base_params, dict)
    for k, v in override_params.items():
        if is_dict:
            if k not in base_params:
                raise ValueError('Key \'%s\' not found in base parameters.' % k)
            if type(base_params[k]) != type(v) and base_params[k] is not None:
                raise TypeError('Expected \'%s\' in override parameters to have type \'%s\', but found \'%s\'.' %
                                (k, type(base_params[k]), type(v)))
            base_params[k] = v
        else:
            if not hasattr(base_params, k):
                raise ValueError('Key \'%s\' not found in base parameters.' % k)
            if type(getattr(base_params, k)) != type(v) and getattr(base_params, k) is not None:
                raise TypeError('Expected \'%s\' in override parameters to have type \'%s\', but found \'%s\'.' %
                                (k, type(getattr(base_params, k)), type(v)))
            setattr(base_params, k, v)
    return base_params
    
class Trainer:
    def __init__(self, 
                 hsi_path_train,
                 label_path_train,
                 hsi_path_test,
                 label_path_test,
                 config):
        """
        初始化训练器
        
        参数:
            hsi_path_train: 训练数据路径
            label_path_train: 训练标签路径
            hsi_path_test: 测试数据路径
            label_path_test: 测试标签路径
            config: 配置字典，包含所有超参数
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        
        # 初始化数据管理器
        self.data_manager = HSIDataManager(
            hsi_path_train=hsi_path_train,
            label_path_train=label_path_train,
            hsi_path_test=hsi_path_test,
            label_path_test=label_path_test,
            patch_size=config['patch_size'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            bg_num=config['bg_num'],
            val_ratio=config['val_ratio']
        )
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = \
            self.data_manager.create_data_loaders()
        
        # 创建模型和优化器
        self.model, self.optimizer = create_model_and_optimizer(
            model_type=config.get('model_type', 'vit'),  # 默认使用ViT模型
            patch_size=config['patch_size'],
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            embed_dim=config.get('embed_dim', 256),  # 默认嵌入维度
            hidden_dims=config.get('hidden_dims', [512, 1024, 512]),  # MLP默认隐藏层
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            num_refine_steps=config['num_refine_steps']
        )
        self.model = self.model.to(self.device)
        input_size = (1, config['in_channels'], config['patch_size'], config['patch_size'])
        input = torch.randn(input_size).to(self.device)
        
        # 添加自定义操作计数
        custom_ops = {}
       
   
        
        # 创建学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.criterion1=nn.CrossEntropyLoss(reduction='none')
        
        # 设置日志
        # self.setup_logging()
        
        # 用于记录训练过程中的损失和准确率
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.val_os_scores = []
        
        # 创建保存图表的目录
        os.makedirs('plots', exist_ok=True)
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            filename=f'training_{time.strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
     
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
       
        num_steps = self.config['num_refine_steps']
        step_weights = torch.linspace(0.5, 1.0, num_steps).to(self.device)
        step_weights = step_weights / step_weights.sum()
        
    
        
        pbar = tqdm(self.train_loader, desc='Training')
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
         
            with torch.no_grad():
                final_pred, all_preds,all_state,all_alpha,all_action = self.model(data)
            
              
                step_losses = []
                for i, pred in enumerate(all_preds):
                    ce_loss = self.criterion1(pred, target)
                    
                    step_losses.append(step_weights[i] * ce_loss.unsqueeze(0))
                ce_total = sum(step_losses)

            
              
                all_rewards=-torch.cat(step_losses)
                
                # 使用价值网络预测状态价值
                state_values = self.model.value_net(torch.cat(all_state, 0))
                state_values = state_values.view(all_rewards.size())
                
                # 计算优势函数
                gamma =params['gamma']# 折扣因子
                lambda_gae = 0.95  # GAE参数
                
                advantages = torch.zeros_like(all_rewards)
                returns = torch.zeros_like(all_rewards)
                last_gae = 0
                last_return = 0
                
                # 从后向前计算GAE和累积回报
                for t in reversed(range(all_rewards.size(0))):
                    if t == all_rewards.size(0) - 1:
                        next_value = 0  # 最后一步的下一个状态值为0
                    else:
                        next_value = state_values[t + 1]
                    
                    # 计算TD误差
                    delta = all_rewards[t] + gamma * next_value - state_values[t]
                    
                    # 计算GAE
                    advantages[t] = last_gae = delta + gamma * lambda_gae * last_gae
                    
                    # 计算累积回报（用于价值网络训练）
                    returns[t] = all_rewards[t] + gamma * last_return
                    last_return = returns[t]
                
                # 标准化优势函数
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                mu= torch.cat(all_alpha[0],0)
                var= torch.cat(all_alpha[1],0)
                action=torch.cat(all_action,0)
                

              

                action_dists = torch.distributions.Normal(mu, scale=var)

               
               
                old_log_probs = action_dists.log_prob(action)
            for KK in range(10):
                final_pred, all_preds,all_state,all_alpha,all_action= self.model(data)
            
              
                step_losses = []
                for i, pred in enumerate(all_preds):
                    ce_loss = self.criterion(pred, target)
                    step_losses.append(step_weights[i] * ce_loss.unsqueeze(0))
                ce_total = sum(step_losses)
            
              
                mu= torch.cat(all_alpha[0],0)
                var= torch.cat(all_alpha[1],0)



                

                action_dists = torch.distributions.Normal(mu, scale=var)

                # 确保action是有效的概率分布
                new_log_probs = action_dists.log_prob(action)
                ratio = torch.exp(new_log_probs - old_log_probs)  # (T, 100)
                clipped_ratio = torch.clamp(ratio, 1. - 0.4, 1. +0.4)

                surr1 = ratio * advantages.unsqueeze(-1).expand_as(ratio)
                surr2 = clipped_ratio * advantages.unsqueeze(-1).expand_as(ratio)
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                  # 计算价值网络损失
                state_values = self.model.value_net(torch.cat(all_state, 0))
                state_values = state_values.view(returns.size())
                value_loss = F.mse_loss(state_values, returns)
              
                
                # 组合所有损失
                loss = policy_loss+ 0.1*(value_loss) 
                self.optimizer.zero_grad()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                loss.backward()
                self.optimizer.step()          
            total_loss += loss.item()
            # 使用最终预测计算准确率
            pred = final_pred.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 更新进度条显示
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
             
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), correct / total
    

    def evaluate(self, loader, phase='val'):
        """
        评估模型，包括开放集识别指标和每个时间步的预测精度
        
        参数:
            loader: 数据加载器
            phase: 评估阶段 ('val' 或 'test')
        """
        self.model.eval()
        total_loss = 0
        
        # 收集所有预测和真实标签
        all_preds = []
        all_targets = []
        
        # 收集每个时间步的预测
        step_predictions = []
        start_time = time.time()
        
        with torch.no_grad():
            for data, target in tqdm(loader, desc=f'{phase.capitalize()} Evaluation'):
                data, target = data.to(self.device), target.to(self.device)
                
                # 获取模型输出
                output, all_logits = self.model(data)
                
                # 如果输出是元组，取第一个元素（最终预测）
                if isinstance(output, tuple):
                    final_pred = output[0]
                else:
                    final_pred = output
                    
                loss = self.criterion(final_pred, target)
                
                total_loss += loss.item()
                pred = final_pred.argmax(dim=1)
                
                # 收集预测和真实标签
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
                
                # 收集每个时间步的预测
                batch_step_preds = []
                for step_logits in all_logits:
                    step_pred = step_logits.argmax(dim=1).cpu()
                    batch_step_preds.append(step_pred)
                
               
                if len(step_predictions) == 0:
                    step_predictions = [[] for _ in range(len(all_logits))]
                
             
                for i, step_pred in enumerate(batch_step_preds):
                    step_predictions[i].append(step_pred)
        
        # 合并所有批次的预测和标签
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        # 合并每个时间步的预测
        for i in range(len(step_predictions)):
            step_predictions[i] = torch.cat(step_predictions[i])
        
        # 计算混淆矩阵
        num_classes = self.config['num_classes']
        confusion_matrix = torch.zeros(num_classes, num_classes)
        for t, p in zip(all_targets.view(-1), all_preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        
        confusion_matrix = confusion_matrix.numpy()
        
        # 计算总体精度（所有样本中预测正确的比例）
        overall_acc = (all_preds == all_targets).float().mean().item()
        
        # 计算每类精度时处理可能的零除
        class_acc = np.zeros(num_classes)
        for i in range(num_classes):
            if confusion_matrix[i].sum() > 0:  # 只在类别有样本时计算精度
                class_acc[i] = confusion_matrix[i, i] / confusion_matrix[i].sum()
            else:
                class_acc[i] = 0  # 或者设置为nan: np.nan
                print(f"Warning: Class {i} has no samples")

     
        
        
        # 计算Kappa系数
        n = confusion_matrix.sum()
        sum_po = confusion_matrix.diagonal().sum()
        sum_pe = sum(confusion_matrix.sum(axis=0) * confusion_matrix.sum(axis=1))
        po = sum_po / n
        pe = sum_pe / (n * n)
        kappa = (po - pe) / (1 - pe)
        
        # 假设最后一类是未知类
        known_classes = num_classes - 1
        
        # 计算开放集识别指标
        # 1. 计算已知类的准确率
        known_acc = np.zeros(known_classes)
        for i in range(known_classes):
            if confusion_matrix[i].sum() > 0:
                known_acc[i] = confusion_matrix[i, i] / confusion_matrix[i].sum()
            else:
                known_acc[i] = 0
        
        # 2. 计算未知类的准确率
        unknown_acc = confusion_matrix[-1, -1] / confusion_matrix[-1].sum() if confusion_matrix[-1].sum() > 0 else 0
        
        # 3. 计算OS (OpenSet Classification Rate)
        os_score = (np.mean(known_acc) + unknown_acc) / 2
        
        # 4. 计算OS* (Normalized OpenSet Classification Rate)
        valid_known_classes = sum(confusion_matrix[:known_classes].sum(axis=1) > 0)
        valid_known_acc = [acc for i, acc in enumerate(known_acc) if confusion_matrix[i].sum() > 0]
        os_star = np.mean(valid_known_acc) if valid_known_acc else 0
        
        # 5. 计算HOS (Harmonic OpenSet Classification Rate)
        mean_known_acc = np.mean(valid_known_acc) if valid_known_acc else 0
        hos = 2 * (mean_known_acc * unknown_acc) / (mean_known_acc + unknown_acc) if (mean_known_acc + unknown_acc) > 0 else 0
        # if os_score > 0.75:
        #     self.plot_step_improvements(step_class_accs, step_overall_accs, phase)
        metrics = {
            'loss': total_loss / len(loader),
            'overall_acc': overall_acc,
            'class_acc': class_acc,
            'kappa': kappa,
            'os': os_score,
            'os_star': os_star,
            'hos': hos,
            'unknown_acc': unknown_acc,
            'known_acc_mean': mean_known_acc,
            'all_preds': all_preds,  # 添加预测结果
        'all_targets': all_targets  # 添加真实标签
          
        }
        
        # 打印详细的评估结果
        print(f"\n{phase.capitalize()} Results:")
        print(f"Overall Accuracy: {overall_acc*100:.2f}%")
        print(f"Kappa: {kappa:.4f}")
        

        print("\nClass-wise Accuracies:")
        for i, acc in enumerate(class_acc):
            if np.isnan(acc):
                print(f"Class {i}: N/A (no samples)")
            else:
                print(f"Class {i}: {acc*100:.2f}%")
        print("\nConfusion Matrix:")
        print(confusion_matrix)
        
        # 打印每个类别的样本数
        class_samples = confusion_matrix.sum(axis=1)
        print("\nSamples per class:")
        for i, samples in enumerate(class_samples):
            print(f"Class {i}: {int(samples)}")
        
        # 打印开放集识别指标
        print(f"\nOpenSet Classification Metrics:")
        print(f"OS: {os_score*100:.2f}%")
        print(f"OS*: {os_star*100:.2f}%")
        print(f"HOS: {hos*100:.2f}%")
        print(f"Unknown Class Accuracy: {unknown_acc*100:.2f}%")
        print(f"Known Class Mean Accuracy: {mean_known_acc*100:.2f}%")
        
        return metrics
    
    def train(self):
        """训练模型"""
        best_val_acc = 0
        patience = self.config['patience']
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            start_time_epoch = time.time()

            # 训练阶段
            train_loss, train_acc = self.train_epoch()
            
            # 验证阶段
            if epoch % 1== 0 :
                val_metrics = self.evaluate(self.val_loader, 'val')
                # 记录训练和验证指标
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.val_losses.append(val_metrics["loss"])
                self.val_accs.append(val_metrics["overall_acc"])
                self.val_os_scores.append(0.5*(val_metrics['os_star']+val_metrics['os']))
                
                # 更新学习率
                self.scheduler.step(val_metrics['overall_acc'])
                
                # 记录日志
                log_message = [
                    f'Epoch {epoch+1}/{self.config["epochs"]}',
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%',
                    f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["overall_acc"]*100:.2f}%',
                    f'Val Kappa: {val_metrics["kappa"]:.4f}',
                    'Class Accuracies:'
                ]
                
                # 添加每类精度
                for i, acc in enumerate(val_metrics['class_acc']):
                    log_message.append(f'  Class {i}: {acc*100:.2f}%')
                
                # log_message = ' - '.join(log_message)
                # logging.info(log_message)
                print(log_message)
               
                
                # 保存最佳模型
                if 0.5*(val_metrics['os_star']+val_metrics['os']) > best_val_acc:
                    best_val_acc = 0.5*(val_metrics['os_star']+val_metrics['os'])
                    
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
               
            
            
            
           
        
        # 加载最佳模型进行测试

      
        
        

        self.model.load_state_dict(torch.load('best_model.pth'))
        test_metrics = self.evaluate(self.test_loader, 'test')
    #     self.plot_classification_map(
    #     test_metrics['all_preds'], 
    #     test_metrics['all_targets'], 
    #     phase='test'
    # )
    
        
        # 绘制训练损失和准确率曲线
        # self.plot_training_curves()
        
        # 记录测试结果
        log_message = [
            'Test Results:',
            f'Loss: {test_metrics["loss"]:.4f}',
            f'Overall Accuracy: {test_metrics["overall_acc"]*100:.2f}%',
            f'Kappa: {test_metrics["kappa"]:.4f}',
            'Class Accuracies:'
        ]
        
        for i, acc in enumerate(test_metrics['class_acc']):
            log_message.append(f'  Class {i}: {acc*100:.2f}%')
        
        log_message = '\n'.join(log_message)
        logging.info(log_message)
        print(log_message)

    

if __name__ == '__main__':

    import argparse
    
   
       

    params = vars(get_params())

    print(params)
    
    # 配置参数
    dataset=params['dataset']
    
    # 基础配置
    base_config = {
        'seed': params['seed'],
        'patch_size': params['patch_size'],
        'batch_size': params['batch_size'],
        'num_workers': params['num_workers'],
        'bg_num': params['bg_num'],
        'val_ratio': params['val_ratio'],
        'in_channels': params['in_channels'],
        'num_classes': params['num_classes'],
        'learning_rate': params['learning_rate'],
        'weight_decay': params['weight_decay'],
        'epochs': params['epochs'],
        'patience': params['patience'],
        'num_refine_steps': params['num_refine_steps'],
        # 模型特定配置
        'model_type': 'vit',  # 或 'vit'
        'embed_dim': params['embed_dim'],
        'dataset': params['dataset'],
        #'9809243,5224391,2408820,9476142,  6894645,9337188,8086717,1439101,1762348,1607451
    }
    
    
    if dataset=='HS':
        config = base_config
        print(config)
        trainer = Trainer(
            hsi_path_train="../dataset/Houston/Houston13.mat",
            label_path_train="../dataset/Houston/Houston13_7gt.mat",
            hsi_path_test="../dataset/Houston/Houston18.mat",
            label_path_test="../dataset/Houston/Houston18_7gt.mat",
            config=config)

    if dataset=='Pavia':
        # 对于Pavia数据集的特定配置
        pavia_config = base_config.copy()
        pavia_config.update({
         
         
         
            'in_channels': 102,  
            'patch_size': 3,
            'batch_size': 700,
            'learning_rate': 0.00005,
            'bg_num': 9000,
            'seed': 7745412,
  
            
            
           
            
        
             
                       
             #92385,5996913,8516453,7271840,4003445,3181729,8599124,5074018,2764649,5791717
        })
        config = pavia_config
        print(config)
    
        # 初始化训练器
        trainer = Trainer(
            hsi_path_train="../dataset/Pavia/paviaC.mat",
            label_path_train="../dataset/Pavia/paviaC_7gt.mat",
            hsi_path_test="../dataset/Pavia/paviaU.mat",
            label_path_test="../dataset/Pavia/paviaU_7gt.mat",
            
            
            config=config
        )
    if dataset=='BOT':
        # 对于Pavia数据集的特定配置
        bot_config = base_config.copy()
        bot_config.update({
        
            'in_channels': 145, 
             'patch_size': 1,
            'bg_num': 5000,
            'seed': 1642238,
             
            
                       
 
             #5771429,199101,1058808,8677583,7229492,7654172,7955524,5137355,202785,1814895
        })
        config = bot_config
        print(config)
    
        # 初始化训练器
        trainer = Trainer(
            hsi_path_train="../dataset/BOT/BOT5.mat",
            label_path_train="../dataset/BOT/BOT5_gt.mat",
            hsi_path_test="../dataset/BOT/BOT6.mat",
            label_path_test="../dataset/BOT/BOT6_gt.mat",
            
            
            config=config
        )
    # if dataset=='HS_all':
    #     # 对于Pavia数据集的特定配置
    #     HS_all_config = base_config.copy()
    #     HS_all_config.update({
    #         'patch_size': 3,
    #         'batch_size': 2000,
    #         'in_channels': 50,   
    #         'seed': 92385,  
    #         'num_classes': 8,
    #         'bg_num': 50000,             
 
    #          #92385,5996913,8516453,7271840,4003445,3181729,8599124,5074018,2764649,5791717
    #     })
    #     config = HS_all_config
    #     print(config)
    
    #     # 初始化训练器
    #     trainer = Trainer(
    #         hsi_path_train="dataset/HS_all/Houston13.mat",
    #         label_path_train="dataset/HS_all/Houston13_gt.mat",
    #         hsi_path_test="dataset/HS_all/Houston18.mat",
    #         label_path_test="dataset/HS_all/Houston18_gt.mat",
           
    #         config=config
    #     )
   
    # 打印模型信息
    print(f"\nUsing Model Type: {config['model_type']}")
    print(f"Model Parameters Total: {sum(p.numel() for p in trainer.model.parameters()) / 1e6:.2f}M")
    
    # 开始训练
    trainer.train()
  
