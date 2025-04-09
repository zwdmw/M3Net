import torch
import torch.nn as nn
from torch.optim import Adam
import math
import torch.nn.functional as F

from mamba import ModelArgs,Mamba
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return nn.Tanh()( self.fc2(x))

class PatchEmbed(nn.Module):
    """将输入的HSI patch转换为序列"""
    def __init__(self, patch_size=7, in_channels=103, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, 
                             stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, 1, 1)
        return x.flatten(2).transpose(1, 2)  # (B, 1, embed_dim)

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, embed_dim=512, num_heads=1, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class HSIViT(nn.Module):
    """HSI Vision Transformer特征提取器"""
    def __init__(self, patch_size=7, in_channels=103, embed_dim=256, 
                 depth=1, num_heads=4, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化位置编码
        self._init_weights()

    def _init_weights(self):
        pos_embed = self.pos_embed
        pos_embed.data.normal_(std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # 返回[CLS]令牌的特征

class APN(nn.Module):
  
    def __init__(self, in_features=256, hidden_dim=128, num_classes=7, num_steps=3, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.dropout = dropout
        
        self.feature_net = nn.Sequential(
            nn.Linear(in_features+num_classes, hidden_dim),
         
            nn.LeakyReLU(0.1),  # 使用LeakyReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
           
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
       

        self.var = nn.Sequential(
          
            nn.Linear(hidden_dim , num_classes),
            nn.Softplus()
        
        )
        self.mu= nn.Sequential(
        
            nn.Linear(hidden_dim , num_classes)
            )
        self.arg = ModelArgs(d_model=in_features ,
                    n_layer=1,
                    vocab_size=in_features)
  
        self.mamba = Mamba(self.arg)

        
     
    def forward(self, features, meta=False):
        batch_size = features.shape[0]
        # 将零向量初始化改为随机初始化
        y_t = nn.Softmax(1)(torch.randn(batch_size, self.num_classes, device=features.device) )
        
        all_logits = []  # 存储所有步骤的预测
        all_state=[]
        all_alpha=[]
        all_mu=[]
        all_var=[]
        all_action=[]
        features=(self.mamba(features.unsqueeze(0)).squeeze(0))
        
        # 迭代细化预测
        for step in range(self.num_steps):
           
            # 连接特征和当前预测
            combined = torch.cat([(features), y_t], dim=1)
            
            # 特征提取
            hidden = self.feature_net(combined)
            
           
            
            
            mu = self.mu(hidden) 
            var=self.var(hidden)
            
           
            action_dist = torch.distributions.Normal(mu, scale=var)
            # 从分布中采样
            if self.training:
                action = action_dist.rsample()
                
            else:
                action = action_dist.rsample()
            
            # 更新预测
            y_t = y_t -action
            all_logits.append(y_t.clone())
            all_state.append(combined.clone().unsqueeze(0))
            all_mu.append(mu.clone().unsqueeze(0))
            all_var.append(var.clone().unsqueeze(0))
            all_action.append(action.clone().unsqueeze(0))
            
        
        # 返回最终预测和所有步骤的预测
        if self.training:
            all_alpha.append(all_mu)
            all_alpha.append(all_var)
            return y_t, all_logits,all_state,all_alpha,all_action
        
        else:
            return y_t,all_logits

class MLPFeatureExtractor(nn.Module):
    """基于MLP的特征提取器"""
    def __init__(self, in_channels=103, embed_dim=256, hidden_dims=[512, 1024, 512]):
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_flatten = nn.Flatten()  # 添加Flatten层
        
        layers = []
        input_dim = in_channels  # 第一层的输入维度是通道数
        
        # 构建MLP层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], embed_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        # 输入x的形状为(B, C, H, W)
        B, C, H, W = x.shape
        # 将空间维度展平
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = x.reshape(B, H*W, C)    # (B, H*W, C)
        x = x.mean(dim=1)           # (B, C) - 对空间维度取平均
        
        # 通过MLP
        x = self.mlp(x)            # (B, embed_dim)
        return x

class HSIModel(nn.Module):
    """增强版HSI分类模型"""
    def __init__(self, model_type='vit', patch_size=7, in_channels=103, 
                 num_classes=7, embed_dim=256, hidden_dims=[512, 1024, 512],
                 num_refine_steps=6, dropout=0.2):
        super().__init__()
        
        self.model_type = model_type
        
        # 根据model_type选择特征提取器
        if model_type == 'vit':
            self.feature_extractor = HSIViT(
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
                dropout=dropout
            )
        elif model_type == 'mlp':
            self.feature_extractor = MLPFeatureExtractor(
                in_channels=in_channels,
                embed_dim=embed_dim,
                hidden_dims=hidden_dims
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        
        self.policy = APN(
            in_features=embed_dim,
            hidden_dim=64,
            num_classes=num_classes,
            num_steps=num_refine_steps,
            dropout=dropout
        )
        self.value_net = ValueNetwork(embed_dim+num_classes) 
       
    def forward(self, x):
        features = self.feature_extractor(x)
        
        if self.training:
            logits, all_logits,all_state,all_alpha,all_action = self.policy(features)
            return logits, all_logits,all_state,all_alpha,all_action
        else:
            logits = self.policy(features)
            return logits

def create_model_and_optimizer(model_type='vit', 
                             patch_size=7, 
                             in_channels=103, 
                             num_classes=7, 
                             embed_dim=256, 
                             hidden_dims=[512, 1024, 512],
                             learning_rate=1e-4,
                             weight_decay=1e-4,
                             num_refine_steps=3):
    """
    创建模型和优化器
    
    参数:
        model_type: 模型类型 ('vit' 或 'mlp')
        patch_size: patch大小
        in_channels: 输入通道数
        num_classes: 类别数
        embed_dim: 嵌入维度
        hidden_dims: MLP隐藏层维度列表
        learning_rate: 学习率
        weight_decay: 权重衰减
        num_refine_steps: 预测细化的迭代步数
    """
    model = HSIModel(
        model_type=model_type,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        num_refine_steps=num_refine_steps
    )
    
    optimizer = Adam(model.parameters(), 
                    lr=learning_rate,
                    weight_decay=weight_decay)
    
    return model, optimizer


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=0.001, dt_max=0.1, dt_init="random"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 投影到更高维度
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv-1,
            groups=self.d_inner
        )
        
        # S4D参数
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # SSM参数
        self.A = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        
        # 时间步长参数
        if dt_init == "random":
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        else:
            dt = torch.ones(self.d_inner) * dt_init
        self.dt = nn.Parameter(dt)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        residual = x
        x = self.norm(x)
        
        # 投影并分离
        x_proj = self.in_proj(x)  # [batch_size, seq_len, 2*d_inner]
        x_gate, x_proj = torch.chunk(x_proj, 2, dim=-1)  # 两个 [batch_size, seq_len, d_inner]
        
        # 卷积
        x_proj = x_proj.transpose(1, 2)  # [batch_size, d_inner, seq_len]
        x_conv = self.conv1d(x_proj)[:, :, :x.size(1)]  # 确保输出长度正确
        x_conv = x_conv.transpose(1, 2)  # [batch_size, seq_len, d_inner]
        
        # 计算时间步长
        dt = torch.exp(self.dt_proj(x_conv))
        
        # 状态空间模型计算（使用更高效的并行实现）
        x_ssm = self._ssm_forward(x_conv, dt)
        
        # 应用门控
        x_out = x_ssm * torch.sigmoid(x_gate)
        
        # 投影回原始维度
        x_out = self.out_proj(x_out)
        
        # 残差连接
        return x_out + residual
    
    def _ssm_forward(self, x, dt):
        """
        高效的SSM前向传播实现
        x: [batch_size, seq_len, d_inner]
        dt: [batch_size, seq_len, d_inner]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device  # 获取输入张量的设备
        
        # 离散化A
        # 对角矩阵exp(-dt * A)的计算
        A_discrete = torch.exp(-dt.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(0))
        
        # 初始化隐藏状态 - 确保在正确的设备上创建
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
        outputs = []
        
        # 序列扫描
        for t in range(seq_len):
            # 更新隐藏状态
            h = h * A_discrete[:, t]
            h = h + torch.einsum('bd,ds->bds', x[:, t], self.B)
            
            # 计算输出
            y = torch.einsum('bds,ds->bd', h, self.C)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # [batch_size, seq_len, d_inner]

# # 使用示例
# if __name__ == "__main__":
#     # ViT模型
#     model_vit, _ = create_model_and_optimizer(
#         model_type='vit',
#         patch_size=7,
#         in_channels=103,
#         num_classes=7
#     )
    
#     # MLP模型
#     model_mlp, _ = create_model_and_optimizer(
#         model_type='mlp',
#         patch_size=7,
#         in_channels=103,
#         num_classes=7,
#         hidden_dims=[512, 1024, 512]
#     )
    
#     # 测试前向传播
#     x = torch.randn(4, 103, 7, 7)
    
#     print("Testing ViT model:")
#     out_vit = model_vit(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {out_vit.shape}")
    
#     print("\nTesting MLP model:")
#     out_mlp = model_mlp(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {out_mlp.shape}")

# # 在配置中指定模型类型
# config = {
#     'model_type': 'mlp',  # 或 'vit'
#     'patch_size': 7,
#     'in_channels': 103,
#     'num_classes': 7,
#     'embed_dim': 256,
#     'hidden_dims': [512, 1024, 512],
#     'learning_rate': 1e-4,
#     'weight_decay': 1e-4
# }

# # 创建模型
# model, optimizer = create_model_and_optimizer(**config)
