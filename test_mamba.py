import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import Mamba  # 导入我们实现的Mamba模型

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 测试参数
batch_size = 8
seq_length = 100
feature_dim = 64
hidden_dim = 128
n_layers = 2

# 创建随机测试数据
def generate_test_data(batch_size, seq_length, feature_dim):
    # 创建随机特征序列
    x = torch.randn(batch_size, seq_length, feature_dim)
    # 创建目标值（这里我们简单地使用x的和作为目标）
    y = torch.sum(x, dim=-1, keepdim=True)
    return x, y

# 初始化模型
def test_model_initialization():
    print("测试模型初始化...")
    model = Mamba(
        d_model=feature_dim,
        n_layer=n_layers,
        d_state=16,
        d_conv=4,
        expand=2,
        output_dim=1  # 输出维度为1
    )
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")
    return model

# 测试前向传播
def test_forward_pass(model):
    print("\n测试前向传播...")
    x, y = generate_test_data(batch_size, seq_length, feature_dim)
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    
    # 将数据移动到与模型相同的设备上
    x = x.to(device)
    y = y.to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"目标形状: {y.shape}")
    
    return x, y, output

# 测试训练循环
def test_training(model, epochs=10):
    print("\n测试训练循环...")
    x, y = generate_test_data(batch_size, seq_length, feature_dim)
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    
    # 将数据移动到与模型相同的设备上
    x = x.to(device)
    y = y.to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练循环
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        output = model(x)
        
        # 计算损失
        loss = criterion(output, y)
        losses.append(loss.item())
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('mamba_training_loss.png')
    plt.close()
    
    print(f"训练完成，最终损失: {losses[-1]:.6f}")
    print(f"损失曲线已保存为 'mamba_training_loss.png'")
    
    return losses

# 测试序列预测
def test_sequence_prediction(model):
    print("\n测试序列预测...")
    # 创建一个简单的正弦波序列
    t = np.linspace(0, 4*np.pi, seq_length)
    sine_wave = np.sin(t)
    
    # 创建输入特征（使用滑动窗口）
    window_size = 10
    x_list = []
    y_list = []
    
    for i in range(len(sine_wave) - window_size):
        x_list.append(sine_wave[i:i+window_size])
        y_list.append(sine_wave[i+window_size])
    
    x = torch.tensor(np.array(x_list), dtype=torch.float32).unsqueeze(0)
    x = x.repeat(batch_size, 1, 1)
    
    # 打印形状以便调试
    print(f"x shape before padding: {x.shape}")
    
    # 添加额外的特征维度以匹配模型输入
    x_padded = torch.zeros(batch_size, x.shape[1], feature_dim)
    
    # 修复维度不匹配问题 - 只使用第一个特征
    x_padded[:, :, 0] = x[:, :, 0]
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    
    # 将数据移动到与模型相同的设备上
    x_padded = x_padded.to(device)
    
    # 前向传播
    with torch.no_grad():
        predictions = model(x_padded)
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(sine_wave, label='真实值')
    plt.plot(range(window_size, window_size + len(predictions[0])), 
             predictions[0, :, 0].cpu().numpy(), 'r--', label='预测值')
    plt.legend()
    plt.title('Mamba模型序列预测')
    plt.savefig('mamba_sequence_prediction.png')
    plt.close()
    
    print(f"序列预测结果已保存为 'mamba_sequence_prediction.png'")

# 主测试函数
def main():
    print("开始Mamba模型测试...")
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试模型初始化
    model = test_model_initialization()
    model = model.to(device)
    
    # 测试前向传播
    x, y, output = test_forward_pass(model)
    
    # 测试训练循环
    losses = test_training(model)
    
    # 测试序列预测
    test_sequence_prediction(model)
    
    print("\nMamba模型测试完成!")

if __name__ == "__main__":
    main() 