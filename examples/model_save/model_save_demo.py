import torch
import torch.nn as nn
import torch.optim as optim


# https://zhuanlan.zhihu.com/p/620688513


class Net(nn.Module):
    """Define a model"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = Net()

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loss = 0.1
    model_pt = './model_name.pt'
    # 保存pt模型
    torch.save(
        {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_pt)

    model_bin = './model_name.bin'
    # 保存bin模型
    torch.save(model.state_dict(), model_bin)

    model_script = './model_script.pt'
    # 将模型转换为TorchScript
    scripted_model = torch.jit.script(model)
    # 保存序列化的模型
    torch.jit.save(scripted_model, model_script)

    # -----------------------------------------------
    # 加载模型状态字典
    state_dict = torch.load(model_pt)
    model.load_state_dict(state_dict)

    # 加载模型权重
    weights = torch.load(model_bin)
    model.load_state_dict(weights)

    # 直接加载序列化的模型（不需要先定义模型结构）
    model = torch.jit.load(model_script)
