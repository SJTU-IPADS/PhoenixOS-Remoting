import torch
import torch.nn as nn
from utils.readData import read_dataset
from utils.ResNet import ResNet18
import numpy as np
import time
import sys
import ctypes
import os

# load remoting bottom library
remoting_bottom_lib_path = os.getenv('REMOTING_BOTTOM_LIBRARY')
if remoting_bottom_lib_path is not None:
    cpp_lib = ctypes.CDLL(remoting_bottom_lib_path)
    start_trace = cpp_lib.startTrace
    end_trace = cpp_lib.endTrace

log_breakpoint_lib_path = os.getenv('LOG_BREAKPOINT_LIBRARY')
if log_breakpoint_lib_path is not None:
    cpp_lib = ctypes.CDLL(log_breakpoint_lib_path)
    breakpoint = cpp_lib.log_breakpoint

if(len(sys.argv) != 3):
    print('Usage: python3 inference.py num_iter batch_size')
    sys.exit()

num_iter = int(sys.argv[1])
batch_size = int(sys.argv[2])

print("NOTICE: you should run traning script first to get the model weight file in checkpoint folder")
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 10
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='dataset')
model = ResNet18() # 得到预训练模型
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层修改
# 载入权重
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model = model.to(device)
model.eval()  # 验证模型

data_sample, target_sample = next(iter(test_loader))

# remove initial overhead
torch.cuda.empty_cache()
for i in range(2):
    data = data_sample.to(device)
    target = target_sample.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    # convert output probabilities to predicted class
    _, pred = torch.max(model(data), 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.data.cpu().numpy())

if log_breakpoint_lib_path is not None:
    breakpoint()

if remoting_bottom_lib_path is not None:
    start_trace()

print("begin trace")
sys.stdout.flush()

T1 = time.time()

for i in range(num_iter):
    data = data_sample.to(device)
    target = target_sample.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    # convert output probabilities to predicted class(将输出概率转换为预测类)
    _, pred = torch.max(model(data), 1) 
    # compare predictions to true label(将预测与真实标签进行比较)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.data.cpu().numpy())
    
T2 = time.time()
print('time used: ', T2-T1)

if remoting_bottom_lib_path is not None:
    end_trace()

# print(correct)
