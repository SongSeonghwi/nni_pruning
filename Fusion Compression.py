import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn

from examples.compression.models import (
    prepare_optimizer,
)
from train_model import train_loader, criterion, Train, test_loader,optimizer
from nni.compression import TorchEvaluator
from nni.compression.base.compressor import Quantizer
from nni.compression.distillation import DynamicLayerwiseDistiller
from nni.compression.pruning import TaylorPruner, AGPPruner
from nni.compression.quantization import QATQuantizer
from nni.compression.utils import auto_set_denpendency_group_ids
from nni.compression.speedup import ModelSpeedup

# 加载原始模型
teacher_model = torch.load('./models/original_model.pth')

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(10 * 7 * 7, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 10 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

student_model = StudentNet()

# 构建混合剪枝所需的配置
bn_list = [module_name for module_name, module in student_model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
p_config_list = [{
    'op_types': ['Conv2d'],
    'sparse_ratio': 0.5
}, *[{
    'op_names': [name],
    'target_names': ['_output_'],
    'target_settings': {
        '_output_': {
            'align': {
                'module_name': name.replace('bn', 'conv').replace('downsample.1', 'downsample.0'),
                'target_name': 'weight',
                'dims': [0],
            },
            'granularity': 'per_channel'
        }
    }
} if 'bn' in name or 'downsample.1' in name else {
    'op_names': [name],
    'target_names': ['_output_'],
    'target_settings': {
        '_output_': {
            'granularity': 'per_channel'
        }
    }
} for name in bn_list]]
dummy_input = torch.rand(64, 1, 28, 28) 
p_config_list = auto_set_denpendency_group_ids(student_model, p_config_list, dummy_input)

# student optimizer
optimizer = prepare_optimizer(student_model)
trainer = Train(student_model, train_loader, test_loader, criterion, optimizer)


# student evaluator
evaluator = TorchEvaluator(Train.train, optimizer,Train.training_step)
# 
sub_pruner = TaylorPruner(student_model, p_config_list, evaluator, training_steps=100)

# 创建定时剪枝器
scheduled_pruner = AGPPruner(sub_pruner, interval_steps=100, total_times=30)

# 构建量化配置
q_config_list = [{
    'op_types': ['Conv2d'],
    'quant_dtype': 'int8',
    'target_names': ['_input_'],
    'granularity': 'per_channel'
}, {
    'op_types': ['Conv2d'],
    'quant_dtype': 'int8',
    'target_names': ['weight'],
    'granularity': 'out_channel'
}]

# 创建量化器
quantizer = QATQuantizer.from_compressor(scheduled_pruner, q_config_list, quant_start_step=100)

# 创建蒸馏器
def teacher_predict(batch, teacher_model):
    return teacher_model(batch[0])

d_config_list = [{
    'op_types': ['Conv2d'],
    'lambda': 0.1,
    'apply_method': 'mse',
}]
distiller = DynamicLayerwiseDistiller.from_compressor(quantizer, d_config_list, teacher_model, teacher_predict, 0.1)

# 开始压缩
distiller.compress(max_steps=100 * 60, max_epochs=None)
distiller.unwrap_model()
distiller.unwrap_teacher_model()

# 加速模型
masks = scheduled_pruner.get_masks()
speedup = ModelSpeedup(student_model, dummy_input, masks)
model = speedup.speedup_model()

# 输出压缩后模型的参数数量和精度
print('Compressed model parameter number:', sum([param.numel() for param in model.parameters()]))

trainer = Train(student_model, train_loader,test_loader,criterion,optimizer)
trainer.evaluate()
