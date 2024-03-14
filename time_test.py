import torch
import time

model1 = torch.load('./models/model.pth')
start1 = time.time()
model1(torch.rand(64, 1, 28, 28))
print('Original Model - Elapsed Time:', time.time() - start1)

model2 = torch.load('./models/level_pruned_model.pth')
start2 = time.time()
model2(torch.rand(64, 1, 28, 28))
print('Level pruner Model - Elapsed Time:', time.time() - start2)

model3 = torch.load('./models/L1_model.pth')
start3 = time.time()
model3(torch.rand(64, 1, 28, 28))
print('L1  pruner Model - Elapsed Time:', time.time() - start3)

model4 = torch.load('./models/speedup_L1_model.pth')
start4 = time.time()
model4(torch.rand(64, 1, 28, 28))
print('Speedup pruner Model - Elapsed Time:', time.time() - start4)

