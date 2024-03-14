import torch
from torchsummary import summary

model0 = torch.load('./models/original_model.pth')
model1 = torch.load('./models/level_pruned_model.pth')
model2 = torch.load('./models/speedup_L1_model.pth')

print("Model Summary:")
summary(model0, (1,28,28))

print('\nLevel pruned Model Summary')
summary(model1,(1,28,28))

print('\nspeedup Model Summary:')
summary(model2, (1,28,28))