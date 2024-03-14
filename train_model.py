import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import _LRScheduler


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # input size = 28*28
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        
        # input channel = 1, filter = 10 ,kernel = 5, zero padding = 0, stribe = 1
        # ((W-K+2P)/S)+1 = ((input size-kernel+2padding)/stribe) = ((28-5+0)/1)+1 = 24 -> 24*24
        # maxpooling = 12*12
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        # ((12-5+0)/1)+1 = 8 -> 8*8
        self.drop2D = nn.Dropout2d(p = 0.25, inplace=False)
        self.mp = nn.MaxPool2d(2)  # 4*4
        self.fc1 = nn.Linear(320,10)  # 4*4*20  = 320
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp(x)
        x = F.relu(self.conv2(x))
        x = self.mp(x)
        x = self.drop2D(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class Train:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer

    def training_step(self, model, batch):
        data, target = batch
        output = model(data)
        loss = self.criterion(output, target)
        return loss

    def train(self, optimizer, training_step, max_steps: int, max_epochs: int):
        assert max_epochs is not None or max_steps is not None
        max_epochs = max_epochs if max_epochs else max_steps // len(self.train_loader) + (0 if max_steps % len(self.train_loader) == 0 else 1)
        count_steps = 0

        self.model.train()
        for epoch in range(max_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                loss = training_step(self.model, (data, target))
                loss.backward()
                optimizer.step()
                count_steps += 1
                if count_steps >= max_steps:
                    acc = self.evaluate(self.model)
                    print(f'[Training Epoch {epoch} / Step {count_steps}] Final Acc: {acc}%')
                    return
            acc = self.evaluate(self.model)
            print(f'[Training Epoch {epoch} / Step {count_steps}] Final Acc: {acc}%')

    def evaluate(self, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)')
        return accuracy


train_dataset = datasets.MNIST(root="../dataset/MNIST",
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

test_dataset = datasets.MNIST(root="../dataset/MNIST",
                              train=False,
                              transform=transforms.ToTensor())

batch_size = 64

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
                                  
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


if __name__:
    trainer = Train(model, train_loader, test_loader, criterion, optimizer)

    trainer.train(optimizer, trainer.training_step,  max_steps=1000 * 10, max_epochs=5)
    torch.save(model, './models/originai_model.pth')
    trainer.evaluate(model)
