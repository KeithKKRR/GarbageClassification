import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import GarbageDataset

'''
- constants
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
- hyper-parameters
'''
dataset_path = "garbage_classification"
train_path = os.path.join(dataset_path, "train.txt")
val_path = os.path.join(dataset_path, "val.txt")
test_path = os.path.join(dataset_path, "test.txt")
batch_size = 16
max_epoch = 100
learning_rate = 0.01
learning_rate_adjust_step = 1
num_classes = len(GarbageDataset.class_dict.keys())
model_name = "resnet50"
train_log_step = 100

'''
- datasets
'''
train_dataset = GarbageDataset(train_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = GarbageDataset(val_path)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
test_dataset = GarbageDataset(test_path)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

'''
- models
'''
new_model = models.resnet50(pretrained=True)
for param in new_model.parameters():
    param.requires_grad = False

num_feature = new_model.fc.in_features
new_model.fc = nn.Sequential(
    nn.Linear(num_feature, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes),
    # nn.Linear(num_feature, num_classes),
    nn.Softmax(dim=1)
)
for param in new_model.fc.parameters():
    param.requires_grad = True

new_model.to(device)

'''
- optimizer
'''
# optim = torch.optim.SGD(new_model.fc.parameters(), lr=1e-2, momentum=0.9)   # only finetune params on FC
ignored_params = list(map(id, new_model.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,new_model.parameters())
optim = torch.optim.SGD([
            {'params': base_params},
            {'params': new_model.fc.parameters(), 'lr': 1e-2}
            ], lr=1e-3, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=learning_rate_adjust_step, gamma=0.1)

'''
- loss
'''
loss_fn = nn.CrossEntropyLoss()

'''
- train & val
'''
for epoch in range(max_epoch):
    total_loss, correct, total = 0, 0, 0
    new_model.train()
    for index, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)
        output = new_model(img)                                         # go through network
        optim.zero_grad()
        loss = loss_fn(output, label)                                   # calculate loss function
        loss.backward()
        optim.step()
        _, predicted = torch.max(output, dim=1)                         # predict label according to softmax output
        total += label.size(0)                                        # count total number of data
        correct += (predicted == label).squeeze().cpu().sum().numpy()   # count total number of correct prediction
        total_loss += loss.item()                                       # sum up loss

        if (index + 1) % train_log_step == 0:
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch+1, max_epoch, index + 1, len(train_loader), total_loss / label.shape[0], correct / total))
            total_loss = 0

    scheduler.step()    # adjust learning rate

    # validation
    total_loss_val, correct_val, total_val = 0, 0, 0
    new_model.eval()
    with torch.no_grad():
        for index, (img, label) in enumerate(val_loader):
            img, label = img.to(device), label.to(device)
            output = new_model(img)
            loss = loss_fn(output, label)
            _, predicted = torch.max(output, dim=1)
            total_val += label.shape[0]
            correct_val += (predicted == label).squeeze().cpu().sum().numpy()
            total_loss_val += loss.item()
    print("Validation:Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
        epoch+1, max_epoch, total_loss_val, correct_val / total_val))
    total_loss, correct, total = 0, 0, 0
    total_loss_val, correct_val, total_val = 0, 0, 0
    new_model.train()





