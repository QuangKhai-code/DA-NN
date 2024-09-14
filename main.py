import wandb
import random
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader import GetLoader
from models.model import CNNModel
import numpy as np
from test import test
from torchvision import transforms
from torch.utils.data import random_split
from sklearn.metrics import recall_score
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score

# Dataset settings
source_dataset_name = 'mendeley_lbc'
target_dataset_name = 'data_private'
source_image_root = os.path.join('.', 'dataset', source_dataset_name)
target_image_root = os.path.join('.', 'dataset', target_dataset_name)
#create a model root if it does not exist
if not os.path.exists(os.path.join('.', 'cache')):
    os.makedirs(os.path.join('.', 'cache'))
model_root = os.path.join('.', 'cache')

wandb.login()
# Initialize wandb
wandb.init(project="domain-adaptation", name="mendeley-to-private")

# Training settings
cuda = True
cudnn.benchmark = True
lr = 1e-4
batch_size = 32
image_size = 224
n_epoch = 100

# Log hyperparameters
wandb.config.update({
    "learning_rate": lr,
    "batch_size": batch_size,
    "image_size": image_size,
    "epochs": n_epoch,
    "Model": "VGG19"
})

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
    
# Image transformations for RGB images with augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(image_size),
    transforms.RandAugment(2, 9),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Define the datasets
source_dataset_name_train = source_image_root + "_train"
source_dataset_name_val = source_image_root + "_val"
source_dataset_name_test = source_image_root + "_test"

target_dataset_name_train = target_image_root + "_train"
target_dataset_name_val = target_image_root + "_val"
target_dataset_name_test = target_image_root + "_test"

# Get data loaders for source and target datasets
#source_train_loader, source_val_loader, source_test_loader = get_data_loaders(dataset_source, batch_size, train_transform, test_transform)
#target_train_loader, target_val_loader, target_test_loader = get_data_loaders(dataset_target, batch_size, train_transform, test_transform)

# Assuming GetLoader returns a Dataset object
source_train_dataset = GetLoader(data_root=source_dataset_name_train, transform=train_transform)
source_val_dataset = GetLoader(data_root=source_dataset_name_val, transform=test_transform)
source_test_dataset = GetLoader(data_root=source_dataset_name_test, transform=test_transform)

target_train_dataset = GetLoader(data_root=target_dataset_name_train, transform=train_transform)
target_val_dataset = GetLoader(data_root=target_dataset_name_val, transform=test_transform)
target_test_dataset = GetLoader(data_root=target_dataset_name_test, transform=test_transform)

# Create DataLoader objects
source_train_loader = torch.utils.data.DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
source_val_loader = torch.utils.data.DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
source_test_loader = torch.utils.data.DataLoader(source_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

target_train_loader = torch.utils.data.DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
target_val_loader = torch.utils.data.DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
target_test_loader = torch.utils.data.DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

my_net = CNNModel(num_classes=3)

# Setup optimizer
optimizer = optim.AdamW(my_net.parameters(), lr=lr, weight_decay=0.0005)
loss_class = LabelSmoothingCrossEntropy()
loss_domain = nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# Training
best_f1_source = 0.0
best_f1_target = 0.0
best_f1_score = 0.0
best_acc_source = 0.0
best_acc_target = 0.0
best_acc_score = 0.0
for epoch in range(n_epoch):
    my_net.train()
    len_dataloader = min(len(source_train_loader), len(target_train_loader))
    data_source_iter = iter(source_train_loader)
    data_target_iter = iter(target_train_loader)

    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Training model using source data
        s_img, s_label = next(data_source_iter)

        my_net.zero_grad()
        batch_size = len(s_label)

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()

        class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, torch.zeros(batch_size).long().cuda() if cuda else torch.zeros(batch_size).long())

        # Training model using target data
        t_img, _ = next(data_target_iter)

        if cuda:
            t_img = t_img.cuda()

        _, domain_output = my_net(input_data=t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, torch.ones(batch_size).long().cuda() if cuda else torch.ones(batch_size).long())
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        print(f'epoch: {epoch}, [iter: {i + 1} / all {len_dataloader}], err_s_label: {err_s_label.cpu().item():.4f}, err_s_domain: {err_s_domain.cpu().item():.4f}, err_t_domain: {err_t_domain.cpu().item():.4f}')

    # Save model
    torch.save(my_net.state_dict(), os.path.join(model_root, f'model_epoch_{epoch}.pth'))
    
    # Validation
    val_acc_source, val_f1_source = test(source_dataset_name, epoch, source_val_loader)
    val_acc_target, val_f1_target = test(target_dataset_name, epoch, target_val_loader)
    val_avg_f1 = (val_f1_source + val_f1_target) / 2
    val_avg_acc = (val_acc_source + val_acc_target) / 2
    print(f'Validation accuracy on {source_dataset_name}: {val_acc_source:.4f}, F1-score: {val_f1_source:.4f}')
    print(f'Validation accuracy on {target_dataset_name}: {val_acc_target:.4f}, F1-score: {val_f1_target:.4f}')
    print(f"val Avg F1: {val_avg_f1:.4f}")
    print(f"val Avg acc: {val_avg_acc:.4f}")
    
    # Log validation metrics
    wandb.log({
        "val_acc_source": val_acc_source,
        "val_f1_source": val_f1_source,
        "val_acc_target": val_acc_target,
        "val_f1_target": val_f1_target,
        "val_avg_f1": val_avg_f1,
        "val_avg_acc": val_avg_acc
    })

    # Update best metrics
    if val_avg_f1 > best_f1_score:
        best_f1_score = val_avg_f1
        wandb.run.summary["best_f1_score"] = best_f1_score
    if val_avg_acc > best_acc_score:
        best_acc_score = val_avg_acc
        wandb.run.summary["best_acc_score"] = best_acc_score
    if val_f1_source > best_f1_source:
        best_f1_source = val_f1_source
        wandb.run.summary["best_f1_source"] = best_f1_source
    if val_f1_target > best_f1_target:
        best_f1_target = val_f1_target
        wandb.run.summary["best_f1_target"] = best_f1_target
    if val_acc_source > best_acc_source:
        best_acc_source = val_acc_source
        wandb.run.summary["best_acc_source"] = best_acc_source
    if val_acc_target > best_acc_target:
        best_acc_target = val_acc_target
        wandb.run.summary["best_acc_target"] = best_acc_target
        # Save the best model
        torch.save(my_net.state_dict(), os.path.join(model_root, 'best_model.pth'))
        

print("Loading best model for final evaluation...")

# Test and calculate metrics
def calculate_metrics(loader, dataset_name):
    test_net = CNNModel(num_classes=3)
    test_net.load_state_dict(torch.load(os.path.join(model_root, 'best_model.pth')))
    test_net.eval()
    if cuda:
        test_net = test_net.cuda()
        
    n_total = 0
    n_correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            if cuda:
                images = images.cuda()
                labels = labels.cuda()

            class_output, _ = test_net(input_data=images, alpha=0)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
            n_total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())

    acc = n_correct.double() / n_total
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Test results on {dataset_name}:')
    print(f'Accuracy: {acc:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # Log test metrics
    wandb.log({
        f"final_test_acc_{dataset_name}": acc,
        f"final_test_recall_{dataset_name}": recall,
        f"final_test_f1_{dataset_name}": f1
    })


# Calculate metrics for source and target datasets
calculate_metrics(source_test_loader, source_dataset_name)
calculate_metrics(target_test_loader, target_dataset_name)

print('Training completed')

# Finish the wandb run
wandb.finish()