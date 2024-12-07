import os
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import HAM10000, preload_ham10000
from evaluation import *


def save_checkpoint(model, optimizer, epoch, best_val_acc, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }, path)

def get_dataset_loaders(dataset_root,batch_size=4,val_size=0.2,max_samples=None):
    print("Loading dataset...")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(os.path.join(dataset_root, 'GroundTruth.csv'))

    df['label'] = df.iloc[:, 1:].idxmax(axis=1).map({
        'MEL': 0,
        'NV': 1,
        'BCC': 2,
        'AKIEC': 3,
        'BKL': 4,
        'DF': 5,
        'VASC': 6
    })
    df = df.drop(columns=['MEL','NV','BCC','AKIEC','BKL','DF','VASC'])

    train_data, val_data = preload_ham10000(dataset_root,df, val_size=val_size,max_samples=max_samples)

    train_dataset = HAM10000(data=train_data, transform=preprocess)
    val_dataset = HAM10000(data=val_data, transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_loader)}")
    print(f"Test size: {len(val_loader)}")

    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, class_criterion,optimizer, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader)):
        inputs, masks,label = data
        inputs, masks,label = inputs.to(device), masks.to(device), label.to(device)
        optimizer.zero_grad()
        outputs_seg = model(inputs)['out']
        loss_seg = criterion(outputs_seg, masks)
        outputs_cls = model(inputs)['classification']
        loss_cls = class_criterion(outputs_cls, label)
        loss = loss_seg + loss_cls
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def evaluate_one_epoch(model,val_loader,criterion,class_criterion,device):
    model.eval()
    correct_seg = 0
    correct_cls = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for data in tqdm(val_loader):
            inputs, masks , label= data
            inputs, masks , label= inputs.to(device), masks.to(device), label.to(device)
            outputs_seg = model(inputs)['out']

            # Resize outputs to match mask dimensions
            outputs_seg = F.interpolate(outputs_seg, size=masks.shape[1:], mode="bilinear", align_corners=False)
            loss_seg = criterion(outputs_seg, masks)

            outputs_cls = model(inputs)['classification']
            loss_cls = class_criterion(outputs_cls, label)
            
            loss = loss_seg + loss_cls
            val_loss += loss.item()

            preds = torch.argmax(outputs_seg, dim=1)
            correct_seg += (preds == masks).sum().item()
            total += masks.numel()

            preds_cls = torch.argmax(outputs_cls, dim=1)
            correct_cls += (preds_cls == label).sum().item()
        
    return val_loss, correct_seg, correct_cls, total