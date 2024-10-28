from dataset import Wolf_Husky_Loader
from linear_model import Wolf_Husky_LinearClassifier,linear_train
from torch.utils.data import DataLoader,random_split
import torch
import torch.nn as nn
import torch.optim as optim

TRAIN_RATIO = 0.8
BATCH_SIZE = 16
LEARNING_RATE = 0.001

if __name__ == "__main__":
    dataset = Wolf_Husky_Loader('data/train/')
    batch_size = BATCH_SIZE
    lr = LEARNING_RATE
    train_ratio = TRAIN_RATIO
    val_ratio = 1 - train_ratio

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    model = Wolf_Husky_LinearClassifier()
    linear_train(model, train_loader, val_loader, num_epochs=20, lr=lr,save=True)
