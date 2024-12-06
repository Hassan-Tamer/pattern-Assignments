from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from dataset import HAM10000, preload_ham10000
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import load_model
import torch.nn.functional as F
import wandb
from natsort import natsorted
from evaluation import evaluate_model


if __name__ == '__main__':

    wandb.init(project="ham10000-classification", config={
    "learning_rate": 1e-4,
    "epochs": 10,
    "batch_size": 8,
    "optimizer": "Adam"
})

    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    dataset_root = 'data/'
    train_data, val_data = preload_ham10000(dataset_root, val_size=0)

    train_dataset = HAM10000(data=train_data, transform=preprocess)
    val_dataset = HAM10000(data=val_data, transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)

    print(f"Train size: {len(train_loader)}")
    print(f"Test size: {len(val_loader)}")


    model = load_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    model.to(device)
    criterion.to(device)
    num_epochs = wandb.config.epochs
    best_val_acc = 0.0
    patience = 3
    epochs_without_improvement = 0
    checkpoint_path = "best_model.pth"


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader)):
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader)}")
        wandb.log({"train_loss": train_loss})
    

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader):
                inputs, masks = data
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)['out']

                # Resize outputs to match mask dimensions
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode="bilinear", align_corners=False)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == masks).sum().item()
                total += masks.numel()
        
        val_loss /= len(val_loader)
        val_acc = correct / total * 100
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        wandb.log({"val_loss": val_loss, "val_accuracy": val_acc})


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print("Training completed.")
    wandb.finish()

    evaluate_model(val_loader, model, device)



