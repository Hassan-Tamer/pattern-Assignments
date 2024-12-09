import torch
import torch.nn as nn
import torch.optim as optim
from model import deeplabv3_Multitask
import wandb
from evaluation import *
from train import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--useWandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--datasetroot', type=str, default='data/', help='Root directory of the dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.useWandb:
        wandb.init(project="ham10000-classification", config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam"
        })

    dataset_root = args.datasetroot
    train_loader , val_loader = get_dataset_loaders(dataset_root, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = deeplabv3_Multitask()
    criterion = nn.CrossEntropyLoss() 
    class_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.to(device)
    criterion.to(device)
    class_criterion.to(device)

    num_epochs = args.epochs
    best_val_acc = 0.0
    patience = args.patience
    epochs_without_improvement = 0
    checkpoint_path = "best_model.pth"


    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, class_criterion, optimizer, device)
        print(f"Epoch {epoch+1}, loss: {avg_loss}")
        if args.useWandb:
            wandb.log({"train_loss": avg_loss})

        val_loss, correct_seg, correct_cls, total = evaluate_one_epoch(model, val_loader, criterion, class_criterion, device)
        val_loss /= len(val_loader)
        val_acc_seg = correct_seg / total * 100
        val_acc_cls = correct_cls / len(val_loader.dataset) * 100
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc_seg:.2f}% (seg), {val_acc_cls:.2f}% (cls)")
        if args.useWandb:
            wandb.log({"val_loss": val_loss, "val_accuracy": val_acc_seg, "val_accuracy_cls": val_acc_cls})

        if val_acc_seg > best_val_acc:
            best_val_acc = val_acc_seg
            save_checkpoint(model, optimizer, epoch + 1, best_val_acc, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print("Training completed.")
    if args.useWandb:
        wandb.finish()

    evaluate_model(val_loader, model, device)
    evaluate_classifier(val_loader, model, device, class_criterion)