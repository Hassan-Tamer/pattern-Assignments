from dataset import Wolf_Husky_Loader
from torch.utils.data import DataLoader, random_split
import torch
from linear_model import Wolf_Husky_LinearClassifier, linear_train
import itertools

TRAIN_RATIO = 0.8
NUM_EPOCHS = 10
BATCH_SIZES = [8, 16, 32]  # Different batch sizes to try
LEARNING_RATES = [0.01,0.001, 0.0005, 0.0001]  # Different learning rates to try

def run_tuning():
    dataset = Wolf_Husky_Loader('data/train/')
    train_ratio = TRAIN_RATIO
    val_ratio = 1 - train_ratio

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # keeping track best parameters
    best_accuracy = 0.0
    best_params = {}


    for batch_size, lr in itertools.product(BATCH_SIZES, LEARNING_RATES):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = Wolf_Husky_LinearClassifier()

        
        print(f"\nTraining with batch size: {batch_size}, learning rate: {lr}")
        trained_model = linear_train(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=lr)

        accuracy = evaluate(trained_model, val_loader)
        print(f"Validation Accuracy: {accuracy:.2f}% with batch size: {batch_size}, learning rate: {lr}")

        # Track the best configuration
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'batch_size': batch_size, 'learning_rate': lr}

    print(f"\nBest Hyperparameters - Batch Size: {best_params['batch_size']}, Learning Rate: {best_params['learning_rate']}")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

def evaluate(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            predicted = (output > 0.5).float()
            correct += (predicted == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
    
    return correct / total * 100 

if __name__ == "__main__":
    run_tuning()
