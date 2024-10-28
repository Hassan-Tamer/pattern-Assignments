import torch
import torch.nn as nn
import torch.optim as optim

class Wolf_Husky_LinearClassifier(nn.Module):
    def __init__(self):
        super(Wolf_Husky_LinearClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # Flatten input from 256x256x3 to 196608 (for images resized to 256x256 with 3 color channels
            nn.Linear(256*256*3, 1),  # Linear layer with a single output
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.model(x)

def linear_train(model, train_loader, val_loader, num_epochs=10, lr=0.001, save=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            predicted = (output > 0.5).float()
            correct_train += (predicted == labels.unsqueeze(1)).sum().item()
            total_train += labels.size(0)
        
        train_loss /= len(train_loader)
        train_accuracy = correct_train / total_train * 100
        
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels.float().unsqueeze(1))
                val_loss += loss.item()
                
                predicted = (output > 0.5).float()
                correct_val += (predicted == labels.unsqueeze(1)).sum().item()
                total_val += labels.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val * 100
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    if save:
        torch.save(model.state_dict(), "wolf_husky_linear_classifier.pth")
    return model

def linear_test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            predicted = (output > 0.5).float()
            correct += (predicted == labels.float().unsqueeze(1)).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
