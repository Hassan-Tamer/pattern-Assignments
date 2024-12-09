import torch
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def dice_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)  # Adding epsilon to avoid division by 0

    return dice.item()

def calculateIOU(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + 1e-6) / (union + 1e-6)  

def display_segmentation_examples(inputs, masks, outputs, num_examples=3):
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))
    for i in range(num_examples):
        ax = axes[i]
        
        # Original image
        ax[0].imshow(inputs[i].cpu().numpy().transpose(1, 2, 0))  # Convert to HWC format
        ax[0].set_title("Input Image")
        ax[0].axis('off')
        
        # Ground truth mask
        ax[1].imshow(masks[i].cpu().numpy(), cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[1].axis('off')
        
        # Predicted mask
        pred_mask = torch.argmax(outputs[i], dim=0)  # Take the class with the highest probability
        ax[2].imshow(pred_mask.cpu().numpy(), cmap='gray')
        ax[2].set_title("Prediction")
        ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()



def evaluate_model(val_loader, model, device):
    model.eval()
    dice_scores = []
    iou = []

    with torch.no_grad():
        for i,(inputs, masks,label) in enumerate(val_loader):
            if i > 20:
                break
                
            inputs, masks,label = inputs.to(device), masks.to(device), label.to(device)

            # Forward pass
            outputs = model(inputs)['out']

            # Resize the output to match the mask size (if needed)
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

            # Calculate the Dice score
            pred_mask = torch.argmax(outputs, dim=1)  # Get the predicted mask
            dice = dice_score(pred_mask, masks)
            iou_score = calculateIOU(pred_mask, masks)
            iou.append(iou_score)
            dice_scores.append(dice)

            display_segmentation_examples(inputs, masks, outputs, num_examples=3)

    # Calculate the average Dice score
    avg_dice_score = np.mean(dice_scores)
    iou_tensor = torch.tensor(iou)
    avg_iou = np.mean(iou_tensor.cpu().numpy())
    print(f"Average IOU: {avg_iou:.4f}")
    print(f"Average Dice Score: {avg_dice_score:.4f}")    



def evaluate_classifier(val_loader, model, device, class_criterion):
    model.eval()  # Set the model to evaluation mode
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, _, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass through the classifier
            outputs_cls = model(inputs)['classification']
            loss_cls = class_criterion(outputs_cls, labels)
            
            total_loss += loss_cls.item() * labels.size(0)  # Sum up batch loss
            total_samples += labels.size(0)

            _, predicted = torch.max(outputs_cls, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples  # Calculate average loss
    
    # Compute metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    accuracy = report["accuracy"] * 100
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1_score = report["weighted avg"]["f1-score"]
    
    print("Confusion Matrix:")
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    return avg_loss, accuracy, precision, recall, f1_score, conf_matrix
