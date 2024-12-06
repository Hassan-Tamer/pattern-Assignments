import torch
from matplotlib import pyplot as plt
import numpy as np

def dice_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)  # Adding epsilon to avoid division by 0

    return dice.item()


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

    with torch.no_grad():
        for i,(inputs, masks) in enumerate(val_loader):
            if i > 20:
                break
                
            inputs, masks = inputs.to(device), masks.to(device)

            # Forward pass
            outputs = model(inputs)['out']

            # Resize the output to match the mask size (if needed)
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

            # Calculate the Dice score
            pred_mask = torch.argmax(outputs, dim=1)  # Get the predicted mask
            dice = dice_score(pred_mask, masks)
            dice_scores.append(dice)

            # Display segmentation examples for the first batch
            display_segmentation_examples(inputs, masks, outputs, num_examples=1)

    # Calculate the average Dice score
    avg_dice_score = np.mean(dice_scores)
    print(f"Average Dice Score: {avg_dice_score:.4f}")    