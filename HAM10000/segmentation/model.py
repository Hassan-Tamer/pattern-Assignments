import torch

def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    formatted_number = f"{pytorch_total_params:,}"
    print(f"Total number of parameters in the model: {formatted_number}")
    
    num_classes = 2 
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    
    return model