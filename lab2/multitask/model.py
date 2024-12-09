import torch
import torch.nn as nn

def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    formatted_number = f"{pytorch_total_params:,}"
    print(f"Total number of parameters in the model: {formatted_number}")
    
    num_classes = 2 
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    
    return model

class simple_classifier(torch.nn.Module):
    def __init__(self):
        super(simple_classifier, self).__init__()
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1,stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 64, kernel_size=3, padding=1,stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64*15*19, 7)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits


class deeplabv3_Multitask(torch.nn.Module):
    def __init__(self):
        super(deeplabv3_Multitask, self).__init__()
        self.deeplab = load_model()
        self.clsfr = simple_classifier()
        
    def forward(self, x):
        features = self.deeplab.backbone(x)
        
        out_features = features['out']
        aux_features = features['aux']        
        classification = self.clsfr(out_features)
        
        out = self.deeplab.classifier(out_features)      
        out = torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)  
        aux = self.deeplab.aux_classifier(aux_features)
        aux = torch.nn.functional.interpolate(aux, size=x.shape[2:], mode='bilinear', align_corners=False)
        

        return {'out': out, 'aux': aux, 'classification': classification}


if __name__ == "__main__":
    model = deeplabv3_Multitask()
    # model = load_model()
    model.eval()
    img = torch.randn(10, 3, 450, 600)
    output = model(img)['out']

    print(output.shape)