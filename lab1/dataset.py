from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os   
from PIL import Image
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt



class Wolf_Husky_Loader(Dataset):
    def __init__(self, data_dir):
        path = os.listdir(data_dir)
        self.classes = [p for p in path if os.path.isdir(os.path.join(data_dir, p))]
        self.classes = natsorted(self.classes)        
        self.paths = []
        self.labels = []

        for i, cls in enumerate(self.classes):
            for file_path in natsorted(os.listdir(data_dir+cls)):
                file_path = os.path.join(data_dir, cls, file_path)
                self.paths.append(file_path)
                self.labels.append(i)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


def show_images(images, labels, class_names):
    plt.figure(figsize=(12, 8))
    for i in range(min(8, len(images))):  # Show up to 8 images in a batch
        ax = plt.subplot(2, 4, i + 1)
        image = images[i].permute(1, 2, 0).numpy()  # Reorder dimensions for display
        image = (image * 255).astype(np.uint8)  # Convert to uint8 for display
        plt.imshow(image)
        plt.title(class_names[labels[i].item()])
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    dataset = Wolf_Husky_Loader('data/train/')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, labels in dataloader:
        show_images(images, labels, dataset.classes)
