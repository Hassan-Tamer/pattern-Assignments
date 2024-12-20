import os 
from matplotlib import pyplot as plt
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class HAM10000(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load a single image and its corresponding mask.
        Args:
            idx (int): Index of the data to fetch.
        Returns:
            tuple: (image, mask) - both as tensors.
        """
        image, mask = self.data[idx]

        to_tensor = transforms.ToTensor()
        if self.transform:
            image = self.transform(image)
            mask = to_tensor(mask)  # Ensure consistent preprocessing of the mask
        else:
            image = to_tensor(image)
            mask = to_tensor(mask)

        # Ensure mask is in the proper format
        mask = mask.squeeze(0)  # Remove channel dimension if present
        mask = mask.long()      # Convert to integer for class indices

        return image, mask


def preload_ham10000(root, val_size=0.2, seed=42):
    imgs_path = os.path.join(root, 'images')
    masks_path = os.path.join(root, 'masks')

    # Get sorted file lists to ensure consistent ordering
    image_files = natsorted([f for f in os.listdir(imgs_path) if f.endswith(('.jpg', '.png'))])
    mask_files = natsorted([f for f in os.listdir(masks_path) if f.endswith(('.jpg', '.png'))])

    assert len(image_files) == len(mask_files), "Mismatch between image and mask counts."

    # Preload all data into memory
    all_data = [
        (
            Image.open(os.path.join(imgs_path, img_file)).convert("RGB"),
            Image.open(os.path.join(masks_path, mask_file)).convert("L"),
        )
        for img_file, mask_file in zip(image_files, mask_files)
    ]

    if val_size == 0:
        return all_data, None

    train_data, val_data = train_test_split(all_data, test_size=val_size, random_state=seed)
    return train_data, val_data

    

if __name__ == '__main__':    
    dataset_root = "data/"
    train_data, val_data = preload_ham10000(dataset_root, val_size=0)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    train_dataset = HAM10000(data=train_data, transform=preprocess)
    val_dataset = HAM10000(data=val_data, transform=preprocess)

    # Example usage
    print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Validation dataset size: {len(val_dataset)}")

    for img, mask in train_dataset:
        print(img.shape, mask.shape)
        Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8')).show()
        Image.fromarray((mask.numpy() * 255).astype('uint8')).show()
