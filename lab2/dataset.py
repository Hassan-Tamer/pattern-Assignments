import os 
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd

class HAM10000(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask , label = self.data[idx]

        to_tensor = transforms.ToTensor()
        if self.transform:
            image = self.transform(image)
            mask = to_tensor(mask) 
        else:
            image = to_tensor(image)
            mask = to_tensor(mask)

        mask = mask.squeeze(0)  # Remove channel dimension if present
        mask = mask.long()      # Convert to integer for class indices

        return image, mask , label

def get_labels_df(img_name,labels_df):
    img_name = img_name.split('.')[0]
    label = labels_df.loc[labels_df['image'] == img_name, 'label'].values[0]
    return label

def preload_ham10000(root,labels_df,val_size=0.2, seed=42 , max_samples=None):
    imgs_path = os.path.join(root, 'images')
    masks_path = os.path.join(root, 'masks')

    image_files = []
    mask_files = []

    for i,f in enumerate(os.listdir(imgs_path)):
        if f.endswith(('.jpg', '.png')):
            image_files.append(f)
        
        if max_samples is not None and i >= max_samples:
            break
    
    for i,f in enumerate(os.listdir(masks_path)):
        if f.endswith(('.jpg', '.png')):
            mask_files.append(f)

        if max_samples is not None and i >= max_samples:
            break

    image_files = natsorted(image_files)
    mask_files = natsorted(mask_files)

    assert len(image_files) == len(mask_files), "Mismatch between image and mask counts."

    all_data = []
    i=0
    for img_file, mask_file in zip(image_files, mask_files):
        if max_samples is not None and i >= max_samples:
            break
        img = Image.open(os.path.join(imgs_path, img_file)).convert("RGB")
        mask = Image.open(os.path.join(masks_path, mask_file)).convert("L")
        label = get_labels_df(img_file,labels_df)
        all_data.append((img, mask, label))

    if val_size == 0:
        return all_data, None

    train_data, val_data = train_test_split(all_data, test_size=val_size, random_state=seed)
    return train_data, val_data

    

if __name__ == '__main__':    
    dataset_root = "data/"
    
    labels_df = pd.read_csv('data/labels.csv')
    train_data, val_data = preload_ham10000(dataset_root, labels_df, val_size=0)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = HAM10000(data=train_data, transform=preprocess)
    val_dataset = HAM10000(data=val_data, transform=preprocess)

    print(f"Train dataset size: {len(train_dataset)}")

    for img, mask,label in train_dataset:
        print(img.shape, mask.shape)
        Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8')).show()
        Image.fromarray((mask.numpy() * 255).astype('uint8')).show()
        print(label)
        break