from model import Wolf_Husky_Classifier, test
from dataset import Wolf_Husky_Loader
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    model = Wolf_Husky_Classifier()
    model.load_state_dict(torch.load('wolf_husky_classifier.pth'))
    dataset = Wolf_Husky_Loader('data/train/')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    test(model, dataloader)