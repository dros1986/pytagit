import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from torchvision import transforms, models
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from PIL import Image
from tqdm import tqdm


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class CNNTrainer(pl.LightningModule):
    def __init__(self, model_name, learning_rate, batch_size, pretrained=True, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        if model_name == "ResNet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == "ResNet34":
            self.model = models.resnet34(pretrained=pretrained)
        elif model_name == "ResNet50":
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Invalid model name")
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)
    


    


def train_cnn(image_paths, labels, model_name, learning_rate, batch_size, num_classes, pretrained=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageDataset(image_paths, labels, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = CNNTrainer(model_name, learning_rate, batch_size, pretrained, num_classes)
    
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[RichProgressBar(), ModelCheckpoint(monitor='val_loss')],
        accelerator="auto"
    )
    
    trainer.fit(model, dataloader)

    # return the model
    return model
