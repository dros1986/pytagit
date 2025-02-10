import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from functools import partial
from torchvision import transforms, models
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from PIL import Image
from tqdm.rich import tqdm
import numpy as np
import cv2
from einops import rearrange
import albumentations as A


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image=image)
        label = self.labels[idx] if self.labels is not None else 0
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
    





def transform_fun(train, sz, image):
    # if it is a pil image, transform to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    # TODO: set random transforms when train_transforms is True
    # if train:
    #     # if training, randomly apply some transforms
    #     trans = A.Compose([
    #         A.RandomRotate90(p=0.3),
    #         A.Flip(p=0.3),
    #         A.Transpose(p=0.3),
    #         A.GaussNoise(p=0.1),
    #         A.OneOf([
    #             A.MotionBlur(p=.2),
    #             A.MedianBlur(blur_limit=3, p=0.1),
    #             A.Blur(blur_limit=3, p=0.1),
    #         ], p=0.2),
    #         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    #         A.OneOf([
    #             A.OpticalDistortion(p=0.3),
    #             A.GridDistortion(p=.1),
    #         ], p=0.2),
    #         A.OneOf([
    #             A.CLAHE(clip_limit=2),
    #             A.RandomBrightnessContrast(),
    #         ], p=0.3),
    #         A.HueSaturationValue(p=0.1),
    #         A.LongestMaxSize(max_size=self.sz),
    #         A.PadIfNeeded(min_height=self.sz, min_width=self.sz, border_mode=cv2.BORDER_CONSTANT, value=.0, mask_value=.0),
    #         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    #   ])
    if train:
        # in this case, we want to apply only geometric transforms
        trans = A.Compose([
            A.RandomRotate90(p=0.3),
            # A.Flip(p=0.3),
            A.Transpose(p=0.3),
            A.GaussNoise(p=0.1),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
            ], p=0.2),
            A.LongestMaxSize(max_size=sz),
            A.PadIfNeeded(min_height=sz, min_width=sz, border_mode=cv2.BORDER_CONSTANT, value=.0, mask_value=.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ])
    else:
        # define transform
        trans = A.Compose([
                A.LongestMaxSize(max_size=sz),
                A.PadIfNeeded(min_height=sz, min_width=sz, border_mode=cv2.BORDER_CONSTANT, value=.0, mask_value=.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ])

    # apply it
    image_t = trans(image=image)['image']
    # change order
    # image_t = rearrange(image_t, 'h w c -> 1 c h w')
    image_t = rearrange(image_t, 'h w c -> c h w')
    # convert to tensor
    image_t = torch.tensor(image_t, dtype=torch.float32)
    # return
    return image_t



    


def train_cnn(image_paths, labels, model_name, epochs, learning_rate, batch_size, num_classes, pretrained=True):
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    transform = partial(transform_fun, train=True, sz=256)
    dataset = ImageDataset(image_paths, labels, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = CNNTrainer(model_name, learning_rate, batch_size, pretrained, num_classes)
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[RichProgressBar(), ModelCheckpoint(monitor='train_loss', mode='min')],
        accelerator="auto"
    )
    
    trainer.fit(model, dataloader)

    # return the model
    return model


def classify_with_cnn(image_paths, model, threshold, id_undefined_class, batch_size=32):
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    transform = partial(transform_fun, train=False, sz=256)
    
    model.eval()
    # predictions = {}
    predictions = []

    dataset = ImageDataset(image_paths, None, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(dataloader):
        images, labels = batch
        with torch.no_grad():
            output = model(images)
            output = F.softmax(output)
            prob, pred_label = output.max(dim=1)
            pred_label[prob < threshold] = id_undefined_class
            predictions.append(pred_label)

    predictions = torch.cat(predictions)

    
    return predictions

