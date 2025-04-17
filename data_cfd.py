import glob
import io
import json
import os
import random
import lmdb
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

import torch
import torchvision.transforms.functional as F
import albumentations as A

FILES_TRAIN = 'case_data1.txt'
FILES_TEST = 'case_data2.txt'


def tokenize_caption(tokenizer, caption, is_train=True):
    if isinstance(caption, (list, tuple)):
        caption = random.choice(caption) if is_train else caption[0]
    inputs = tokenizer(
        caption,
        max_length=tokenizer.model_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return inputs.input_ids[0]


def transform_train(image_dict, target_height=256, target_width=512):
    """
    Applies a random rotation (±15°) before resizing, random horizontal flip, and center cropping.
    If the input is a tuple or list of images, the same transformation is applied to each image.
    
    Args:
        images (np.ndarray or tuple/list of np.ndarray): Input single-channel image or tuple/list of images.
        target_height (int): Desired height after resizing.
        target_width (int): Desired width after cropping.
        
    Returns:
        np.ndarray or tuple: Transformed image or tuple of transformed images.
    """

    input_image = image_dict['input']
    # flow_img = np.expand_dims(flow_img, axis=-1)
    # Apply random rotation on the first image to determine the common transformation parameters.
    angle = np.random.uniform(-15, 15)
    center = (input_image.shape[1] // 2, input_image.shape[0] // 2)  # (x, y) center
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Generate a random horizontal flip flag (True with 50% probability).
    flip_flag = np.random.rand() < 0.5
    # Apply rotation to the first image to compute the new dimensions (should remain same as original for cv2.warpAffine)
    # Use cv2.BORDER_REFLECT to avoid black borders.
    def transform_single(image):
        # 1. Random Rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # 2. Resize the rotated image so that its height is target_height while preserving aspect ratio.
        original_height, original_width = rotated.shape
        scale = target_height / original_height
        new_width = int(original_width * scale)
        resized = cv2.resize(rotated, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        # 3. Random horizontal flip (50% chance) using the same decision for all images.
        # (We generate the flip flag once outside, so here we assume that variable is defined)
        if flip_flag:
            resized = np.fliplr(resized)
        # 4. Center crop to target_width.
        width_left = (resized.shape[1] - target_width) // 2
        cropped = resized[:, width_left:width_left + target_width]
        return cropped
    
    transformed_images = {}
    for key, img in image_dict.items():
        img = transform_single(img)
        img = 1.0 - (img.astype(np.float32) / 255.0)
        img = np.expand_dims(img, axis=0)
        tensor = torch.from_numpy(img)
        transformed_images[key] = tensor
        
    return transformed_images


def transform_test(image_dict, target_height=256, target_width=512):
    """
    Applies a center cropping.
    If the input is a tuple or list of images, the same transformation is applied to each image.
    
    Args:
        images (np.ndarray or tuple/list of np.ndarray): Input single-channel image or tuple/list of images.
        target_height (int): Desired height after resizing.
        target_width (int): Desired width after cropping.
        
    Returns:
        np.ndarray or tuple: Transformed image or tuple of transformed images.
    """
    def transform_single(image):
        # 1. Resize the rotated image so that its height is target_height while preserving aspect ratio.
        original_height, original_width = image.shape
        scale = target_height / original_height
        new_width = int(original_width * scale)
        resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        # 2. Center crop to target_width.
        width_left = (resized.shape[1] - target_width) // 2
        cropped = resized[:, width_left:width_left + target_width]
        return cropped
    
    transformed_images = {}
    for key, img in image_dict.items():
        img = transform_single(img)
        img = 1.0 - (img.astype(np.float32) / 255.0)
        img = np.expand_dims(img, axis=0)
        tensor = torch.from_numpy(img)
        transformed_images[key] = tensor
    return transformed_images

    
class TrainDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer=None,
        input_size=(256, 512),
        disable_prompts=False,
        ):
        self.data_root = data_root
        self.input_size = input_size
        self.contour_dir = os.path.join(data_root, 'contour')
        self.pressure_dir = os.path.join(data_root, 'pressure')
        self.temperature_dir = os.path.join(data_root, 'temperature')
        self.velocity_dir = os.path.join(data_root, 'velocity')
        
        self.filenames = [] 
        with open(os.path.join(data_root, FILES_TRAIN), 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]
        
        self.filepaths = []
        for attribute in ['pressure', 'temperature', 'velocity']:
            for filename in self.filenames:
                self.filepaths.append(os.path.join(attribute, filename))
                
        self._length = len(self.filepaths)
        
        self.tokenizer = tokenizer
        self.disable_prompts = disable_prompts 
        self.transform = transform_train

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        filepath = os.path.join(self.data_root, self.filepaths[i])
        
        if filepath.__contains__('pressure'):
            contour_path = filepath.replace('pressure', 'contour')
            text = 'pressure field'
        elif filepath.__contains__('temperature'):
            contour_path = filepath.replace('temperature', 'contour')
            text = 'temperature field'
        elif filepath.__contains__('velocity'):
            contour_path = filepath.replace('velocity', 'contour')
            text = 'velocity field'
        else:
            raise NotImplementedError
        
        contour_image  = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        target_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        h, w = contour_image.shape
        line_values = np.linspace(255, 0, w, dtype=np.float32)
        flow_image = np.tile(line_values, (h, 1))
        image_dict = {'input': contour_image,
                      'flow': flow_image,
                      'target': target_image}
        image_dict = self.transform(image_dict)
        
        example = {}
        example['text'] = tokenize_caption(self.tokenizer, text)
        example['inputs'] = torch.cat(
            (image_dict['input'].repeat(2, 1, 1), image_dict['flow']), dim=0)
        example['targets'] = image_dict['target'].repeat(3, 1, 1)
        
        return example
    
    
class InferDataset(Dataset):
    def __init__(
        self, 
        data_root,
        tokenizer, 
        input_size=(256, 512),
        generator=None):
        
        self.data_root = data_root
        self.input_size = input_size
        self.contour_dir = os.path.join(data_root, 'contour')
        self.pressure_dir = os.path.join(data_root, 'pressure')
        self.temperature_dir = os.path.join(data_root, 'temperature')
        self.velocity_dir = os.path.join(data_root, 'velocity')
        
        self.filenames = [] 
        with open(os.path.join(data_root, FILES_TEST), 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]
        
        self.filepaths = []
        for attribute in ['pressure', 'temperature', 'velocity']:
            for filename in self.filenames:
                self.filepaths.append(os.path.join(attribute, filename))
                
        self._length = len(self.filepaths)
        
        self.tokenizer = tokenizer
        self.generator = generator
        self.transform = transform_test

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        sample = {}
        filepath = os.path.join(self.data_root, self.filepaths[i])
        
        if filepath.__contains__('pressure'):
            contour_path = filepath.replace('pressure', 'contour')
            text = 'pressure field'
            key = 'pressure'
        elif filepath.__contains__('temperature'):
            contour_path = filepath.replace('temperature', 'contour')
            text = 'temperature field'
            key = 'temperature'
        elif filepath.__contains__('velocity'):
            contour_path = filepath.replace('velocity', 'contour')
            text = 'velocity field'
            key = 'velocity'
        else:
            raise NotImplementedError
        
        contour_image  = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        target_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        h, w = contour_image.shape
        line_values = np.linspace(255, 0, w, dtype=np.float32)
        flow_image = np.tile(line_values, (h, 1))
        image_dict = {'input': contour_image,
                      'flow': flow_image,
                      'target': target_image}
        
        image_dict = self.transform(image_dict)
        
        sample['text'] = self.tokenizer(
            text, padding='max_length', max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt').input_ids[0]

        sample['inputs'] = torch.cat(
            (image_dict['input'].repeat(2, 1, 1), image_dict['flow']), dim=0)
        sample['targets'] = image_dict['target'].repeat(3, 1, 1)
        sample['key'] = key
        return sample


if __name__ == '__main__':
    trainset = TrainDataset(data_root='data/case_data1/fluent_data_fig')
    for k, v in trainset[10].items():
        print(k, v.shape)
    