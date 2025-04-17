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
from torch.utils.data import Dataset, DataLoader

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


def transform_test(images, target_height=256, target_width=512):
    """
    Applies a random rotation (±10°) before resizing, random horizontal flip, and center cropping.
    If the input is a tuple or list of images, the same transformation is applied to each image.
    
    Args:
        images (np.ndarray or tuple/list of np.ndarray): Input single-channel image or tuple/list of images.
        target_height (int): Desired height after resizing.
        target_width (int): Desired width after cropping.
        
    Returns:
        np.ndarray or tuple: Transformed image or tuple of transformed images.
    """

    # first_img = images[0]
    
    # flow_img = np.expand_dims(flow_img, axis=-1)
    # Apply random rotation on the first image to determine the common transformation parameters.
    # angle = np.random.uniform(-15, 15)
    # center = (first_img.shape[1] // 2, first_img.shape[0] // 2)  # (x, y) center
    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation to the first image to compute the new dimensions (should remain same as original for cv2.warpAffine)
    # Use cv2.BORDER_REFLECT to avoid black borders.
    def transform_single(image):
        # # 1. Random Rotation
        # rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
        #                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 2. Resize the rotated image so that its height is target_height while preserving aspect ratio.
        original_height, original_width = image.shape
        scale = target_height / original_height
        new_width = int(original_width * scale)
        resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        # 3. Random horizontal flip (50% chance) using the same decision for all images.
        # (We generate the flip flag once outside, so here we assume that variable is defined)
        # if flip_flag:
        #     resized = np.fliplr(resized)
        # 4. Center crop to target_width.
        width_left = (resized.shape[1] - target_width) // 2
        cropped = resized[:, width_left:width_left + target_width]
        return cropped

    # Generate a random horizontal flip flag (True with 50% probability).
    # flip_flag = np.random.rand() < 0.5
    
    transformed_images = []
    for img in images:
        img = transform_single(img)
        img = 1.0 - (img.astype(np.float32) / 255.0)
        img = np.expand_dims(img, axis=0)
        tensor = torch.from_numpy(img)
        transformed_images.append(tensor)
    return transformed_images


class CaseDataDataset(Dataset):
    def __init__(self, root_dir, train=True):
        """
        Args:
            root_dir (str): Root directory containing the four child folders:
                'contour', 'pressure', 'temperature', 'velocity'.
            transform (callable, optional): Transform function to apply to each image.
                If None, a default resize transform is applied that sets the image's
                height to 256 while preserving its aspect ratio.
        """
        self.root_dir = root_dir
        self.contour_dir = os.path.join(root_dir, 'fluent_data_fig', 'contour')
        self.pressure_dir = os.path.join(root_dir, 'fluent_data_fig', 'pressure')
        self.temperature_dir = os.path.join(root_dir, 'fluent_data_fig', 'temperature')
        self.velocity_dir = os.path.join(root_dir, 'fluent_data_fig', 'velocity')
        # Assume file names are identical across subfolders.
        self.file_names = sorted(os.listdir(self.contour_dir))
        # Use provided transform or the default resize transform.
        self.transform = transform_train if train else transform_test

    def read_and_transform(self, path):
        # Read image in grayscale mode using OpenCV.
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        # Apply the transform to resize the image (shorter side height set to 256).
        img = self.transform(img)
        # Normalize: convert from 0-255 to 0-1, then invert (white becomes 0.0, black becomes 1.0).
        img = 1.0 - (img.astype(np.float32) / 255.0)
        # Add a channel dimension: (H, W) -> (1, H, W).
        img = np.expand_dims(img, axis=0)
        # Convert to torch tensor.
        tensor = torch.from_numpy(img)
        return tensor

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        # Construct full paths for each modality.
        contour_path     = os.path.join(self.contour_dir, file_name)
        pressure_path    = os.path.join(self.pressure_dir, file_name)
        temperature_path = os.path.join(self.temperature_dir, file_name)
        velocity_path    = os.path.join(self.velocity_dir, file_name)
        
        # Read and transform images.
        contour_image     = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        pressure_image    = cv2.imread(pressure_path, cv2.IMREAD_GRAYSCALE)
        temperature_image = cv2.imread(temperature_path, cv2.IMREAD_GRAYSCALE)
        velocity_image    = cv2.imread(velocity_path, cv2.IMREAD_GRAYSCALE)

        h, w = contour_image.shape
        line_values = np.linspace(255, 0, w, dtype=np.float32) # generate a fluent velocity data.
        flow_image = np.tile(line_values, (h, 1))
        tuple_images = (flow_image, contour_image, pressure_image, temperature_image, velocity_image)
        flow_tensor, contour_tensor, pressure_tensor, temperature_tensor, velocity_tensor = self.transform(tuple_images)

        # The contour image is the input and the other three are the targets.
        input_tensor = torch.cat((contour_tensor, flow_tensor), dim=0)
        target_tensor = torch.cat((pressure_tensor,temperature_tensor,velocity_tensor), dim=0)
        name = file_name.replace('s.tiff', '')
        return name, input_tensor, target_tensor
    
    
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
        example['inputs'] = torch.cat((image_dict['input'].repeat(2, 1, 1), image_dict['flow']),
                                      dim=0)
        example['targets'] = image_dict['target'].repeat(3, 1, 1)
        
        return example
    
        # tuple_images = (flow_image, contour_image, pressure_image, temperature_image, velocity_image)
        # flow_tensor, contour_tensor, pressure_tensor, temperature_tensor, velocity_tensor = self.transform(tuple_images)
        
        # entry = self.metadata[i]
        # example = {}
        # image = Image.open(os.path.join(self.data_root, entry['file_name']))
        # if not image.mode == 'RGB':
        #     image = image.convert('RGB')

        # trg_key = entry['file_name'].split('.')[0]
        # if self.target_extra_key:
        #     trg_key += f'-{self.target_extra_key}'
        # # if self.target_mode == 'RGB':
        # #     target = Image.open(io.BytesIO(self.txn.get(trg_key.encode())))
        # #     if not target.mode == 'RGB':
        # #         target = target.convert('RGB')
        # # elif self.target_mode == 'F':
        # target = np.load(io.BytesIO(self.txn.get(trg_key.encode())))['x']
        # if self.target_scale is not None:
        #     if self.target_scale == -1:
        #         target = (target - target.min()) / (target.max() - target.min())
        #     else:
        #         target = target / self.target_scale
        # # else:
        # #     raise NotImplementedError

        # if self.do_augment:
        #     x = self.aug_transform(image=np.array(image), mask=np.array(target))
        #     image, target = x['image'], x['mask']

        # example['rgb'] = self.transform(image)
        # example['pixel_values'] = self.transform(target)
        # # repeat 3 times to align with `RGB`
        # example['pixel_values'] = example['pixel_values'].repeat(3, 1, 1) 
        # return example


class InferDataset(Dataset):
    def __init__(self, prompts, tokenizer, latents=None, src_imgs=None, num_samples=None, generator=None):
        if isinstance(prompts, str):
            if os.path.isfile(prompts):
                print('Reading prompts from', prompts)
                self.prompts = open(prompts).read().splitlines()
            else:
                self.prompts = [prompts]
        elif hasattr(prompts, '__iter__'):
            self.prompts = prompts
        else:
            raise NotImplementedError('unsupported prompts', type(prompts))
        self.num_prompts = len(self.prompts)

        self.tokenizer = tokenizer

        self.latents = latents
        self.src_imgs = src_imgs
        self.generator = generator

        self.num_latents = 0
        if src_imgs and os.path.isdir(src_imgs):
            print('Using source images from', src_imgs)
            if os.path.isdir(src_imgs):
                self.src_imgs = glob.glob(os.path.join(src_imgs, '*.png')) + \
                        glob.glob(os.path.join(src_imgs, '*.jpg'))
                self.src_imgs.sort()
            elif os.path.isfile(src_imgs):
                self.src_imgs = [src_imgs]
            num_samples = num_samples or len(self.src_imgs)
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
        else:
            if latents is None:
                if generator is None:
                    print('WARNING: no generator for latents')
            else:
                if os.path.isdir(latents):
                    print('Using latents from', latents)
                    self.latents = glob.glob(os.path.join(latents, '*.npz'))
                    self.latents.sort()
                elif os.path.isfile(latents):
                    print('Using latents from', latents)
                    self.latents = [latents]
                self.num_latents = len(self.latents)

        self.num_samples = num_samples or max(self.num_prompts, self.num_latents)
        print(f'Genrating {self.num_samples} images')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        sample = {}
        sample['key'] = f'{i:05}'
        prompt = self.prompts[i%self.num_prompts]
        sample['text_ids'] = self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt').input_ids[0]

        if self.src_imgs is not None:
            path = self.src_imgs[i]
            src_img = Image.open(path).convert('RGB')
            sample['src_img'] = self.transform(src_img)
            sample['key'] = os.path.basename(path).split('.')[0]
        elif self.latents is None:
            sample['latents'] = torch.randn(4, 64, 64, generator=self.generator)
        else:
            path = self.latents[i%self.num_latents]
            latents = torch.tensor(np.load(path)['x'])
            if latents.size() == 4:
                latents.squeeze(0)
            sample['latents'] = latents
            sample['key'] = os.path.basename(path).split('.')[0]

        return sample


if __name__ == '__main__':
    trainset = TrainDataset(data_root='data/case_data1/fluent_data_fig')
    for k, v in trainset[10].items():
        print(k, v.shape)
    