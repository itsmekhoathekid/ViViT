from moviepy.editor import VideoFileClip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import whisper
import tempfile
import os 


class PreExtractedFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_paths = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(feature_dir)))}
        
        for cls in sorted(os.listdir(feature_dir)):
            cls_path = os.path.join(feature_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith('.pt'):
                    self.feature_paths.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feature = torch.load(self.feature_paths[idx])
        frames = feature['frames']
        mfcc = feature['mfcc']
        text_embedding = feature['text_embedding']
        label = self.labels[idx]

        return {
            'frames': frames,
            'mfcc': mfcc,
            'text_embedding': text_embedding,
            'label': label
        }

import logging
import os 
def logg(log_file):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # vẫn in ra màn hình
        ]
    )