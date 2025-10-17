import numpy as np
from datasets.ecg_id import ECGIDDataset
from datasets.heartprint import HeartPrintDataset
# from preprocessing.filters import bandpass_filter, normalize
# from segmentation.simple import sliding_window
# from models.identification_cnn import CNN1DIdentifier
from torch.utils.data import DataLoader
import torch
import os
from sklearn.model_selection import train_test_split

import config
from config import CONFIG
from preprocessing.filters import Preprocess
from segmentation.rcentered import RCentered
from utils.utils import ECGUtils
from models.deepecg import DeepECGModel
from models.siamese import SiameseECGModel
from models.multibranch import MultiBranchECGModel
# from models.transformer import TransformerModel

import argparse

datasets_path = os.path.join(os.getcwd(), "datasets")
data_path = os.path.join(datasets_path, "ECG-ID")

Person_List = [f for f in os.listdir(data_path) if f.startswith("Person")]
print(f"Total Subjects: {len(Person_List)}")

