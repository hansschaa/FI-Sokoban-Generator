import torch
import numpy as np
from scipy.stats import spearmanr
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor

device = torch.device("cpu")
model = SokobanSEResNetRegressor(dropout_p=0.0205).to(device)
model.load_state_dict(torch.load('surrogate_models/results/final_regressor_fold1.pt', map_location=device))
# wait, wait! I need to load the model I JUST trained! 
# But I didn't save the model I just trained!
