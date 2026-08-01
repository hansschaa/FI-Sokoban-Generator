import torch
import json
import numpy as np
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetClassifier
from data.prepare_classifier import encode_board

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SokobanSEResNetClassifier(dropout_p=0.4, in_channels=12).to(device)
model.load_state_dict(torch.load('surrogate_models/results/final_contrastive_classifier_fold1.pt', map_location=device, weights_only=False))
model.eval()

# A known complex CORRAL deadlock from the previous iterations
corral_board_str = """#################
#    #  ##   #  #
# #  # $##   #  #
#   $#    $  #  #
###$  $ ####. .##
##      ####   ##
#   ###      .###
#          .  # #
# # ## #  #   # #
#   ## ## ## @# #
#################"""

t_b = encode_board(corral_board_str)
t_np = np.concatenate([t_b, t_b], axis=0) # Server fallback behavior
batch_tensor = torch.from_numpy(t_np).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(batch_tensor)
    prob = torch.sigmoid(logits).item()

print(f"Probabilidad de que sea SOLUCIONABLE: {prob:.4f}")
print(f"Predicción (umbral 0.65): {'SOLUCIONABLE (Falso Positivo!)' if prob >= 0.65 else 'DEADLOCK (Correcto)'}")
