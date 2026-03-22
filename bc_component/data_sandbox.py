import numpy as np

import os
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

with open("./bc_component/bc_dataset.npz", "rb") as f:
  with np.load(f) as data:
    zipped_state_action = [x for x in zip(data['states'], data['actions'])]
    print(len(zipped_state_action), len(zipped_state_action[0][0]))