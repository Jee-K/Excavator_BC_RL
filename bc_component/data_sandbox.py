import numpy as np

with open("./bc_component/bc_dataset.npz", "rb") as f:
  with np.load(f) as data:
    zipped_state_action = [x for x in zip(data['states'], data['actions'])]
    print(len(zipped_state_action))