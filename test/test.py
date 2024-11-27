import numpy as np
import torch

input_path = "/root/humanoid-gym/test/dump_inputs/0.bin"
model = torch.jit.load("/root/humanoid-gym/logs/XBot_ppo/exported/policies/policy_1.pt")
inputs = np.fromfile(input_path, dtype=np.float32)
inputs = torch.tensor(inputs)
print(model(inputs))