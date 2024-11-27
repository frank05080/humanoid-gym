import torch

obs = torch.rand(1,705)
print(obs.dtype)
print(obs.shape)

model = torch.jit.load("/root/humanoid-gym/logs/XBot_ppo/exported/policies/policy_1.pt")

torch.onnx.export(
    model,
    args=(obs),
    f="humanoid_policy.onnx",
    input_names=["policy"],
    output_names=["action"],
    verbose=True,
)
