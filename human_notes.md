change cuda device:

isaacgym/python/isaacgym/gymutil.py中修改device


python train.py --task=humanoid_ppo --headless --load_run log_file_path --run_name run_name --sim_device=cuda:2 --rl_device=cuda:2

wandb API: db67a08d0626fdab7c943e5db5279735eceeaa83

python train.py --task=humanoid_ppo --load_run log_file_path --run_name run_name


## 安装 vulkan，否则执行gym.draw_viewer(viewer, sim, True) 会报错seg fault

1. pkg upgrade

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt upgrade
```

2. install vulkan tools

```
sudo apt install vulkan-tools
```

3. test vulkan runtime env

```
vulkaninfo
vkcube
```

4. set up vulkan dev environment

libvulkan-dev: vulkan dev library
libglfw3: cross platform window management library
libglm-dev: linear algebra library
libxi-dev: input library

```
sudo apt install libvulkan-dev
sudo apt install libglfw3-dev
sudo apt install libglm-dev
sudo apt-get install libxi-dev
```

5. example program

```
#define GLFW_INCLUDE_VULKAN 
/*
without this, I got error: ‘vkEnumerateInstanceExtensionProperties’ was not declared in this scope
*/
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
int main() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::cout << extensionCount << " extensions supported\n";
    glm::mat4 matrix;
    glm::vec4 vec;
    auto test = matrix * vec;
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
```

MakeFile

```
CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXrandr -lXi
VulkanTest: main.cpp
    g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)
.PHONY: test clean
test: VulkanTest
    ./VulkanTest
clean:
    rm -f VulkanTest
```

Before running humanoid, do `export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` or `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json.`

## 如果想launch.json生效，必须定义configurations中的name、type、request、program的参数

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Train Humanoid PPO",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "cwd": "${workspaceFolder}",
            "justMyCode": false,
            "args": [
                "--task",
                "humanoid_ppo",
                "--headless",
                "--load_run", "log_file_path",
                "--run_name", "run_name",
                "--sim_device=cuda:2",
                "--rl_device=cuda:2"
            ]
        }
    ],
}
```

## 模型保存机制

模型保存在logs/XBot_ppo/Date_time_run_name/model_iterNum.pt下

run_name在参数中定义

一共迭代3000个iteration，每个iteration，做60次step，对于4096个envs

## 运行

python train.py --task=humanoid_ppo --run_name run_name --headless

python play.py --task=humanoid_ppo --run_name run_name --load_run /root/humanoid-gym/logs/XBot_ppo/Nov13_14-08-08_run_name 

load_run参数只有在play时才会用到

play结束后，会将视频储存至videos下面


## GPU模型保存和加载的坑：

报错
```
Traceback (most recent call last):
  File "play.py", line 169, in <module>
    play(args)
  File "play.py", line 77, in play
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
  File "/root/humanoid-gym/humanoid/utils/task_registry.py", line 159, in make_alg_runner
    runner.load(resume_path, load_optimizer=False)
  File "/root/humanoid-gym/humanoid/algo/ppo/on_policy_runner.py", line 293, in load
    loaded_dict = torch.load(path)
  File "/root/miniconda3/envs/myenv/lib/python3.8/site-packages/torch/serialization.py", line 789, in load
    return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
  File "/root/miniconda3/envs/myenv/lib/python3.8/site-packages/torch/serialization.py", line 1131, in _load
    result = unpickler.load()
  File "/root/miniconda3/envs/myenv/lib/python3.8/site-packages/torch/serialization.py", line 1101, in persistent_load
    load_tensor(dtype, nbytes, key, _maybe_decode_ascii(location))
  File "/root/miniconda3/envs/myenv/lib/python3.8/site-packages/torch/serialization.py", line 1083, in load_tensor
    wrap_storage=restore_location(storage, location),
  File "/root/miniconda3/envs/myenv/lib/python3.8/site-packages/torch/serialization.py", line 215, in default_restore_location
    result = fn(storage, location)
  File "/root/miniconda3/envs/myenv/lib/python3.8/site-packages/torch/serialization.py", line 182, in _cuda_deserialize
    device = validate_cuda_device(location)
  File "/root/miniconda3/envs/myenv/lib/python3.8/site-packages/torch/serialization.py", line 173, in validate_cuda_device
    raise RuntimeError('Attempting to deserialize object on CUDA device '
RuntimeError: Attempting to deserialize object on CUDA device 2 but torch.cuda.device_count() is 1. Please use torch.load with map_location to map your storages to an existing device.
```

This error is occurring because the saved model was trained on a system with multiple GPUs (specifically, it appears to have been saved on CUDA device 2), while your current system only has one GPU. The `torch.load` function is trying to load the model onto a non-existent CUDA device.

To fix this, use the `map_location` argument in `torch.load` to specify a valid device for loading. For example, you can map all CUDA devices to `cuda:0` (your single available GPU) or `cpu` if you want to load the model on the CPU instead.

Here’s how you can modify the `load` function call to include `map_location`:

```python
# Modify this line to load with map_location
loaded_dict = torch.load(path, map_location='cuda:0')  # Or 'cpu' if you prefer
```

This should map the model to the correct device and prevent the `RuntimeError`. Let me know if you need further assistance!