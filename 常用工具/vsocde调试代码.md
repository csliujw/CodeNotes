# 深度学习代码调试

## 调试分布式训练代码

[VS Code调试pytorch分布式训练脚本](https://blog.csdn.net/qianbin3200896/article/details/108182504)

[PyTorch分布式训练简介_nporc_per_node-CSDN博客](https://blog.csdn.net/baidu_19518247/article/details/89635181)

> 常见的 PyTorch 分布式训练命令如下

```shell
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --model bisenetv2
```

- 常规的 Python 文件启动方式是 `python xxx.py`
- PyTorch 分布式训练则是 `python -m torch.distributed.launch`，它会启动 torch/distributed/launch.py，自动去调用 torch 分布式启动文件 launch.py 执行代码。
- `--nproc_per_node=2` 表示使用两个节点进行分布式训练。

> VSCode 配置分布式训练相关文件

创建 debug 需要的 launch.json 文件，为 debug 设置配置参数。

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            // 分布式启动代码的路径
            "program": "/home/xxx/torch/distributed/launch.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            // 分布式训练参数设置
            "args":[
                "--nproc_per_node=1",
                "tools/train.py",
                "--model", // 控制台参数模型参数
            ],
    					"env": { "CUDA_VISIBLE_DEVICES" : "0" }
        }
    ]
}
```

python 代码，设置主服务器的 IP 和地址。

```python
import torch.distributed as dist
import os
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '5678'
```

## 控制台参数的设置

args

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            // 分布式启动代码的路径
            "program": "/home/xxx/torch/distributed/launch.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            // 分布式训练参数设置
            "args":[
                "--nproc_per_node=1", // 可以用等号
                "--model", "resnet101"// 控制台参数模型参数
            ],
    					"env": { "CUDA_VISIBLE_DEVICES" : "0" }
        }
    ]
}
```

