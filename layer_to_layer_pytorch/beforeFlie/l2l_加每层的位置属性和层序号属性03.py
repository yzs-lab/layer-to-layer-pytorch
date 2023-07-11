from typing import List, Optional

import torch
from torch import nn

from layer_to_layer_pytorch.helpers import enumerator
from layer_to_layer_pytorch.types import Criterion, Device


# class Layer2Layer:
class Layer2Layer:
    def __init__(
        self,
        model: nn.Module,
        microbatch_size: Optional[int],
        layers_attr: str = "layers",
        mixed_precision: bool = False,
        loss_scale: float = 128.0,  # 2**7
        gpu_device: Device = "cuda",
        verbose: bool = False,
    ):
        print("进入 Layer2Layer init")
        layers = getattr(model, layers_attr, None)
        if (layers is None) or (not isinstance(layers, nn.ModuleList)):
            raise ValueError(
                f"Model must contain `nn.ModuleList` in attribute `{layers_attr}`."
                f"Got {type(layers)}"
            )

        if (microbatch_size is not None) and (microbatch_size < 0):
            raise ValueError(
                f"Size of a microbatch must be greater than zero."
                f"Got microbatch_size={microbatch_size}"
            )

        if mixed_precision and loss_scale <= 0.0:
            raise ValueError(
                f"Loss scale cannot less or equal to zero if mixed_precision is True."
                f"Got loss_scale={loss_scale}"
            )

        # model stuff
        self.layers_attr: str = layers_attr
        self.model: nn.Module = model.cpu()
        self._master_params = self._copy_master_params(self.model)

        # hyperparams
        self.microbatch_size: Optional[int] = microbatch_size
        self.gpu_device: Device = gpu_device

        # fp16 stuff
        self.mixed_precision = mixed_precision
        self.loss_scale = loss_scale

        if self.mixed_precision:
            self.model.half()

        self.verbose: bool = verbose

        # inner stuff
        self._num_layers: int = len(layers)
        self._activations: List[torch.Tensor] = []
        self._grads: List[torch.Tensor] = []
    
    # @property: 这是一个装饰器，用于定义一个属性的 getter 方法。
    @property
    # 这是一个方法 num_layers，用于获取模型的层数
    def num_layers(self) -> int:
        #  返回属性 _num_layers 的值。属性 _num_layers 是在类的构造函数中初始化的，表示模型的层数。
        # 以一个下划线 _ 开头的变量或属性通常被视为内部使用或私有的。这是一种约定，用于表示这些变量或属性仅供类内部使用，不应该在类外部直接访问。
        return self._num_layers

    @property
    def main_params(self):
        return self._master_params

    def zero_grad(self) -> None:
        # print("来喽")
        for model_param, master_param in zip(
            self.model.parameters(), self._master_params
        ):
            model_param.grad = None
            master_param.grad = None

        self._reset_activations()

    def update_main_model_params(self):
        for model_params, master_params in zip(
            self.model.parameters(), self._master_params
        ):
            model_params.data.copy_(master_params.data)

    @torch.no_grad()
    # def forward(self, batch: torch.Tensor, **kwargs, nlayer_by_nlayer) -> torch.Tensor:
    
    def forward(self, batch: torch.Tensor,nlayer_by_nlayer: int , **kwargs) -> torch.Tensor:
        # print("进来啦")
        # 检查是否使用混合精度
        if self.mixed_precision:
            # 如果使用混合精度，将输入数据转换为半精度（FP16）并添加到激活值列表中。
            self._activations.append(batch.half())
        else:
            # 如果不使用混合精度，将输入数据添加到激活值列表中。
            self._activations.append(batch)
            
        # 使用 enumerator 函数遍历模型的各个层。
        # enumerator 函数根据 verbose 参数的值选择使用 enumerate 函数或者 tenumerate 函数来进行遍历。
        # 在这里，我们使用 tenumerate 函数，它是 tqdm.contrib.tenumerate 的别名，用于在遍历过程中显示进度条和其他详细信息。
        # idx 是当前层的索引，layer 是当前层的对象。
        for idx, layer in enumerator(
            self._get_layers(),
            verbose=self.verbose,
            desc="Layers",
            total=self.num_layers,
            leave=False,
        ):
            print(f"forward idx={idx} num_layers={self.num_layers}")
            
            key="key"
            value="value"
            setattr(layer, key, value)
            
            assert value == getattr(layer, key)
            
            # host-to-device weight   (layer)
            # 1.1 主机 到 GPU: 将层移动到指定的 GPU 设备。
            layer.to(self.gpu_device)
            print(f"layer={layer}")
            input: torch.Tensor = self._activations[idx]
            microbatch_size = self._get_microbatch_size(input)

            layer_activations: List[torch.Tensor] = []
            # 1.2 主机 到 GPU
            for microbatch in input.split(microbatch_size):
                activation: torch.Tensor = layer(
                    microbatch.to(self.gpu_device), **kwargs
                )

                # 激活值 
                # 2.1 GPU到主机
                layer_activations.append(activation.cpu())
            # 剪切  device-to-host  删除device上的 layer weight (weight + 激活函数  新的激活值)
            # 2.2 GPU 到 主机
            # layer.cpu()
            
            # 需要还回的层数
            back_nlayer_by_nlayer = 0
            if nlayer_by_nlayer == 0:
                layer.cpu()
            # 如果没到最后一层  
            # elif idx <= self.num_layers:
            else:
                nlayer_by_nlayer -= 1
                back_nlayer_by_nlayer += 1
                
            # else:
            #     while back_nlayer_by_nlayer == 0:
            #         layer.cpu()
            #         back_nlayer_by_nlayer -= 1
                
            self._activations.append(torch.cat(layer_activations, dim=0))

        return self._activations[-1]


    # def forward(self, batch: torch.Tensor,nlayer_by_nlayer: int) -> torch.Tensor:
    
    # def forward(self, batch: torch.Tensor) -> torch.Tensor:
    #     print("来喽")


    def compute_loss(
        self, targets: torch.Tensor, criterion: Criterion, **criterion_kwargs
    ) -> float:
        loss_value = 0.0
        grads = []

        inputs: torch.Tensor = self._activations[-1]
        microbatch_size = self._get_microbatch_size(inputs)
        num_steps: int = inputs.shape[0] // microbatch_size

        for _activation, _target in zip(
            inputs.split(microbatch_size), targets.split(microbatch_size)
        ):
            activation = _activation.to(self.gpu_device).requires_grad_(True)
            target = _target.to(self.gpu_device)

            loss = (
                criterion(activation.float(), target, **criterion_kwargs)
                / num_steps
            )
            loss_value += loss.item()  # Append Before Scaling

            if self.mixed_precision and self.loss_scale != 0:
                loss *= self.loss_scale

            loss.backward()
            grads.append(activation.grad.cpu())

        self._grads.append(torch.cat(grads, dim=0))
        return loss_value

    def backward(self) -> None:
        # print("backward啦")
        for idx, (layer, activations) in enumerator(
            zip(reversed(self._get_layers()), reversed(self._activations[:-1])),
            verbose=self.verbose,
            desc="Reverse Layers",
            total=self.num_layers,
            leave=False,
        ):
            print(f"backward idx={idx}")
            # 主机-GPU： 将当前层移至指定的GPU设备上进行计算。
            layer.to(self.gpu_device)

            microbatch_size = self._get_microbatch_size(activations)
            grads = []

            for _activation, gradient in zip(
                activations.split(microbatch_size),
                self._grads[idx].split(microbatch_size),
            ):
                # 怎么迅速知道这一层layer的位置 位于gpu还在cpu，或者怎么验证它已经在gpu/cpu，每层有独立的标号吗，每层的存储位置可以怎么识别吗
                # 主机-GPU：将激活值移至指定的GPU设备，并设置requires_grad为True，以便计算梯度。
                activation: torch.Tensor = _activation.to(
                    self.gpu_device
                ).requires_grad_(True)
                output: torch.Tensor = layer(activation) # 计算
                
                output.backward(gradient.to(self.gpu_device)) # 计算
                # GPU-主机：将计算得到的梯度移回主机，并添加到梯度列表中。
                grads.append(activation.grad.cpu())
            # GPU-主机：将当前层移回主机。
            layer.cpu()
            self._grads.append(torch.cat(grads, dim=0))
        self._model_grad_to_master()


    def backward_save230711(self) -> None:
        # print("backward啦")
        for idx, (layer, activations) in enumerator(
            zip(reversed(self._get_layers()), reversed(self._activations[:-1])),
            verbose=self.verbose,
            desc="Reverse Layers",
            total=self.num_layers,
            leave=False,
        ):
            # 
            
            layer.to(self.gpu_device)

            microbatch_size = self._get_microbatch_size(activations)
            grads = []

            for _activation, gradient in zip(
                activations.split(microbatch_size),
                self._grads[idx].split(microbatch_size),
            ):
                activation: torch.Tensor = _activation.to(
                    self.gpu_device
                ).requires_grad_(True)
                output: torch.Tensor = layer(activation) # 计算

                output.backward(gradient.to(self.gpu_device)) # 计算
                #
                grads.append(activation.grad.cpu())
            # 
            layer.cpu()
            self._grads.append(torch.cat(grads, dim=0))
        self._model_grad_to_master()



    def _get_microbatch_size(self, batch: torch.Tensor) -> int:
        batch_size = batch.shape[0]
        return (
            batch_size if self.microbatch_size is None else self.microbatch_size
        )

    def _copy_master_params(self, model):
        master_params = [
            #  PyTorch sets `requires_grad = False` when clone and detach
            p.detach().clone().float().requires_grad_(True)
            for p in model.parameters()
            if p.requires_grad == True
        ]

        return master_params

    def __len__(self) -> int:
        return self._num_layers

    def _reset_activations(self):
        self._activations = []
        self._grads = []
        
    #  这是一个类内部的私有方法 _get_layers，用于获取模型的层。
    def _get_layers(self) -> nn.ModuleList:
        # 使用 getattr 函数获取模型对象 self.model 的属性 self.layers_attr 对应的值。
        # self.layers_attr 是在类的构造函数中传入的参数，表示模型对象中包含层的属性的名称。
        # 通过调用 _get_layers 方法，我们可以获取到模型中的层对象，以便后续的遍历和计算。
        return getattr(self.model, self.layers_attr)

    def _model_grad_to_master(self):
        for model_param, master_param in zip(
            self.model.parameters(), self._master_params
        ):
            if master_param.grad is None:
                master_param.grad = torch.empty_like(master_param.data).float()

            master_param.grad.data.copy_(model_param.grad.data)

            if self.mixed_precision and self.loss_scale != 0:
                master_param.grad.data = (
                    master_param.grad.data / self.loss_scale
                )


__all__ = ["Layer2Layer"]
