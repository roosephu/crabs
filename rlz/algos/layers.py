import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
from rlz.multi_layer_perception import MLP


class BatchLinear(nn.Module):
    def __init__(self, n, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n = n
        self.weight = nn.Parameter(torch.empty((n, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((n, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            bound = 1 / np.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

        gain = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
        std = gain / np.sqrt(self.in_features)
        bound = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # weight: shape = NIO
        shape = input.shape
        if len(shape) == 2:  # shape = NI
            assert shape == [self.n, self.in_features]
            output = torch.einsum('bi,nio->nbo', input, self.weight)
            if self.bias is not None:
                return output + self.bias
            return output
        if len(shape) == 3:  # shape = NBI
            assert shape[0] == self.n and shape[2] == self.in_features
            output = torch.matmul(input, self.weight)
            if self.bias is not None:
                return output + self.bias
            return output

        raise NotImplementedError

    def extra_repr(self) -> str:
        return f'n={self.n}, in_features={self.in_features}, out_features={self.out_features}'

    @staticmethod
    def from_nets(nets: Sequence[nn.Linear]) -> 'BatchLinear':
        n = len(nets)
        bias = nets[0].bias is not None
        layer = BatchLinear(n, nets[0].in_features, nets[0].out_features, bias=bias, device=nets[0].weight.device,
                            dtype=nets[0].weight.dtype)
        for i, net in enumerate(nets):
            layer.weight.data[i] = net.weight.T.clone().detach()
            if bias:
                layer.bias.data[i] = net.bias.clone().detach()

        return layer


class BatchSequential(nn.Sequential):
    @staticmethod
    def from_nets(nets: Sequence[nn.Sequential]) -> 'BatchSequential':
        modules = [list(net.children()) for net in nets]
        assert [len(x) == len(modules[0]) for x in modules]
        modules = [convert_to_batch_module(subnets) for subnets in zip(*modules)]
        return BatchSequential(*modules)


class BatchMultiLayerPerceptron(BatchSequential):
    def __init__(self, n, n_units, activation=nn.ReLU, auto_squeeze=True, output_activation=None):
        layers = []
        for in_features, out_features in zip(n_units[:-1], n_units[1:]):
            if layers:
                layers.append(activation())
            layers.append(nn.Linear(in_features, out_features))
        if output_activation:
            layers.append(output_activation())
        super().__init__(*layers)

        self._n = n
        self._n_units = n_units
        self._auto_squeeze = auto_squeeze
        self._activation = [activation]  # to prevent nn.Module put it into self._modules

    def fast_forward(self, inputs):
        outputs = inputs
        for layer in self:
            outputs = layer(outputs)

        if self._auto_squeeze and outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)
        return outputs

    def forward(self, *inputs):
        inputs = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
        return self.fast_forward(inputs)

    def extra_repr(self):
        return f'n = {self._n}, activation = {self._activation}, ' \
               f'# units = {self._n_units}, squeeze = {self._auto_squeeze}'

    @staticmethod
    def from_nets(nets: Sequence[MLP]) -> 'BatchMultiLayerPerceptron':
        modules = [list(net.children()) for net in nets]
        assert [len(x) == len(modules[0]) for x in modules]
        modules = [convert_to_batch_module(subnets) for subnets in zip(*modules)]
        ret = BatchMultiLayerPerceptron(len(nets), [],
                                        auto_squeeze=nets[0]._auto_squeeze, activation=nets[0]._activation[0])
        ret._n_units = nets[0]._n_units
        for idx, module in enumerate(modules):
            ret.add_module(str(idx), module)
        return ret


def convert_to_batch_module(modules: Sequence[nn.Module]) -> nn.Module:
    if isinstance(modules[0], MLP):
        return BatchMultiLayerPerceptron.from_nets(modules)
    if isinstance(modules[0], nn.Sequential):
        return BatchSequential.from_nets(modules)
    if isinstance(modules[0], nn.Linear):
        return BatchLinear.from_nets(modules)
    if isinstance(modules[0], nn.ReLU):
        return nn.ReLU()
    assert False, f"unknown type: {[type(net) for net in modules]}"
