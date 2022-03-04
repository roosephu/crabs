import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, n_units, activation=nn.ReLU, auto_squeeze=True, output_activation=None):
        layers = []
        for in_features, out_features in zip(n_units[:-1], n_units[1:]):
            if layers:
                layers.append(activation())
            layers.append(nn.Linear(in_features, out_features))
        if output_activation:
            layers.append(output_activation())
        super().__init__(*layers)

        self._n_units = n_units
        self._auto_squeeze = auto_squeeze
        self._activation = [activation]  # to prevent nn.Module put it into self._modules

    def forward(self, *inputs):
        inputs = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)

        outputs = inputs
        for layer in self:
            outputs = layer(outputs)

        if self._auto_squeeze and outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)
        return outputs

    def copy(self):
        return MultiLayerPerceptron(self._n_units, self._activation[0], self._auto_squeeze)

    def extra_repr(self):
        return f'activation = {self._activation}, # units = {self._n_units}, squeeze = {self._auto_squeeze}'


MLP = MultiLayerPerceptron
