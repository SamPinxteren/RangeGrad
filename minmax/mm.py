import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

class mmTensor():
    """
    Object stores a center, an upper offset and a lower offset.
    """
    def __init__(self, input, device=None):
        self.device = device
        self.masks = None
        """
        self.__data is a concatenation of 3 equally sized matrices:
         __min <= __center <= __max
        """
        if isinstance(input, torch.Tensor):
            self.__min = torch.clone(input)
            self.__center = input
            self.__max = torch.clone(input)
            self.__batch_size = input.shape[0]
        elif isinstance(input, tuple) and len(input) == 3:
            x_min, center, x_max = input
            if type(x_min) != torch.Tensor or type(x_max) != torch.Tensor or type(center) != torch.Tensor:
                raise ValueError("Tuple must contain three tensors")
            if x_min.shape != x_max.shape or x_min.shape != center.shape:
                raise ValueError("Tuple must contain three tensors of the same size")
            self.__batch_size = center.shape[0]
            self.__min, self.__center, self.__max = input
        else:
            raise ValueError("\"input\" must be a tensor or a tuple of two (lower, upper) or three (down offset, center, up offset) tensors")

    def lower(self):
        return self.__min
    
    def center(self):
        return self.__center

    def upper(self):
        return self.__max

    def data(self):
        return self.__min, self.__center, self.__max
    
    def batch_size(self):
        return self.__batch_size

    def shape(self):
        return (self.__batch_size, ) + self.__center.shape[1:]
    
    def add_mask(self, lower_mask, upper_mask):
        self.__min = self.__center - lower_mask
        self.__max = self.__center + upper_mask
        return self
        
    def __iadd__(self, other):
        other_min, other_center, other_max = other.data()
        self.__min += other_min
        self.__center += other_center
        self.__max += other_max
        return self

    def add_rand_range(self, range_width):
        self.__min -= torch.rand(size=self.shape(),device=self.device) * (range_width / 2)
        self.__max += torch.rand(size=self.shape(),device=self.device) * (range_width / 2)
        return self
    
    def get_range(self):
        return torch.abs(self.lower() - self.upper()).mean().item()
    
    def scale_range(self, factor):
        if factor != 1:
            self.__min += (self.__center - self.__min) * (1-factor)
            self.__max += (self.__center - self.__max) * (1-factor)

    def cpu(self):
        return mmTensor((self.lower().cpu(), self.center().cpu(), self.upper().cpu()))

def _relu_process(self, input):
    scale = None
    if hasattr(self, 'autoscale_relu'):
        scale = 1
        with torch.no_grad():
            allowed = input.center().numel() * self.autoscale_relu
            L = input.center() / (input.center() - input.lower())
            U = input.center() / (input.center() - input.upper())
            distances = F.relu(L * (L < 1)) + F.relu(U * (U < 1))

            overcrossed = int(torch.sum(distances > 0).item() - allowed)
            if overcrossed > 0:
                scale, index = torch.kthvalue(-distances.flatten(), overcrossed)
                scale = -scale.item()
        input.scale_range(scale)
            
    if hasattr(self, 'debug_relu'):
        relus_crossed = torch.sum(torch.logical_and(input.lower() <= 0, input.upper() >= 0)).item()
        relus_total = input.center().numel()
        print(type(self).__name__.ljust(16), str(relus_crossed).rjust(16), "/", str(relus_total).ljust(10), "=", "{:.4f}%".format(100 * relus_crossed / relus_total), " - Scale: ", scale)

def _layer_print(self, out, relu=False):
    if hasattr(self, 'debug_range'):
        r = log(out.get_range())
        print(type(self).__name__.ljust(16), r)

def _layer_postprocess(self, out):
    if hasattr(self, 'scale_factor'):
        out.scale_range(self.scale_factor)
    return out
    
class Conv2d(nn.Conv2d):
    def forward(self, input):
        W_p = F.relu(self.weight)
        W_n = F.relu(-self.weight)
        
        min_p = self._conv_forward(input.lower(), W_p, self.bias)
        min_n = self._conv_forward(input.lower(), W_n, None)
        max_p = self._conv_forward(input.upper(), W_p, self.bias)
        max_n = self._conv_forward(input.upper(), W_n, None)
        x = self._conv_forward(input.center(), self.weight, self.bias)
        
        out = mmTensor((min_p - max_n, x, max_p - min_n))
        _layer_print(self, out)
        return _layer_postprocess(self, out)

class Linear(nn.Linear):
    def forward(self, input):
        W_p = F.relu(self.weight)
        W_n = F.relu(-self.weight)

        out_min = F.linear(input.lower(), W_p, self.bias) - F.linear(input.upper(), W_n)
        out_max = F.linear(input.upper(), W_p, self.bias) - F.linear(input.lower(), W_n)
        x = F.linear(input.center(), self.weight, self.bias)

        out = mmTensor((out_min, x, out_max))
        _layer_print(self, out)
        return _layer_postprocess(self, out)

class BatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        lower = super().forward(input.lower())
        center = super().forward(input.center())
        upper = super().forward(input.upper())
        out = mmTensor((lower, center, upper))
        _layer_print(self, out)
        return out

class ReLU(nn.ReLU):
    def forward(self, input):
        _relu_process(self, input)
        lower = super().forward(input.lower())
        center = super().forward(input.center())
        upper = super().forward(input.upper())
        out = mmTensor((lower, center, upper))
        _layer_print(self, out)
        return out

class MaxPool2d(nn.MaxPool2d):
    def forward(self, input):
        lower = super().forward(input.lower())
        center = super().forward(input.center())
        upper = super().forward(input.upper())
        out = mmTensor((lower, center, upper))
        _layer_print(self, out)
        return out

class AvgPool2d(nn.AvgPool2d):
    def forward(self, input):
        lower = super().forward(input.lower())
        center = super().forward(input.center())
        upper = super().forward(input.upper())
        out = mmTensor((lower, center, upper))
        _layer_print(self, out)
        return out

class Dropout(nn.Dropout):
    def forward(self, input):
        lower = super().forward(input.lower())
        center = super().forward(input.center())
        upper = super().forward(input.upper())
        out = mmTensor((lower, center, upper))
        _layer_print(self, out)
        return out

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input):
        lower = super().forward(input.lower())
        center = super().forward(input.center())
        upper = super().forward(input.upper())
        out = mmTensor((lower, center, upper))
        _layer_print(self, out)
        return out

def cat(input, *args):
    if type(input) == mmTensor:
        return input
    if type(input) == list:
        lower = torch.cat([i.lower() for i in input], *args)
        center = torch.cat([i.center() for i in input], *args)
        upper = torch.cat([i.upper() for i in input], *args)
        return mmTensor((lower, center, upper))

def flatten(input, *args):
    lower = torch.flatten(input.lower(), *args)
    center = torch.flatten(input.center(), *args)
    upper = torch.flatten(input.upper(), *args)
    return mmTensor((lower, center, upper))

def LogSoftMax(x):
    S_min = x.lower() - (x.upper().exp().sum() - x.upper().exp() + x.lower().exp()).log()
    S_max = x.upper() - (x.lower().exp().sum() - x.lower().exp() + x.upper().exp()).log()
    m = nn.LogSoftmax()
    S = m(x.center())
    return mmTensor((S_min, S, S_max))
