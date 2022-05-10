#!/usr/bin/env python

import torch

class Sequential(torch.nn.Sequential):
  def reverse(self, input):
    for module in reversed(self):
      input = module.reverse(input)
    return input

class Flatten(torch.nn.Flatten):
    def __init__(self, *args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.computing = False
        self.original_shape = None

    def forward(self, input):
        if not self.computing:
            self.original_shape = input.shape[1::]
            self.computing=True
        return input.flatten(self.start_dim, self.end_dim)

    def reverse(self,input):
        return input.view(input.shape[0],*self.original_shape)

class Linear(torch.nn.Linear):
    def reverse(self,input):
        bias = 0
        if self.bias != None:
            bias = self.bias
        return torch.nn.functional.linear(input-bias,torch.linalg.pinv(self.weight),None)

class Conv2d(torch.nn.Conv2d):
  def _conv_reverse(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            raise Exception('None zero padding mode reverse function not implemented')
        return torch.nn.functional.conv_transpose2d(input-bias.view(1,bias.numel(),1,1),weight, bias=0., stride=self.stride,
                        padding=self.padding, output_padding=0, groups=self.groups, dilation=self.dilation)

  def reverse(self,input):
    return self._conv_reverse(input, self.weight, self.bias)

class MaxPool2d(torch.nn.MaxPool2d):
  def __init__(self, *args,**kwargs) :
      super().__init__(return_indices=True,*args,**kwargs)
      self.indices=None
      self.output_size =None

  def forward(self,input):
    output,indices=super().forward(input)
    self.indices = indices
    self.output_size=input.size()
    return output

  def reverse(self,input):
    return torch.nn.functional.max_unpool2d(input,self.indices,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,output_size=self.output_size)

class ReLU(torch.nn.ReLU):
  def reverse(self,input):
    return torch.nn.functional.relu(input, inplace=self.inplace)

class LeakyReLU(torch.nn.LeakyReLU):
  def reverse(self,input):
    return torch.nn.functional.leaky_relu(input, self.negative_slope, self.inplace)
    # return input+(0.5-0.5*torch.abs(input)/input)*(1./self.negative_slope-1.)*input
class SELU(torch.nn.SELU):
  def reverse(self,input):
    return torch.nn.functional.selu(input, self.inplace)

class Sigmoid(torch.nn.Sigmoid):
  def reverse(self,input):
    # return torch.log(input/(torch.ones_like(input)-input))
    return torch.sigmoid(input)

class Tanh(torch.nn.Tanh):
  def reverse(self,input):
    # return torch.atanh(input)
    return torch.tanh(input)
