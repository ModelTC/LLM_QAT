import torch
import torch.nn as nn


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


class Int4WeightOnlyQuantizer(nn.Module):
    def __init__(self):
        '''
        w4a16g128 asymmetric
        '''
        self.group_size = 128
        self.qmax = 15
        self.qmin = 0
        super().__init__()

    def fake_quant(self, x):
        ori_shape = x.shape
        x = x.reshape(-1, self.group_size)
        max_val = x.amax(dim=-1, keepdim=True)
        min_val = x.amin(dim=-1, keepdim=True)
        scales = (max_val - min_val).clamp(min=1e-5) / (self.qmax - self.qmin)
        zeros = clamp_ste((self.qmin - round_ste(min_val / scales)), self.qmin, self.qmax)
        x = clamp_ste(round_ste(x / scales) + zeros, self.qmin, self.qmax) # quant
        x = (x - zeros) * scales # dequant
        x = x.reshape(ori_shape)
        return x

    def forward(self, x: torch.Tensor):
        return self.fake_quant(x)


class Int4WeightOnlyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.quantizer = Int4WeightOnlyQuantizer()

    def forward(self, x):
        weight = self.quantizer(self.weight)
        return nn.functional.linear(x, weight, self.bias)

    @classmethod
    def new(cls, linear):
        new_linear = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, linear.weight.dtype)
        new_linear.weight = linear.weight
        new_linear.bias = linear.bias
        return new_linear
        

def prepare(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if True: # add condition to check if the module is quantized
                setattr(model, name, Int4WeightOnlyLinear.new(module))
        else:
            prepare(module)
    return model


if __name__ == '__main__':
    import os
    import random
    import numpy as np

    def seed_all(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_all(1)
    
    
    x = torch.randn(1, 128)
    print(x)
    quantizer = Int4WeightOnlyQuantizer()
    print(quantizer(x))

    from quant import IntegerQuantizer
    llmc_quantizer = IntegerQuantizer(4, False, 'per_group', group_size=128)
    print(llmc_quantizer.fake_quant_weight_dynamic(x))

    class Network1(nn.Module):
        def __init__(self, input_size, output_size):
            super(Network1, self).__init__()
            self.linear = nn.Linear(input_size, output_size, bias=False).to(torch.bfloat16)

        def forward(self, x):
            return self.linear(x)


    class Network2(nn.Module):
        def __init__(self, input_size, output_size):
            super(Network2, self).__init__()
            self.linear = nn.Linear(input_size, output_size, bias=True).to(torch.bfloat16)

        def forward(self, x):
            return self.linear(x)


    input_size = 128
    output_size = 8
    input = torch.randn(1, input_size).to(torch.bfloat16)

    model1 = Network1(input_size, output_size)
    # model1 = Network2(input_size, output_size)

    weight_ori = model1.linear.weight.data.clone()
    print(model1)
    output1_ori = model1(input)
    
    model1 = prepare(model1)
    weight_quant = model1.linear.weight.data.clone()
    print(model1)
    output1_quant = model1(input)
    
    print("cosine similarity", nn.functional.cosine_similarity(weight_ori.to(torch.float64).reshape(1, -1), weight_quant.to(torch.float64).reshape(1, -1)))
    print("cosine similarity", nn.functional.cosine_similarity(output1_ori.to(torch.float64).reshape(1, -1), output1_quant.to(torch.float64).reshape(1, -1)))
    
    quantizer = Int4WeightOnlyQuantizer()
    print("cosine similarity", nn.functional.cosine_similarity(weight_ori.to(torch.float64).reshape(1, -1), quantizer(weight_ori).to(torch.float64).reshape(1, -1), dim=1))

    loss = output1_quant.sum()
    loss.backward()

    print("weight grad", model1.linear.weight.grad)
