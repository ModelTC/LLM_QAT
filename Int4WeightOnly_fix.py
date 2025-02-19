import torch
import torch.nn as nn
import json

def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x

@torch.no_grad()
def initialize_quant_params(weight_ln, group_size=128, qmin=0, qmax=15):
    weight = weight_ln.clone()
    weight = weight.reshape(-1, group_size)
    max_val = weight.amax(dim=-1, keepdim=True)
    min_val = weight.amin(dim=-1, keepdim=True)
    scales = (max_val - min_val).clamp(min=1e-5) / (qmax - qmin)
    zeros = clamp_ste((qmin - round_ste(min_val / scales)), qmin, qmax)
    return scales, zeros

def extract_quant_params(model, quant_params = {}, group_size=128, qmin=0, qmax=15, prefix=''):
    for name, module in model.named_children():
        full_name = f'{prefix}.{name}' if prefix else name
        if isinstance(module, nn.Linear):
            scales, zeros = initialize_quant_params(module.weight, group_size, qmin, qmax)
            
            quant_params[full_name] = {
                'scales': scales,
                'zeros': zeros
            }
        else:
            extract_quant_params(module, quant_params, group_size, qmin, qmax, full_name)
    
    return quant_params
    

class Int4WeightOnlyQuantizer(nn.Module):
    def __init__(self):
        '''
        w4a16g128 asymmetric
        '''
        super().__init__()
        self.group_size = 128
        self.qmax = 15
        self.qmin = 0
        self.register_buffer('scales', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zeros', torch.tensor([0], dtype=torch.int))
        

    def fake_quant(self, x):
        ori_shape = x.shape
        x = x.reshape(-1, self.group_size)
        x = clamp_ste(round_ste(x / self.scales) + self.zeros, self.qmin, self.qmax) # quant
        x = (x - self.zeros) * self.scales # dequant
        x = x.reshape(ori_shape)
        return x
    
    def set_quant_params(self, scales, zeros):
        self.scales = scales
        self.zeros = zeros

    def forward(self, x: torch.Tensor):
        return self.fake_quant(x)


class Int4WeightOnlyLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,        
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.quantizer = Int4WeightOnlyQuantizer()


    def forward(self, x):
        weight = self.quantizer(self.weight)
        return nn.functional.linear(x, weight, self.bias)
    
    @classmethod
    def new(cls, linear: nn.Linear):
        new_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype
        )
        new_linear.weight = linear.weight
        new_linear.bias = linear.bias
        return new_linear
        

def prepare(model, quant_params, prefix=''):
    for name, module in model.named_children():
        full_name = f'{prefix}.{name}' if prefix else name
        if isinstance(module, nn.Linear):
            if True: # add condition to check if the module is quantized
                new_m = Int4WeightOnlyLinear.new(module)
                scales = quant_params[full_name]['scales'].to(module.weight.device)
                zeros = quant_params[full_name]['zeros'].to(module.weight.device)
                new_m.quantizer.set_quant_params(scales, zeros)
                setattr(model, name, new_m)
        else:
            prepare(module, quant_params, full_name)
    return model

def build_model(model_path):
    model_config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
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

    seed_all(42)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    model = build_model('./yinnengzhong/models/v4-20250121-142416/checkpoint-2069')
    quant_params = extract_quant_params(model.model)
    torch.save(quant_params, "./models/v4-20250121-142416/checkpoint-2069/quant_params.pt")
    exit()
    x = torch.randn(1, 128)
    # print(x)
    # quantizer = Int4WeightOnlyQuantizer()
    # print(quantizer(x))

    # from quant import IntegerQuantizer
    # llmc_quantizer = IntegerQuantizer(4, False, 'per_group', group_size=128)
    # print(llmc_quantizer.fake_quant_weight_dynamic(x))

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


    input_size = 1024
    output_size = 1024
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
