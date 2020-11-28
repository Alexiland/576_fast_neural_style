from neural_style_quant.transformer_net_quant import TransformerNet
from thop import profile

import torch

model = TransformerNet().to("cuda")
input = torch.randn(1, 3, 32, 32).to("cuda")
qops, flops, macs, params = profile(model, inputs=(input, ))
print("qops =", qops)
print("flops =", flops)
print("macs =", macs)
print("params =", params)