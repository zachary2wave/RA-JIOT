import torch
from torch import nn
weight = nn.Parameter(torch.FloatTensor(1, 10))
nn.init.xavier_uniform_(weight)
input  = torch.randn(size = (1,10))
output = nn.Softmax()(input*weight*1000)
output2 = torch.ceil(input*weight)
optimizer = torch.optim.Adam([weight], lr=1e-4)

optimizer.zero_grad()
output2[0][1].backward()
print(weight.grad)