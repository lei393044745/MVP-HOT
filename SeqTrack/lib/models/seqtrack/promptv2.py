import torch.nn as nn
from torch import einsum
import torch
import math
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange

class Prompt_all(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.prompt = nn.ModuleList()
        for _ in range(depth):
            self.prompt.append(Prompt())
            
    def forward(self, xz, hsi_xz,idx):
        return self.prompt[idx](xz, hsi_xz)
    

# class Prompt(nn.Module):
#     def __init__(self):
#         super().__init__()
#         in_dim = 768
#         hid_dim = 768 // 16
#         self.norm1 = nn.LayerNorm(768)
#         self.norm2 = nn.LayerNorm(768)
#         self.linear1 = nn.Linear(in_dim, hid_dim, bias=True)
#         self.linear2 = nn.Linear(in_dim, hid_dim, bias=True)
#         self.linear3 = nn.Linear(hid_dim, in_dim, bias=True)
#         self.softmax = nn.Softmax(dim = -1)
#         self.smooth = nn.Parameter(torch.zeros(1) + 10.0)
#         self.init_param()
        
#     def init_param(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 trunc_normal_(m.weight, std=.02)
#                 if isinstance(m, nn.Linear) and m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)

#     def forward(self, x, z):
#         B, N, C = x.shape
#         x, z = self.norm1(x), self.norm2(z)
#         x, z = self.linear1(x).permute(0, 2, 1), self.linear2(z)
        
#         out = (((self.smooth * self.softmax(x)) * x)).permute(0, 2, 1)
#         out = out + z
#         out = self.linear3(out)
#         return out


class Prompt(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 768
        hid_dim = 768 // 16
        self.linear1 = nn.Linear(in_dim, hid_dim, bias=True)
        self.linear2 = nn.Linear(in_dim, hid_dim, bias=True)
        self.linear3 = nn.Linear(hid_dim, in_dim, bias=True)
        self.softmax = nn.Softmax(dim = -1)
        self.smooth = nn.Parameter(torch.zeros(1) + 10.0)
        self.init_param()
        
    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, z):
        B, N, C = x.shape
        x, z = self.linear1(x).permute(0, 2, 1), self.linear2(z)
        
        out = (((self.smooth * self.softmax(x)) * x)).permute(0, 2, 1)
        out = out + z
        out = self.linear3(out)
        return out
