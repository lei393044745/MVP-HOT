import torch.nn as nn
import torch
import math
import torch.functional as F
from timm.models.layers import trunc_normal_

def init_rate(tensor):
    if tensor is not None:
        tensor.data.fill_(1)

class Prompt_all(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.prompt = nn.ModuleList()
        for _ in range(depth):
            self.prompt.append(Prompt())
            
    def forward(self, xz, hsi_xz,idx):
        return self.prompt[idx](xz, hsi_xz)

class Prompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Promt_block()
        self.block2 = Promt_block()
        self.block3 = Promt_block()
        
    def forward(self, xz, hsi_xz):
        x1, z1, z2 = torch.split(xz, split_size_or_sections=[256,256,256], dim=1)
        hsi_x1, hsi_z1, hsi_z2 = torch.split(hsi_xz, split_size_or_sections=[256,256,256], dim=1)
        
        x1 = self.block1(x1, hsi_x1)
        z1 = self.block2(z1, hsi_z1)
        z2 = self.block3(z2, hsi_z2)
        
        return torch.cat([x1,z1,z2], dim=1)

class Promt_block(nn.Module):
    def __init__(self, in_dim = 768, hid_dim = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.conv1 = nn.Conv2d(in_dim, hid_dim, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_dim, hid_dim, kernel_size=1, padding=0, stride=1)
        self.conv3 = nn.Conv2d(hid_dim, in_dim, kernel_size=1, padding=0, stride=1)
        self.softmax = nn.Softmax(dim=-1)
        self.init_param()
        
    def init_param(self):
        for n in self.modules():
            if isinstance(n, nn.Conv2d):
                nn.init.xavier_uniform_(n.weight)
                nn.init.zeros_(n.bias)
            elif isinstance(n, nn.LayerNorm):
                nn.init.constant_(n.bias, 0)
                nn.init.constant_(n.weight, 1.0)
            elif isinstance(n, nn.Linear):
                trunc_normal_(n.weight, std=0.2)
                if isinstance(n, nn.Linear) and n.bias is not None:
                    nn.init.constant_(n.bias, 0)
    
    def forward(self, x, z):
        H = W = 16
        B, N, C = x.shape
        x = self.norm1(x)
        z = self.norm2(z)
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        z = z.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        
        x_, z_ = self.conv1(x), self.conv2(z)
        
        weight = self.softmax((x_ * z_).view(B, -1, H * W)).view(B, -1, H, W)
        
        out = self.conv3(weight * z_).view(B, C, -1).permute(0, 2, 1).contiguous()
        
        return out