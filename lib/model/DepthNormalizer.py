import torch
import torch.nn as nn
import torch.nn.functional as F 

class DepthNormalizer(nn.Module):
    def __init__(self, opt):
        super(DepthNormalizer, self).__init__()
        self.opt = opt

    def forward(self, xyz, calibs=None, index_feat=None):
        '''
        normalize depth value
        args:
            xyz: [B, 3, N] depth value
        '''
        z_feat = xyz[:, 2:3, :] * (self.opt.loadSize // 2) / self.opt.z_size
        return z_feat

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        # Other layers of your generator can be defined here...

        # Add the DepthNormalizer module
        self.depth_normalizer = DepthNormalizer(opt)

    def forward(self, input):
        # Your generator forward pass logic goes here...
        # For example:
        # x = self.conv1(input)
        # x = self.conv2(x)
        # ...

        # Apply the DepthNormalizer to the output of your generator
        normalized_depth = self.depth_normalizer(x)
        return x, normalized_depth

# Example usage:
opt = {'loadSize': 256, 'z_size': 100}  # Replace with appropriate values
generator = Generator(opt)
input_xyz = torch.randn(2, 3, 100)  # Example input depth values (batch size 2, 100 points)
output, normalized_depth = generator(input_xyz)
print(normalized_depth)
