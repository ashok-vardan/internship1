import torch
import torch.nn as nn 
import torch.nn.functional as F 
from ..net_util import conv3x3

class ConvBlock(nn.Module):
    # ... (same as in the given code)

class HourGlass(nn.Module):
    # ... (same as in the given code)

class HGFilter(nn.Module):
    # ... (same as in the given code)

# Now let's create the generator network (based on the existing code)

class Generator(nn.Module):
    def __init__(self, stack, depth, in_ch, last_ch, norm='batch', down_type='conv64', use_sigmoid=True):
        super(Generator, self).__init__()
        self.hg_filter = HGFilter(stack, depth, in_ch, last_ch, norm, down_type, use_sigmoid)
        self.generator_layers = nn.Sequential(
            nn.Conv2d(last_ch, last_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh activation for the final output
        )

    def forward(self, x):
        outputs, _ = self.hg_filter(x)
        generated_output = self.generator_layers(outputs[-1])  # Use the output of the last stack
        return generated_output

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_filters=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Use Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.model(x)
class GAN(nn.Module):
    def __init__(self, stack, depth, in_ch, last_ch, norm='batch', down_type='conv64', use_sigmoid=True):
        super(GAN, self).__init__()
        self.generator = Generator(stack, depth, in_ch, last_ch, norm, down_type, use_sigmoid)
        self.discriminator = Discriminator(last_ch)

    def forward(self, x):
        generated_output = self.generator(x)
        discriminator_output = self.discriminator(generated_output)
        return generated_output, discriminator_output

