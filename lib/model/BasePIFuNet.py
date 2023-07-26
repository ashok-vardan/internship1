import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
class BasePIFuGAN(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()},
                 input_dim=3,
                 output_dim=1,
                 ):
        super(BasePIFuGAN, self).__init__()
        self.name = 'base'

        self.criteria = criteria

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.generator = Generator(input_dim, output_dim)
        self.discriminator = Discriminator(input_dim + output_dim)

    def forward(self, points, images, calibs, transforms=None, labels=None):
        '''
        args:
            points: [B, 3, N] 3d points in world space
            images: [B, C, H, W] input images
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            labels: [B, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction corresponding to the given points
        '''
        # Generate predictions using the generator
        self.preds = self.generator(points.view(points.size(0), -1))

        # Compute adversarial loss using the discriminator
        if labels is not None:
            real_samples = torch.cat((points.view(points.size(0), -1), labels.view(labels.size(0), -1)), dim=1)
            fake_samples = torch.cat((points.view(points.size(0), -1), self.preds.view(self.preds.size(0), -1)), dim=1)
            real_labels = torch.ones(points.size(0), 1).to(points.device)
            fake_labels = torch.zeros(points.size(0), 1).to(points.device)

            real_prob = self.discriminator(real_samples)
            fake_prob = self.discriminator(fake_samples)

            self.adversarial_loss = F.binary_cross_entropy(fake_prob, real_labels) + F.binary_cross_entropy(real_prob, fake_labels)
        else:
            self.adversarial_loss = 0.0

        return self.preds

    def get_error(self, gamma=None):
        '''
        return the loss given the ground truth labels and prediction
        '''
        return self.criteria['occ'](self.preds, self.labels) + self.adversarial_loss

