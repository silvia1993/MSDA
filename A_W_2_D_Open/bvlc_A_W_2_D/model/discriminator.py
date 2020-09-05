import torch.nn as nn
import math

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv6_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
  
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6_2 = nn.ReLU(inplace=True)

        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(512, 512, kernel_size=3)
        self.relu7_2 = nn.ReLU(inplace=True)

        self.fc8 = nn.Linear(512, 256)
        self.relu8 = nn.ReLU(inplace=True)
        self.drop8 = nn.Dropout()

        self.fc9 = nn.Linear(256, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):
        conv6_1 = self.conv6_1(input)
        relu6_1 = self.relu6_1(conv6_1)
        conv6_2 = self.conv6_2(relu6_1)
        relu6_2 = self.relu6_2(conv6_2)
        pool6 = self.pool6(relu6_2)

        conv7_1 = self.conv7_1(pool6)
        relu7_1 = self.relu7_1(conv7_1)
        conv7_2 = self.conv7_2(relu7_1)
        relu7_2 = self.relu7_2(conv7_2)

        relu7_2 = relu7_2.view(relu7_2.size(0), -1)
        fc8 = self.fc8(relu7_2)
        relu8 = self.relu8(fc8)
        drop8 = self.drop8(relu8)

        fc9 = self.fc9(drop8)
        return fc9
