import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes=31):
        super(Classifier, self).__init__()

        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout()

        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout()
   
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, input):
        input = input.view(input.size(0), 256 * 6 * 6)
        fc6 = self.fc6(input)
        relu6 = self.relu6(fc6)
        drop6 = self.drop6(relu6)

        fc7 = self.fc7(drop6)
        relu7 = self.relu7(fc7)
        drop7 = self.drop7(relu7)

        fc8 = self.fc8(drop7)
        return fc8
