import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from pprint import pprint
import torch.nn.init as init

class NN(nn.Module):
    def __init__(self, num_actions, learning_rate, input_dims):
        super(NN, self).__init__()
        '''self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc4 = nn.Linear(20 * 20 * 64, 12000)
        self.fc5 = nn.Linear(12000, 512)
        self.fc6 = nn.Linear(512, num_actions)'''

        self.fc1 = nn.Linear(84*84, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

        '''init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc5.weight, nonlinearity='relu')

        init.zeros_(self.conv1.bias)
        init.zeros_(self.conv2.bias)
        init.zeros_(self.fc4.bias)
        init.zeros_(self.fc5.bias)

        init.xavier_normal_(self.fc6.weight)
        init.zeros_(self.fc6.bias)'''

        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)

        init.xavier_normal_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        '''
        x = torch.relu(self.conv1(x))

        x = torch.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc4(x))

        x = torch.relu(self.fc5(x))

        x = self.fc6(x)
        '''

        x = torch.relu(self.fc1(x))

        x = torch.relu(self.fc2(x))

        x = self.fc3(x)

        return x
