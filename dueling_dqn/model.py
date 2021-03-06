from os import stat
import torch
import torch.nn as nn
import torch.nn.functional as F

# the convolution layer of deepmind
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, stride=2)
        self.conv2 = nn.Conv2d(4, 8, 3, stride=1)
        self.conv3 = nn.Conv2d(8, 16, 3, stride=1)
        
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 16 * 4 * 4)

        return x

# in the initial, just the nature CNN
class net(nn.Module):
    def __init__(self, num_actions):
        super(net, self).__init__()
        # define the network
        self.cnn_layer = deepmind()
        # the layer for dueling network architecture
        self.action_fc = nn.Linear(16 * 4 * 4, 256)
        self.state_value_fc = nn.Linear(16 * 4 * 4, 256)
        self.action_value = nn.Linear(256, num_actions)
        self.state_value = nn.Linear(256, 1)

    def forward(self, inputs):
        x = self.cnn_layer(inputs)
        # get the action value
        action_fc = F.relu(self.action_fc(x))
        action_value = self.action_value(action_fc)
        # get the state value
        state_value_fc = F.relu(self.state_value_fc(x))
        state_value = self.state_value(state_value_fc)
        # action value mean
        action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
        action_value_center = action_value - action_value_mean
        # Q = V + A
        action_value_out = state_value + action_value_center
        return action_value_out

class simplenet(nn.Module):
    def __init__(self, num_actions):
        super(simplenet, self).__init__()
        self.linear1 = nn.Linear(2+num_actions, 2+num_actions)
        self.state_value_fc = nn.Linear(2+num_actions, 2+num_actions)
        self.action_fc = nn.Linear(2+num_actions, 2+num_actions)
        self.action_value = nn.Linear(2+num_actions, num_actions)
        self.state_value = nn.Linear(2+num_actions, 1)
        nn.init.normal_(self.linear1.weight, mean=0., std=0.1)
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.normal_(self.state_value_fc.weight, mean=0., std=0.1)
        nn.init.constant_(self.state_value_fc.bias, 0.)
        nn.init.normal_(self.action_fc.weight, mean=0., std=0.1)
        nn.init.constant_(self.action_fc.bias, 0.)
        nn.init.normal_(self.action_value.weight, mean=0., std=0.1)
        nn.init.constant_(self.action_value.bias, 0.)
        nn.init.normal_(self.state_value.weight, mean=0., std=0.1)
        nn.init.constant_(self.state_value.bias, 0.)
    
    def forward(self, input):
        x = self.linear1(input)
        action_fc = F.relu(self.action_fc(x))
        action_value = self.action_value(action_fc)
        # get the state value
        state_value_fc = F.relu(self.state_value_fc(x))
        state_value = self.state_value(state_value_fc)
        # action value mean
        action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
        action_value_center = action_value - action_value_mean
        # Q = V + A
        action_value_out = state_value + action_value_center
        return action_value_out
