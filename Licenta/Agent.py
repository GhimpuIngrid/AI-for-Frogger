import torch
import torch.nn as nn
import torch.optim as optim
import NN  # Schimbat: Importul modulului NN care conține rețeaua neuronală actualizată
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, number_of_actions, input_dims, batch_size, learning_rate=0.02, gamma=0.99, initial_epsilon=1.0,
                 epsilon_decay=0.9999, min_epsilon=0.01, max_mem_size=20000):

        self.number_of_actions = number_of_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.eps = initial_epsilon
        self.eps_decay = epsilon_decay
        self.min_eps = min_epsilon

        self.input_dims = input_dims
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.nn = NN.NN(self.number_of_actions, self.lr, self.input_dims)
        self.target_nn = NN.NN(self.number_of_actions, self.lr, self.input_dims)
        self.update_target_nn()

        self.optimizer = optim.Adam(self.nn.parameters(), lr=learning_rate)
        # self.optimizer = optim.SGD(self.nn.parameters(), lr=learning_rate, momentum=0.9)

        self.state_memory = np.zeros((self.mem_size, 84*84), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 84*84), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

        self.counter = 0
        self.prev_weights = {name: param.clone().detach() for name, param in self.nn.named_parameters()}

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def update_target_nn(self, tau=0.1):
        for target_param, param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def choose_action(self, state):
        if np.random.rand() < self.eps:
            action = np.random.randint(self.number_of_actions)
            return action
        else:
            obs = state.to(self.nn.device)

            with torch.no_grad():
                q_values = self.nn.forward(obs)

            return torch.argmax(q_values).item()

    def choose_action_test(self, state):
        q_values = self.nn.forward(state)
        return torch.argmax(q_values).item()

    def prepare_input(self, frame):

        frame = Image.fromarray(frame).resize((84, 84))
        frame = frame.convert('L')

        transform = transforms.Compose([transforms.ToTensor()])
        frame = transform(frame)

        return frame.view(-1).unsqueeze(0)

    def train(self):

        if self.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.nn.device)

        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.nn.device)

        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.nn.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.nn.device)

        action_batch = self.action_memory[batch]

        q_values = self.nn.forward(state_batch)[batch_index, action_batch]

        next_actions = torch.argmax(self.nn.forward(new_state_batch), dim=1).unsqueeze(1)

        next_q_values = self.target_nn.forward(new_state_batch).gather(1, next_actions).squeeze(1).detach()

        target = reward_batch + self.gamma * next_q_values

        loss = self.nn.loss(target, q_values).to(self.nn.device)

        loss.backward()
        self.optimizer.step()

        self.eps = max(self.min_eps, self.eps * self.eps_decay)
        self.counter += 1

    def save_model(self):
        torch.save(self.nn.state_dict(), "agent.txt")

    def load_model(self):
        self.nn.load_state_dict(torch.load("agent.txt"))
        self.nn.eval()


def contains_different_tensors(t):
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if not torch.equal(t[i], t[j]):
                return True  # Sub-tensori diferiți găsiți
    return False
