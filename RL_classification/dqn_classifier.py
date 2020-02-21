import collections
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import layer
import sl_train

GAMMA = 0.99
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_FRAME = 50000

REPLAY_SIZE = 500
REPLAY_START_SIZE = 500

data_dir = './Datasets'
composed = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=composed)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


subset_loader = sl_train.subset_loader

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done'])


class ClassfierEnv:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def sample(self):
        return np.random.randint(10, size=BATCH_SIZE)

    def step(self, action):
        # * new_state, reward, is_done, current_state = self.env.step(action)
        data_iter = iter(self.dataloader)
        current_state, tag = next(data_iter)
        reward = np.where(np.equal(action.cpu(), tag), 10, -1)
        return None, torch.tensor(reward), np.ones(len(action)), current_state
        

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size=4):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones = zip(*[self.buffer[idx] for idx in indices])
        # return states, actions, rewards, dones, next_states
        # return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)
        # return list(states), list(actions), list(rewards, dtype=np.float32), list(dones, dtype=np.uint8), list(next_states)
        sample_states = torch.empty(0)
        sample_actions = torch.empty(0, dtype=torch.int64)
        sample_rewards = torch.empty(0, dtype=torch.int64)
        sample_dones = torch.empty(0, dtype=torch.bool)

        for idx in range(batch_size):
            sample_states = torch.cat((sample_states, states[idx]), 0)
            sample_actions = torch.cat((sample_actions, actions[idx].cpu()), 0)
            sample_rewards = torch.cat((sample_rewards, rewards[idx]), 0)
            sample_dones = torch.cat((sample_dones, dones[idx]), 0)

        return sample_states, sample_actions, sample_rewards, sample_dones


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.total_reward = 0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.sample()
            action = torch.from_numpy(action)
        else:
            # state_a = np.array([self.state], copy=False)
            state_v = self.state.to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = act_v

        _, reward, is_done, current_state = self.env.step(action)
        self.state = current_state
        self.total_reward += torch.sum(reward)

        done_mask = torch.tensor(is_done, dtype=torch.bool)

        exp = Experience(self.state, action, reward, done_mask)
        self.exp_buffer.append(exp)
        
        done_reward = self.total_reward
        self._reset()

        return done_reward


def calc_loss(batch, net, tgt_net, device='cpu'):
    states, actions, rewards, dones = batch

    '''
    state_v = torch.tensor(np.array(states, copy=False)).to(device)
    action_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)
    '''
    state_v = states.to(device)
    action_v = actions.to(device)
    rewards_v = rewards.to(device)
    done_mask = dones.to(device)

    state_action_value = net(state_v).gather(1, action_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(state_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v

    return nn.MSELoss()(state_action_value, expected_state_action_values)


if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "CPU")
    print(f"Run with {device}")

    env = ClassfierEnv(subset_loader)
    net = layer.CNN(shape=[1, 28, 28], number_of_classes=10).to(device)    
    net.eval()

    tgt_net = layer.CNN(shape=[1, 28, 28], number_of_classes=10).to(device)
    tgt_net.load_state_dict(torch.load("./model/with_regulation/accuracy61.pth"))
    tgt_net.eval()

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_reward = None

    tgt_accuracy = 0.7
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards = reward.item()
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()

            if frame_idx % 1000 == 0:
                print(f'Frame : {frame_idx}, Total rewards : {total_rewards}, Epsilon : {epsilon:.1f}, speed : {speed:.1f}')                

            if best_reward is None or best_reward < total_rewards:
                torch.save(net.state_dict(), "best_%.0f.pth" % total_rewards)
                if best_reward is not None:
                    print(f'Total rewards : {total_rewards}, Epsilon : {epsilon:.1f}, speed : {speed:.1f}')                
                    print(f'{frame_idx} frames are done. best_reward : {best_reward}')
                    print(f'Best reward updated {best_reward} -> {total_rewards}')
                    
                    accuracy = sl_train.model_test("best_%.0f.pth" % total_rewards)
                    print("Accuracy : ", accuracy)

                    if accuracy > tgt_accuracy:
                        tgt_accuracy = accuracy
                        tgt_net.load_state_dict(net.state_dict())
                        print("SYNC")
                    
                best_reward = total_rewards

        # * Do not get a loss untill buffer get enough data
        if len(buffer) < REPLAY_START_SIZE:
            continue

        # * Above : push to buffer, Below : Pop from buffer
        # * 업데이트는 에피소드 종료 상관 없이 하고 있음
        
        '''
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
            print("SYNC")
            accuracy = sl_train.model_test("best_%.0f.pth" % total_rewards)
        '''

        net.train()
        optimizer.zero_grad()
        batch = buffer.sample(4)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
        net.eval()
