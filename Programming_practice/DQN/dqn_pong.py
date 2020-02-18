from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # * Get a number of batch_size experience in a range of len(self.buffer)
        # * It does not replace
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        # * zip example '*' is used to unzip
        # * https://www.programiz.com/python-programming/methods/built-in/zip
        '''
            coordinate = ['x', 'y', 'z']
            value = [3, 4, 5]
            result = zip(coordinate, value)
            result_list = list(result)
            print(result_list)
            c, v =  zip(*result_list)
            print('c =', c)
            print('v =', v)

            -------------output-------------
            [('x', 3), ('y', 4), ('z', 5)]
            c = ('x', 'y', 'z')
            v = (3, 4, 5)
        '''
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    # * Decorator -> When play_step function is called, torch.no_grad is called too
    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # TODO: Find how q value is calculated
        # TODO: Find what is calculated in model
        # * When probability is lower than epsilon, select random action
        # * else select the highest probability
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            # * Put state_v(image) to model and get a distribution
            # TODO : Understand how to do backpropagate
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # * There is a return for information
        # * If put a tag in information, maybe classification is possible?
        '''
            Training phase : put a tag
            Estimating phase : Do not put a tag
        '''
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        # * state update
        # TODO: If next state is final state, what is next state
        # TODO: Find what is a next state of last state
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward


def calc_loss(batch, net, tgt_net, device='cpu'):
    states, actions, rewards, dones, next_states = batch

    state_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    action_v = torch.tensor(actions).to(device)
    # * rewards_v : result reward of current state
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    # * source.shape : [1,3]
    # * source.unsqueeze(-1) : Add dimension to end of source -> [1,3,1]
    # * source.unsqueeze(0) : Add dimension to the most front of sourece -> [1,1,3]
    # * source.squeeze(-1) : Delete last dimension. If cannot delete, don't delete -> [1,3]
    # * source.squeeze -> [3]
    # * input.gather(dim, index)
    # ! Source has to have same dimension with input
    # * e.g.
    '''
    z = torch.tensor([1,2,3])
    z.gather(0, torch.tensor([1,1,1])) -> tensor([1,1,1])
    z.gather(0, torch.tensor([1,2,3])) -> tensor([1,2,3])
    '''
    
    # * net(state_v) returns action of eacth batch -> shape is [batch, actions]
    # * net(state_v)[batch][action_index] = value
    # * Select value using action. it does not select value using result of CNN.
    state_action_value = net(state_v).gather(1, action_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        # * Classification
        # * Select a action using max
        # ! Here is the part that different with Supervised learning
        # * value, indices =  torch.max(...)
        next_state_values = tgt_net(next_states_v).max(1)[0]
        # * For example, When target state's value is 1, selected action's value is 1, then loss = 0 else loss is 1
        # * In short, if select the highest value, then loss is small
        # * This algorithm try to find the highest value!
        # TODO : NEED TO CHANGE `next_state_values[done_mask]` for my own experiement
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v

    # TODO : Is it going to best value?
    # * When target net's reward incresed, test net's reward has to increase to reduce loss
    # * Denote MSE(A - B) -> Get a assumtion that B is larger than A...?
    return nn.MSELoss()(state_action_value, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    # * tgt_net : Target network
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None
    
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d : done %d games, reward %.3f, eps %.2f, speed %.2f f/s" % (frame_idx, len(total_rewards), m_reward, epsilon, speed))

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env + "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (best_m_reward, m_reward))
                best_m_reward = m_reward

                if m_reward > MEAN_REWARD_BOUND:
                    print("Solved in %d frames!" % frame_idx)
                    break

        # * Do not get a loss untill buffer get enough data
        if len(buffer) < REPLAY_START_SIZE:
            continue

        # * Above : push to buffer, Below : Pop from buffer
        # * 업데이트는 에피소드 종료 상관 없이 하고 있음        
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
