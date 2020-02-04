import gym
import collections
from tensorboardX import SummaryWriter
# *tensorboard --logdir runs

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODE = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    # * About collection.defaultdict(collections.Counter)
    '''
        >>> import collections
        >>> transits = collections.defaultdict(collections.Counter)
        >>> transits
        defaultdict(<class 'collections.Counter'>, {})
        >>> transits[0][0] += 1
        >>> transits
        defaultdict(<class 'collections.Counter'>, {0: Counter({0: 1})})
        >>> transits[0][0] += 1
        >>> transits
        defaultdict(<class 'collections.Counter'>, {0: Counter({0: 2})})
        >>> transits[0][1] += 1
        >>> transits[0][1] += 1
        >>> transits
        defaultdict(<class 'collections.Counter'>, {0: Counter({0: 2, 1: 2})})
    '''
    def play_n_random_steps(self, count):

        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1

            if is_done:
                self.state = self.env.reset()
            else:
                self.state = new_state

    def calc_action_value(self, state, action):
        '''
        >>> transits.values()
        dict_values([Counter({0: 2, 1: 2})])
        >>> transits[0].values()
        dict_values([2, 2])
        >>> sum(transits[0].values())
        4
        >>> transits[0].items()
        dict_items([(0, 2), (1, 2), (5, 2)])
        '''
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            # * tgt_state : new_state, count : number of transition to new_state
            reward = self.rewards[(state, action, tgt_state)]
            # * Get a expectation
            # * sum(probability * random variable)
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        # * env.observation_space.n : 16
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)


if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(logdir='runs', comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        # * Initialize for value iteration
        agent.play_n_random_steps(100)
        # * Set a value on each state
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODE):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODE
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated % .3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
