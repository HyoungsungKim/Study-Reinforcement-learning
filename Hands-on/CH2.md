# CH2

Let's learn basic OpenAI Gym API

## Environment and Agent

- Agent : It is some piece of code, which implements some policy.
  - This policy must decide what action is needed at every time step, given out observation
- Environment : Some model of the world.

```python
class Environment:
    def __init__(self):
        self.steps_left = 10
```

- Initialize Environment

```python
def get_observation(self):
    return [<some values>]
```

- Return ***current environment's observation*** to the agent

```python
def get_actions(self):
    return [0, 1]
```

- It allows the agent to query the set of actions it can execute.
- In this example, There are only 2 actions that agent can do

```python
def is_done(self):
    return self.steps_left == 0
```

- Signals the end of the episode to the agent.

```python
def actions(self, action):
    if self.is_done():
        raise Exception("Game is over")
    self.steps_left = -1
    return some_reward
```

- Handle the agent's action
- ***Returns the reward for this action.***

```python
class Agent:
    def __init__(self):
        self.total_reward = 0.0
```

- Initialize the counter that will keep the total reward accumulated by the agent during the episode

```python
def step(self, env):
    # 1
    current_obs = env.get_observation()
    # 2
    actions = env.get_actions()
    # 3
    reward = env.action(random.choice(actions))
    # 4
    self.total_reward += reward
```

1. Observe the environment
2. Make a decision about the action to take based on the observations
3. Submit the action to the environment
4. Get the reward for the current step

```python
if __name__ == "__main__":
    env = Environment()
    agent = Agent()
    
    while not env.is_done():
        agent.step(env)
        
    print("Total reward got: %.4f" % agent.total_reward)
```

- Create both classes and run one episode

## OpenAI Gym API

- `step` : Execute action
- `reset` : Return to initial state

### The environment

The environment is represented in Gym by the `Env` class, which has the following members

- `action_space` : Field of the `Space` class
- `observation_space` : Field of the same `Space` class
- `reset()`
- `step()`

Communications with the environment are performed via two methods of the `Env` class:`step` and `reset`

- Note that you have to call `reset()` after the creation of the environment.

### Wrappers

There are many situations that have the same structure: you'd like to "wrap" the existing environment and add some extra logic doing something. 

- Gym provides you with a convenient framework for these situations, called the `Wrapper` class. 

