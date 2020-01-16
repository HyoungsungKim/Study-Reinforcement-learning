# Week 6

## Measuring exploration 

### Previous - Family of value based method

- Value-based methods
  - We know Q(s,a) -> we know optimal policy
- Model free setting : Don't have p(s', r | s, a)
  - Q-learning, SARSA, Expected SARSA
- Large state spaces
  - Approximate Q-learning, DQN, Double DQN, Etc

Problem : They only learn by training actions, and seeing which of them work better

### Today - Exploration

- Balance between using what you learned & trying to find something even better

### A simplified MDP with only one step

- It is known as multi-armed bandit
- A simplified MDP with only one step
  - Observation -> Agent - > action -> feedback
- Why only one step?
  - It is simpler to explain exploration methods, formula are shorter
  - We can generalize MDP if we wish

## Regret: measuring the quality of exploration

### How to measure exploration

- There are two ways to learn optimal ad placement
  - Method A vs Method B
  - How do you choose one?
  - Which one is better?
    - Bad idea : Choose just what we want
    - Good idea : With money it brought/lost you
- In theoretical field, this bring you to a notion of regret
  - Regret is basically how much money your algorithm wasted!
- Regret of policy : $$\pi(a|s)$$:
  - Consider an optimal policy $$\pi^*(a|s)$$
  - $$\text{Regret per tick} = \text{optimal - yours}$$

$$
\eta = \sum_t[E_{s, a \text{~} \pi^*} r(s,a) - E_{s,a \text{~} \pi}r(s,a)]
$$

- 차이가 크면 regret도 커짐
- Regret : basically how much did you wasted

## The message just repeats. "Regret, regret, regret"

### What we already know

Stratifies

- $$\epsilon$$ - greedy

  - With probability $$\epsilon$$ take a uniformly random action
  - Otherwise take optimum

- Boltzman

  - Pick an action proportionally to transformed Q values
  - Q value is

  $$
  \pi(a|s) = softmax(\frac{Q(s,a)}{\tau})
  $$

- Policy based : add entropy

  - REINFORCEMENT
  - actor-critic
  - They all learn the probability of taking action explicitly

### Quiz time

- Your agent uses $$\epsilon$$-greedy strategy with s = 0.1
- What about his regret?

Regret
$$
\eta = \sum_t[E_{s,a \text{~}\pi^*}r(s,a) - E_{s,a \text{~} \pi}r(s,a)]
$$

- It grows linearly with time!
- 현재의 Policy가 optimal policy와 같더라고 epsilon만큼의 확률로 다른 action을 선택하기 때문에 regret의 합은 계속 증가함
- How do we fix it?

Solution

- Epsilon이 0이 될 때까지 조금씩 줄여가면서 exploration을 함
- 만약 epsilon을 바로 0으로 하면 greedy하지만 optimal한 policy는 아님

### Greedy in the limit

- Practical : Multiply epsilon by 0.99 every K steps