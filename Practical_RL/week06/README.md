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

## Intuitive exploration

- How many random actions does it take to exit in maze?
  - Only get reward at the end
  - Epsilon-greedy is bad not only theoritical way buy also practical way
    - 시간이 매우 오래 걸림
  - Q-learning algorithm will need a lot of repeated experience of getting reward
  - Epsilon-greedy는 불필요한 동작을 많이 해야 함.

There is a lot of more practical things that humans are much better at than reinforcement learning algorithm.

- What human can learn easily is not easy for machine
  - Human don't explore using epsilon greedy

## Thomson Sampling

### Uncertainty in rewards

- We want to try actions if we believe there is a chance they turn out optimal
  - 우리는 optimal한 가능성이 있는 action을 하고 싶음
  - Ideas : Let's model how certain we are that Q(s,a) is what we predicted
    - 우리가 추측한게 얼마나 잘 맞을까?
- Bayesian -> It means that the variance of plot doesn't represent the randomness in the action itself, but only our belief
- How to pick? -> Thomson sampling

### Thomson Sampling

- Policy
  - Sample once from each Q distribution
    - Assume that samples are normal distribution
  - Take argmax over samples
  - Which actions will be taken?

## Optimism in face of uncertainty

- Idea : Prioritize(우선 순위를 정하다) action outcome
  - More uncertain = better
  - Greater expected value = better
  - Math : Try until upper confidence bound is small enough
- Policy 
  - Compute 95% upper confidence bound for each action
  - Take action with hight confidence bound
  - What can we tune here to explore more/less?

## UCB-1

### Frequentist approach

- There is a number of inequalities that bound $$p(x > t) < \text{some probability}$$

  - Upper bound - 조금의 정보만 가지고 있음
  - E.g. Hoeffding inequality (arbitrary in [0, 1])

  $$
  p(x - Ex \ge t) \ge e^{-2nt^2}
  $$

  - The essence here is that the probability of your action value, q-value, being larger than the average q-value you've measured so far by more than t
  - And chevshev, etc...
  - Hoeffding inequality
    - 독립적인 Random variable의 합이 예상 값에서 일정량 벗어날 가능성에 대한 상한(Upper bound)을 제공

### UCB-1 for Bandits

- Take actions in proportion to

$$
\tilde{v_a} = v_a + \sqrt{\frac{2 log{N}}{n_a}}
$$

- N : Number of time-step so far
- n_a : times action 'a' is taken

## Bayesian UCB

The usual way

- Start from prior P(Q)
- Learn posterior P(Q|data)
- Take q-th percentile

### What models can learn that?

How do we learn these distribution?

- Approach 1 : learn parametric P(Q), e.g. normal
  - There is a limitation
  - It relies wholly on your choice of the distribution
  - Probability decays exponentially, the exponent of the square lower from the distance
  - It works well for some case, but others not
    - It might even you overestimate or underestimate some action you did
- Approach 2 : use Bayesian neural network
  - 단순한 normal distribution이 아니게 됨 
    - 복합적인 정규분포(구불 구불 거리는...)
  - Non-parametric Bayesian inference
  - Bayesian neural networks would be a mechanism by which you learn a natural model that predicts not just one value at its output, but a distribution. 

### Bayesian UCB

- UCB vs Bayesian UCB
  - Choose any distribution you want
  - Learn actual distribution, not upper bound
  - Only as good as your prior
- 

### Recap

Epsilon-greedy and Boltzmann never behave optimally

- Make them greedy in the limit

Using uncertainty

- Thompson sampling, UCB-1, Bayesian UCB