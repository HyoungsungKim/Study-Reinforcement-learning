#  CH1

## Action and Environment

Agent < - > Environment

- Agent actions
  - Action
    - Discrete action : Finite set of mutually exclusive
    - Continuous action :  Continuing action like a steering something
- Environment give reward and observations
  - Observation
    - Observation is another information from environment(Reward and observation)
    - Observation has relationship with upcoming rewards
    - ***Observation is different with state*** 
      - Observation is function of state(not necessarily)
      - State satisfy Markov property
      - [What is the difference between an observation and a state in reinforcement learning?](https://ai.stackexchange.com/questions/5970/what-is-the-difference-between-an-observation-and-a-state-in-reinforcement-learn)
      - For example, In stock market
        - State : Everything about environment like stock price, news, twitter trend, etc
        - Observation : Limited information about environment like stock price and news and so on

## Markov process(Markov chain)

- It is allowed to capture more dependencies
- Markov property implies ***stationarity***

### Markov reward process

Add reward to markov process

- Return

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3} + ... = \sum_{k = 0}^\infin \gamma ^k R_{t+k+1}
$$

- The discount factor let agent get foresightedness.
  - Gamma = 0 : short-sightedness
  - Gamma = 1 : Return goes infinite
  - Commonly used gamma is 0.9 ~ 0.99
  - Gamma can prevent positive feedback
  - As gamma higher, agent get foresightedness
  - However just using `return` is not practical

- Value of state

$$
V(s) = \mathbb{E}[G|S_t = s]
$$

- State가 주어졌을때 return의 기댓값
- G는 random variable
- ***Markov에서 A와 B가 있다면 A와 B는 state고 A와 B가 각각 가지는 선택의 distribution이 policy***

### Markorv decision process

Add set of action to markov reward process

- By Choosing action, the agent can affect the probabilities of target states
- MDP에서는 action들 중 하나를 선택 했을 때 이 선택이 확률에 영향을 줌

#### Policy

- Policy affect action
- ***Policy is defined as the probability of distribution for every possible state***
- Policy는 state에 따라 결정되는 확률 분포

$$
\pi(a|s) = P[A_t = a | S_t = s]
$$

### Recap : MP, MRP, MDP

- MP : Basic Markov process
  - No reward
  - Only exist basic distribution
- MRP : MP with reward
  - Each probability of distribution has reward
- MDP : MRP with action
  - There is action and action can affect to the probability
  - Reward matrix depends on action
  - Policy can affect action