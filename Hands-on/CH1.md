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