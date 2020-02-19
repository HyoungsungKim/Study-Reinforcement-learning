# Ch8 DQN Extensions

DQN extensions

- N-step DQN
  - How to improve convergence speed and stability with a simple unrolling of the Bellman equation, and why it is not an ultimate solution
- Double DQN
  - How to deal with DQN overestimation of the values of the actions
- Noisy networks
  - How to make exploration more efficient by adding noise to the network weights
- Prioritized replay buffer
  - Why uniform sampling of our experience is not the best way to train
- Dueling DQN
  - How to improve convergence speed by making out network's architecture represent more closely the problem that we are solving
- Categorical DQN
  - How to go beyond the single expected value of the action and work with full distributions

## N-step DQN

(Remind)Bellman equation
$$
Q(s_t, a_t) = r_t + \gamma \max_{a} Q(s_{t+1}, a_{t+1})
$$
Recursive form
$$
Q(s_t, a_t) = r_t + \gamma \max_{a} [r_{a, t+1} + \gamma \max_{a'}Q(s_{t+2}, a')]
$$

- $$r_{a,t+1}$$ means local reward after issuing action $$a$$. 

$$
Q(s_t, a_t) = r_t + \gamma r_{t+1} + \gamma^2 \max_{a'}Q(s_{t+2},a')
$$

- When a is optimal or close to optimal
- ***Knowing steps a head is faster to converge***
  - Then if we know 100 steps, than it converges 100 times faster?
  - NO!
  - Because there is no guarantee that actions are optimal

## On-policy vs Off-policy

- Off-policy methods allow you to train on the previous large history of data or even on human demonstrations, but they usually are slower to converge.
- On-policy methods are typically faster, but require much more fresh data from the environment, which can be costly.