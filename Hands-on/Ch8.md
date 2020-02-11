# Ch8 DQN Extensions

DQN extensions

- N-step DQN
  - How to improve convergence speed and stability with a simple unrolling of the Bellman equation, and why it is not an ultimatie solution
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

## On-policy vs Off-policy

- Off-policy methods allow you to train on the previous large history of data or even on human demonstrations, but they usually are slower to converge.
- On-policy methods are typically faster, but require much more fresh data from the environment, which can be costly.