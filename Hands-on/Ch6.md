# Ch6 Deep Q-Networks

We'll look at the application of Q-learning to so-called ***"grid world" environments,*** which is called ***tabular Q-learning,*** and then we'll discuss Q-learning in conjunction with neural networks.

## Tabular Q-learning

 If some state in the state space is not shown to us by the environment, why should we care about its value?

- 모든 state를 고려 할 필요 없음

As we take samples from the environment, it's generally a bad idea to just assign new values on top of existing values, as training can become unstable. What is usually done in practice is to update the *Q(s, a)* with approximations using a "blending" technique, which is just averaging between old and new values of Q using learning rate $$\alpha$$ with a value form 0 to 1
$$
Q_{s,a} \leftarrow (1 - \alpha)Q_{s,a} + \alpha(r + \gamma \max_{a' \in A} Q_{s', a'})
$$
This make Q converge smooth

### Final algorithm

1. Start with an empty table for Q(s,a)
2. Obtain (s, a, r, s') from the environment
3. Make a Bellman update: $$Q_{s,a} \leftarrow (1 - \alpha)Q_{s,a} + \alpha(r + \gamma \max_{a' \in A} Q_{s', a'})$$
4. Check convergence conditions. It not met, repeat from step 2.

This method is called ***tabular Q-learning***

## Deep Q-learning

Modification of Q-learning

1. Initialize Q(s,a) with some initial approximation
2. By interacting with the environment, obtain the tuple (s, a, r, s')
3. Calculate loss : $$\mathcal{L} = (Q_{s,a} - r)^2$$ if episode has ended  -> 에피소드가 끝나서 value == reward 일때
   - Or $$\mathcal{L} = (Q_{s,a} - (r + \gamma \max_{a' \in A}Q_{s', a'}))^2$$
4. Update Q(s,a) using the stochastic gradient descent(SGD) algorithm, by minimizing the loss with respec to tne model parameters
5. Repeat form step 2 until converged

It looks simple, However it will not work well.

### Interaction with the environment

We're in trouble when our approximation is not perfect .

### SGD Optimization

- One of the fundamental requirements for SGD optimization is that the training data is independent and identically distributed

### Correlation between steps

- Practical issue with the default training procedure is also related to the lack of i.i.d in our data
- When we perform an update of our network's parameters, to make *Q(s, a)* closer to the desired result, we indirectly can alter the value produced for *Q(s′, a′)* and other states nearby. This can make our training really unstable, like chasing our own tail: when we update *Q* for state *s*, then on subsequent states we discover that *Q(s′, a′)* becomes worse, but attempts to update it can spoil our *Q(s, a)* approximation, and so on.
  - To make training more stable, there is a trick, called ***target network***, when we keep a copy of our network and use it for the *Q(s′, a′)* value in the Bellman equation.
  - This network is synchronized with our main network only periodically, for example, once in *N* steps (where *N* is usually quite a large hyperparameter, such as 1k or 10k training iterations).

## The final form of DQN training

- epsilon-greedy, replay buffer and target network

Procedure

1. Initialize parameters for Q(s,a) and Qhat(s,a) with random weights, epsilon is 1.0 and empty replay buffer
2. With probability epsilon, select a random action a, otherwise a = argmax_a Q(s,a)
3. Execute action a in an emulator and observe reward r and the nest state s'
4. Store transition (s, a, r, s') in the replay buffer
5. Sample a random minibatch of transitions from the replay buffer
6. For every transition in the buffer, calculate target y = r if the episode has ended at this step or $$y = r + \gamma \max_{a' \in A} \hat{Q}_{s',a'}$$
7. Update Q(s,a) using SGD algorithm by minimizing the loss in respect to model parameters
8. Every N steps copy wrights from Q to Qhat_t
9. Repeat from step 2 until converged