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

## On-policy vs Off-policy

- Off-policy methods allow you to train on the previous large history of data or even on human demonstrations, but they usually are slower to converge.
- On-policy methods are typically faster, but require much more fresh data from the environment, which can be costly.

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

## Double DQN

- In the paper, ***the authors demonstrated that the basic DQN tends to overestimate values for Q,*** which may be harmful to training performance and sometimes can lead to suboptimal policies.
  - As a solution to this problem, the authors proposed modifying the Bellman update a bit

Basic DQN
$$
Q(s_t, a_t) = r_t + \gamma \max_aQ'(s_{t+1}, a)
$$

- $$Q'(s_{t+1}, a)$$ was Q-values calculated using our target network, so we update with the trained network every n steps.
- ***The authors of the paper proposed choosing actions for the next state using the trained network,*** but taking values of Q from the target network. So, the new expression for target Q-values will look like this:

$$
Q(s_t, a_t) = r_t + \gamma \max_a Q'(s_{t+1}, \underset{a}{argmax}Q(s_{t+1}, a))
$$

- 만약 argmax가 2개 이상 일때 max함수로 다음 state에서 맥스가 되는 걸 선택  함
- 트레이닝 할 네트워크에서 선택 가능한 action만 고려 함.
  - 고려 할 action중에서 가장 큰 action을 선택함
  - Basic DQN에서는 타겟 네트워크에서 선택 가능한 모든 action을 고려함
    - 하지만 Double DQN에서는 트레이닝 할 네트워크가 선택 가능한 action만 고려함
- Q' : Result of target network
- Q : Result of (being) trained network
- It is called Double DQN

## Noisy networks

***Noisy Networks for Exploration*** : It has a very simple idea for learning exploration characteristics during training instead of having a separate schedule related to exploration.

- In the Noisy Networks paper, the authors proposed a quite simple solution that, nevertheless, works well.
- They add noise to the weights of fully connected layers of the network and adjust the parameters of this noise during training using backpropagation.

The authors proposed two ways of adding the noise

- Independent Gaussian noise
  - For every weight in a fully connected layer, we have a random value that we draw from the normal distribution.
  - Parameters of the noise, $$\mu$$ and $$\sigma$$, are stored inside the layer and get trained using back propagation in the same way that we train weights of the standard linear layer.
  - The output of such a "noisy layer" is calculated in the same way as in a linear layer
- Factorized Gaussian noise
  - To minimize the number of random values to be sampled, ***the authors proposed keeping only two random vectors:*** 
    - One with the ***size of the input*** and
    - Another with the ***size of the output of the layer.***
  - Then, a random matrix for the layer is created by calculating the outer product of the vectors

## Prioritized replay buffer

This method tries to improve the efficiency of samples in the replay buffer by prioritizing those samples according to the training loss.

- The basic DQN used the replay buffer to break the correlation between immediate transitions in our episodes.
  - 학습 할 때 비선형성이 있어야 하는게 좋은 것 생각
  - Stochastic Gradient Descent(SGD) method assumes that the data we use for training has an i.i.d. property.
  - To solve this problem, the classic DQN method uses a large buffer of transitions, randomly sampled to get the next training batch.
- The authors of the paper questioned this uniform random sample policy and proved that by ***assigning priorities to buffer samples, according to training loss and sampling the buffer proportional to those priorities,*** we can significantly improve convergence and the policy quality of the DQN.
  - 트레이닝 로스와 샘플링 비율에 따라 버퍼 샘플에 우선순위를 지정 함
  - The tricky point here is to keep the balance of training on an "unusual" sample and training on the rest of the buffer.
  - If we focus only on a small subset of the buffer, ***we can lose our i.i.d. property and simply overfit on this subset***

$$
P(i)=\frac{p^\alpha_i}{\sum_k p^{\alpha '}_k}
$$

- $$p_i$$ is the priority of the i th sample in the buffer
- $$\alpha$$ is the number that shows how much emphasis we give to the priority.
  - If $$\alpha = 0$$, our sampling will become uniform as the classic DQN method.
  - Larger values for $$\alpha$$ put more stress on samples with higher priority.
  - It is another hyperparameter to tune. This paper proposed $$\alpha$$ as 0.6
- ***New samples added to the buffer need to be assigned a maximum value of priority*** to be sure that they will be sampled soon.

### Implementation

- We need a new replay buffer that will track priorities, sample a batch according to them, calculated weights, and let us update priorities after the loss has become known
- The second change will be the loss function itself.
  - Now we not only need to incorporate weights for every sample, ***but we need to pass loss values back to the replay buffer to adjust the priorities of the sampled transitions.***

## Dueling DQN

The core observation of this paper is that the Q-values, $$Q(s, a)$$, that our network is trying to approximate can be divided into quantities:

- The value of the state, $$V(s)$$
  - Same with what we have known
- The ***advantage of actions in this state,*** $$A(s, a)$$.
  - The advantage $$A(s, a)$$ is supposed to bridge the gap from $$A(s)$$ to $$Q(s, a)$$, as, by definition, $$Q(s, a) = V(s) + A(s, a)$$.
    - In other words, the advantage $$A(s, a)$$ is just the delta, saying ***how much extra reward some particular action from the state brings us.***
    - The advantage could be positive or negative and, in general, can have any magnitude.
    - For example, at some tipping point, ***the choice of one action over another can cost us a lot of the total reward.***
- The Dueling paper's ***contribution was an explicit separation of the value and the advantage in the network's architecture, which brought better training stability, faster convergence, and better results.***

### Architecture

The architecture is difference from the classic DQN network.

- The classic DQN network takes features from the convolution layer and, using fully connected layers, transforms them into a vector of Q-values, one for each action.
- On the other hand, dueling DQN takes convolution features and processes them using two independent paths:
  - One path is responsible for V(s) prediction, which is just a single number
  - And another path predicts individual advantage values, having the same dimension as Q-values in the classic case.
  - After that, we add V(s) to every value of A(s, a) to obtain Q(s, a), which is used and trained as normal.
- Constraint to set: we want the mean value of the advantage of any state to be zero. (e.g. A(s) = [-2, -1, 1, 2])
  - This constraint could be enforced in various ways.
  - In the Dueling paper, the authors proposed a very elegant solution of ***subtracting the mean value of the advantage from the Q expression*** in the network, which effectively pulls the mean for the advantage to zero:

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{N}\sum_k A(s,k)
$$

## Categorical DQN

In the paper, the authors questioned the fundamental piece of Q-learning—Q-values—and tried to replace them with a more generic Q-value probability distribution.

- Why do we limit ourselves by trying to predict an average value for an action, when the underlying value may have a complicated underlying distribution? Maybe it will help us to work with distributions directly.
- ***Overall idea is to predict the distribution of value for every action,*** similar to the distributions for our car/train example. 
- The resulting distribution can be used to train our network to give better predictions of value distribution for every action of the given state, exactly in the same way as with Q-learning
- The only difference will be in the loss function, which now ***has to be replaced with something suitable for distribution comparison***

## Combining everything: Rainbow

It converges faster