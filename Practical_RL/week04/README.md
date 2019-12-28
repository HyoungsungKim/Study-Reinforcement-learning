# Week 04

The most practical widely-known application of reinforcement learning techniques nowadays, is playing video games. This is in part because video games are perfect small-sized and oversimplified reflections of real life, but also in part because current methods are not efficient in nature for applications to real life problems.

현재 강화 학습이 가장 널리 사용되고 있는 분야는 비디오 게임임. 비디오 게임은 현실 세계를 간단화 하여 담고 있다는 장점이 있지만, 실제 세상에 바로 적용 할 수없다는 단점이 존재함.

## Quickly reminder

### Tabular methods are limited

- In the tabular SARSA

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t + 1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$

- Any single cell update does not affect any other cell
  - Parameters가 업데이트 되어도 다른 값들은 바뀌지 않음
  - 오직 Q, S, A,에만 영향을 줌.

- Atari game에서 가능한 경우의 수(# of parameters)는 몇개일까?
  - 매우 많음(e.g. # of pixels)
- Limitation
  - \# of parameters > \# of states
    - \# : memory, data, time
- ***To learn such environments, we should know how to approximate.***
  - One straightforward ideas is to reduce number of parameters as much as possible.

### Function approximation in RL

- If we want to approximate value function or action value function using some parametric model $$\hat{v}$$ or $$\hat{q}$$, whose parameter w

$$
\hat{v}(s, w) \approx V_\pi(s) \\
\hat{q}(s,a,w) \approx q_\pi(s,a)
$$

- This is a straightforward idea that hypothetically, could allow us to learn in environments with infinite and combinatorial number of states.
  - How?
- This formulation is very similar with supervised learning
  - But how to transfer this supervised learning paradigm to reinforcement learning?
  - We don't have training data. because we don't have outcomes.
  - Develop two different approaches of application supervised learning to reinforcement learning. 
- In fact, in addition to dynamic programming, there are two core approaches in reinforcement learning, which we will make use of and which allow our agent to learn model free. 
  - Monte Carlo
  - Time difference learning

#### Reduction(환원) to Supervised Learning problem - Monte Carlo(MC)

$$
s \mapsto \mathbb{E}_\pi[G_t|S_t = s] \\
s, a \mapsto \mathbb{E}_\pi[G_t|S_t = s, A_t = a]
$$

- This is expectations are our goals in a perfect world.
- If we know this goal for every to state and action, we are done.
  - We could use any of the supervised learning methods to approximate these goals and once we build such approximations like SARSA, expected SARSA or Q learning.
- ***The most straightforward way to approximate an expectations, is to replace it with its sample-based estimate.***
  - Sample based estimates of goals $$g(s), g(s,a)$$
  - $$G_t$$ : total discounted reward

$$
s \mapsto R(s, \pi(s)) + \gamma G_{t+1} \\
s, a \mapsto R(s,a) + \gamma G_{t+1}
$$

- These numbers are known at the end of an episode
- so can we start learning process without worry? 
  - Well, both yes and no.
  - Yes, because Supervised Learning task is set up properly
  - And no, because it is not the best solution available.
  - ***For Monte Carlo approximations, we will need too many samples to learn*** in an environment which lasts more than several hundreds of time steps. 
  - ***Therefore, variance is very large***
  - Very slow
- Can we do better?

#### Temporal difference learning

$$
s \mapsto \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t = s] \\
s,a \mapsto \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}|S_t = s, A_t = a)]
$$

- In practice, Temporal difference methods are a lot more sample efficient than Monte Carlo methods.
- We want to approximate these expectations with it's sample based estimates. 
- Sample based estimates of goals $$g(s), g(s,a)$$

$$
s \mapsto R(s, \pi(s)) + \gamma \hat{v}_\pi(S_{t+1}, w) \\
s,a \mapsto R(s,a) + \gamma \hat{v}_\pi(S_{t+1}, w)
$$

- ***Goal is function of `w`***
- Temporal difference targets otherwise to learn as it plays a game, we don't need to wait until the very end of the episode to update our model. 
- They also have a very small variance

### Loss in the same as for a regression problem

- Many losses are available - MSE, MAE, Huber loss
  - MSE : Mean Square Error
  - MAE : Mean Absolute Error
- $$g(s,a)$$ : goal

$$
\mathcal{L}(w) = \frac{1}{2} \cdot \sum_{s,a} \rho_\pi (s, a)[g(s,a) - \hat{q}_\pi(s,a,w)]^2
$$

- $$\rho_\pi (s,a)$$ : Weights of importance
  - Fraction of time (<- Distribution in the limit)
    - policy encounters state `s`
    - and in that state makes action `a`
- $$\sum_{s,a}$$ : All possible states and actions 

### Recap: Online, offline, on-policy, off-policy

- Off-policy : more general, very difficult
  - ***Behaviour policy is different with target policy***
  - Behaviour policy - collects data(makes actions)
  - Target policy - subject to(~을 조건으로) evaluation & improvement
- On-policy : less general, easier
  - ***Behaviour policy equals target policy***
    - We are in control of acting in that environment, and in control of changing the policy.
- If algorithm can learn off-policy, it also can learn on policy, but not the other way around

> Usually but not always:
>
> - Online - update policy during the episode(e.g. TD)
> - Offline - update policy after an episode ends(e.g. MC)

## Loss functions in value based RL

 One of the most simple and most widespread method of minimizing the loss is by a means of gradient descent.
$$
\mathcal{L}(w) = \frac{1}{2} \cdot \sum_{s,a} \rho_\pi(s,a)[g(s,a) - \hat{q}_\pi(s,a,w)]^2 \\
\mathcal{L}_{s,a}(w) = [g(s,a) - \hat{q}_\pi(s,a,w)]^2
$$
Gradient descent(GD)
$$
w \leftarrow w - \alpha \nabla_w \mathcal{L}(w)
$$

- To update parameters `w` with gradients descent, we should differentiate the whole loss.
  - We actually cannot even compute the loss.
  - We only have only is sample-based estimate.

Stochastic gradient descent(SGD)

- On-policy : $$s, a \sim \rho_\pi$$
- Off-policy : $$s, a \sim \rho_b$$

$$
w \leftarrow w - \alpha\nabla_w \mathcal{L}_{s,a}(w)
$$

- We should differentiate them with respect to parameters w.
- We will not only make sure estimate of current state and action.
- Thus, we introduce a so-called semi grading methods which ***treat goal as fixed and this for any particular type of goal*** - semi gradient descent

$$
\nabla_w g(s,a) = 0
$$

- This assumption simplifies math a lot

$$
w \leftarrow w - \alpha \nabla_w \mathcal{L}_{s,a}(w) \\
w \leftarrow w + \alpha[g(s,a) - \hat{q}(s,a,w)]\nabla_w\hat{q}_\pi(s,a,w)
$$

- $$\mathcal{L}$$ 에서 $$\mathcal{L}_{s,a}$$로 바뀜
- $$\rho_\pi$$를 고려 할 필요가 없어짐(미분 쉬워짐)

### Semi-gradient update

- Treats goals $$g(s,a)$$ as fixed numbers
- Changes parameters to move estimates closer to targets
- ***Ignores effect update on the target***
- Is not a proper gradient
  - No SGD's theoretical convergence properties
  - Converges reliably tin most cases
  - More computationally efficient than true SGD
- Meaningful thing to do

Targets g(s,a) define what and how do we learn

- SARSA (On-policy TD control)

$$
g(s,a) = R(s,a) + \gamma \hat{q}_\pi(S_{t+1}, A_{t+1}, w)
$$

- Expected SARSA

$$
g(s,a) = R(s, a) + \gamma \sum_{a}\pi(a | S_{t+1})\hat{q}_\pi(S_{t+1}, a, w)
$$

- Q-learning (Off-policy TD control)

$$
g(s,a) = R(s,a) + \gamma \max_a \hat{q}_\pi(S_{t+1}, a, w)
$$

- In each case, $$g(s,a)$$ is random variable. because it depends on the next state $$S_{t+1}$$. and additionally, on $$a_{t+1}$$ in case of SARSA.
- And on $$A_{t+1}$$ in case of SARSA. These doubles stochastic of SARSA target is not good thing.
  - That is main reason for why using expected SARSA is preferable than SARSA

### Semi-gradient SARSA(on-policy TD control)

Tabular SARSA
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$

- SARSA algorithm is an example of application of Bellman expectation equation.

Approximate SARSA
$$
w \leftarrow w + \alpha[R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, w) - \hat{q}(S_t, A_t, w)]\nabla\hat{q}(S_t, A_t, w)
$$

### Semi-gradient expected SARSA (off-policy)

Tabular expected SARSA
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \mathbb{E}_\pi[Q(S_{t+1}, A_{t+1}) | S_{t+1}] - Q(S_t, A_t)] \\
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma\sum_a \pi(a|S_{t+1})Q(S_{t+1}, a) - Q(S_t, A_t)]
$$
Approximate expected SARSA
$$
w \leftarrow w + \alpha[R_{t+1} + \gamma \sum_a\pi(a|S_{t+1})\hat{q}(S_{t+1}, a, w) - \hat{q}(S_t, A_t, w)]\nabla\hat{q}(S_t, A_t, w)
$$

### Semi-gradient Q-learning (off-policy)

Tabular Q-learning
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
$$
Approximate Q-learning
$$
w \leftarrow w + \alpha[R_{t+1} + \gamma \max_a \hat{q}(S_{t+1}, a, w) - \hat{q}(S_t, A_t, w)]\nabla\hat{q}(S_t, A_t, w)
$$
