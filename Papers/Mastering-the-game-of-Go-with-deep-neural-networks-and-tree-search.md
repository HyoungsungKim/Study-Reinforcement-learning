# Mastering the game of Go with deep neural networks and tree search

## Abstract

Here we introduce a new approach to computer Go that uses

- `value networks` to evaluate board positions
- `policy networks` to select moves. 
- These deep neural networks are trained by a novel combination of supervised learning from human expert games, and reinforcement learning from games of self-play.

We also introduce a new search algorithm that combines Monte Carlo simulation with value and policy networks.

## Introduction

체스나 바둑 같은 상대방의 모든 정보가 공개 된 경우에는 depth search로 optimal한 값을 찾을 수 있음

- $$b^d$$의 경우의 수를 가짐
- 체스 같은 경우 $$(b \approx 35, d \approx 80)$$
- 바둑은 $$(b \approx 250, d \approx 150)$$
- ***but the effective search space can be reduced by two general principles.***

First, the depth of the search may be reduced by position evaluation:

- Truncating the search tree at state `s` and replacing the subtree below `s` by an approximate value function $$v(s)\approx v^*(s)$$ that predicts the outcome from state `s`.
  - State tree를 state `s` 에서 자르고 하위 트리를 optimal하다고 추정한 value function으로 대체 함
  - 이 방법은 체스나 오셀로에서는 유효했지만, 바둑에서는 경우의 수가 너무 많아 불가능

Second, the breadth of the search may be reduced by sampling actions from a policy p(a|s) that is a probability distribution over possible moves `a` in position `s`.

- 주어진 state 가능한 action a의 distribution을 계산하면 breath search에서 경우의 수를 줄일 수 있음
- 이 방법으로 아마추어 레벨의 바둑 플레이가 가능 함

> `Rollouts` are Monte Carlo simulations, in which ***random search steps are performed without branching until a solution has been found or  a maximum depth is reached.***
>
> - Rollout이란 Monte carlo tree seach에서 가지치기 없이 정답을 찾거나 트리의 끝에 도달 할 때까지 탐색 하는 방법
>
> These random steps can be sampled from machine learned policies $$p(a|s)$$, which predict the probability of taking the move (applying the transformation) `a` in position `s`, and are trained to predict the winning move by using human games or self-play

Monte Carlo tree search (MCTS) uses Monte Carlo rollouts to estimate the value of each state in a search tree. 

- As more simulations are executed, the search tree grows larger and the relevant values become more accurate. 
- However, prior work has been limited to shallow policies or value functions based on a linear combination of input features.

We employ a similar architecture of visual domain for the game of Go. We pass in the board position as a 19 × 19 image and use convolutional layers to construct a representation of the position.

- ***We use these neural networks to reduce the effective depth and breadth of the search tree***:
  - Evaluating positions using a `value network`
  - And sampling actions using a `policy network`

1. We begin by training a supervised learning ***(SL) policy network $$p_\sigma$$***directly from expert human moves.
   - This provides fast, efficient learning updates with immediate feedback and high-quality gradients.
2. We also train a fast policy $$p_\pi$$ that can rapidly sample actions during rollouts.
   - We also trained a faster but less accurate rollout policy $$p_\pi(a|s)$$, using a linear softmax of small pattern features with weights π; this achieved an accuracy of 24.2%, using just 2 μs to select an action, rather than 3 ms for the policy network
   - Rollout policy는 SL policy network 보다 정확도는 떨어지지만 빠르게 action을 선택 함

1과 2에서 Rollout policy와 SL policy network를 훈련시켜서 뛰어난 실력을 가진 바둑 기사의 수를 추정 함

A fast rollout policy $$p_\pi$$ and supervised learning ***(SL) policy network $$p_\sigma$$*** are trained to predict human expert moves in a data set of positions.

3. Next, we train a reinforcement learning (RL) policy network $$p_\rho $$ that improves the SL policy network by optimizing the final outcome of games of self-play.
   - This adjusts the policy towards the correct goal of winning games, rather than maximizing predictive accuracy
4. Finally, we train a value network $$v_\theta$$ that predicts the winner of games played by the RL policy network against itself. Our program AlphaGo efficiently combines the policy and value networks with MCTS.

***A reinforcement learning (RL) policy network $$p_\rho$$ is initialized to the SL policy network,*** and is then improved by policy gradient learning to maximize the outcome (that is, winning more games) against previous versions of the policy network. Finally, a value network $$v_\theta $$ is trained by regression to predict the expected outcome

- SL과 RL policy network 존재
- policy network는 probability distribution을 추정, value network는 scalar를 추정 함.
- Given `s` -> get a distribution using p(a|s) -> select `a` -> get a value of next state(s')  using v(s')

## Supervised learning of policy networks

The policy network is trained on randomly sampled state-action pairs (s, a), using stochastic ***gradient ascent*** to
maximize the likelihood of the human move `a` selected in state `s`
$$
\Delta \sigma \propto \frac{\partial \log p_\sigma (a|s)}{\partial \sigma}
$$

## Reinforcement learning of policy networks

The second stage of the training pipeline aims at improving the policy network by policy gradient reinforcement learning (RL). ***The RL policy network $$p_\rho$$is identical in structure to the SL policy network,*** and its weights $$\rho$$ are initialized to the same values, $$\rho = \sigma$$ .

- RL 네트워크는 SL과 같은 구조, 같은 weights를 사용함
  - 훈련되지 않은 RL을 SL과 붙이면 RL이 SL 절대 못이길것 -> 따라서 SL과 같은 weights로 시작하는게 맞는 것 같음

We use a reward function r(s) that is zero for all non-terminal time steps t < T. The outcome $$z_t = ± r(s_T)$$ is the terminal reward at the end of the game from the perspective of the current player at time step t: +1 for winning and −1 for losing
$$
\Delta \rho \propto \frac{\partial \log p_\rho(a_t|s_t)}{\partial \rho}z_t
$$

- The RL policy network won more than 80% of games against the SL policy network.
- SL을 이용하여 RL을 훈련
  - 저번에 classification 실험 할 때 이런 방법 사용했었는데 SL 이상의 성능 얻기 힘들었었는데...
  - 실제로 rollout 해보면 SL loss나 RL loss 비슷하게 나옴
  - 만약 RL이 SL을 이길 확률이 1%만 높아져도 장기적으로 보면 승률차이가 매우 크게 벌어질것(casino vs gambler)

## Reinforcement learning of value networks

The final stage of the training pipeline ***focuses on position evaluation***, estimating a value function $$v^p(s)$$ that predicts the outcome from position `s` of games played by using policy `p` for both players
$$
v^p(s) = \mathbb{E}[z_t|s_t=s,a_{t...T} \sim P]
$$

- State와 Policy의 distribution으로 선택한 action a가 주어졌을 때 reward 함수 z의 평균값 -> value of state under policy p
- Ideally, we would like to know the optimal value function under perfect play $$v^*(s)$$ 
  - In practice, we instead estimate the value function $$v^{p_\rho}$$ for our strongest policy, using the RL policy network $$p_\rho $$. 
  - 이상적으로는 가장 optimal한 value를 찾고싶지만, 실제로는 RL로 얻은 policy의 value 값을 계산 함.
  -  We approximate the value function using a value network $$v_\theta(s)$$ with weights $$\theta$$, $$v_\theta(s) \approx v^{p_\rho}(s ) ≈ v^*(s )$$. 

$$
\Delta \theta \propto \frac{\partial v_\theta(s)}{\partial \theta}(z-v_\theta(s))
$$

When trained on the KGS data set in this way, ***the value network memorized the game outcomes rather than generalizing to new positions***, achieving a minimum MSE of 0.37 on the test set, compared to 0.19 on the training
set.

- Test set과 training set간의 차이가 큼 -> overfitting
- To mitigate this problem, we generated a new self-play data set consisting of ***30 million distinct positions, each sampled from a separate game***.
- Training on this data set led to MSEs of 0.226 and 0.234 on the training and test set respectively, indicating
  minimal overfitting.

## Searching with policy and value networks

The tree is traversed by simulation (that is, descending the tree in complete games without backup), starting from the root state.
$$
a_t = \underset{a}{argmax}(Q(s_t,a) + u(s_t, a))
$$

- $$N(s,a)$$ : visit count

$$
u(s,a) \propto \frac{P(s,a)}{1 + N(s,a)}
$$

- $$P(s,a)$$ v : Prior probability
- Decays with repeated visits to encourage exploration

***The leaf position $$s_L$$ is processed just once by the SL policy network $$p_\sigma $$***. The output probabilities are stored as prior probabilities P for each legal action `a`, $$P(s, a ) = p_\sigma(a |s)$$ .

The leaf node is evaluated in two very different ways:

- First, by the value network $$v_\theta(s_L)$$;
  - Leaf node에서 얻은 게임의 결과로 계산한 value
- Second, by the outcome $$z_L$$ of a random rollout played out until terminal step T using the fast rollout policy $$p_\pi$$
  - Rollout policy로 계산한 reward

These evaluations are combined, using a mixing parameter λ, into a leaf evaluation $$V(s_L)$$
$$
v(s_L) = (1-\lambda)v_\theta(s_L) + \lambda z_L
$$

- $$\lambda$$를 이용해 RL과 value network를 이용한 결과와 rollout policy로 얻은 결과의 가중치 결정
- $$\lambda$$가 0.5일 때 가장 좋은 퍼포먼스 모여줌

At the end of simulation, the action values and visit counts of all traversed edges are updated.
$$
\begin{align}
N(s,a) & = \sum^n_{i=1} 1(s,a,i) \\
Q(s,a) & = \sum ^n _{i=1} 1(s,a,i) V(s^i_L)
\end{align}
$$
where $$s_L^i$$ is the leaf node from the ith simulation, and $$l(s, a, i)$$ indicates whether an edge (s, a) was traversed during the ith simulation. ***Once the search is complete, the algorithm chooses the most visited move from the root position***.

***It is worth noting that the SL policy network $$ p_\sigma $$ performed better in AlphaGo than the stronger RL policy network $$p_\rho $$***, presumably because humans select a diverse beam of promising moves, whereas RL optimizes for the single best move.

