# Week02

## Reward design

### Explaining goals to agent through reward

Reward hypothesis(R.Sutton)

- Goals and purposes can be thought of as the maximization of the expected value of the cumulative sum of a received scalar signal
- Cumulative reward is called ***return***

$$
G_t \triangleq R_t + R_{t+1} + R_{t+2} + ... + R_T
$$

- T : End of episode
- Return is random variable
- Infinite return for non optimal behavior is problem
  - Tasks who has infinite horizons are often called continuing task
- Positive feedback loop
  - There is no motivation to find better loop

### Reward discounting

- Get rid of infinite sum by discounting

$$
G_t \triangleq R_t + \gamma R_{t+1} + \gamma ^2 R_{t+2} + ... \\
 = R_t + \gamma G_{t+1}
$$

### Reward design - scaling, shaping

- What transformations do not change an optimal policy?
  - Reward scaling - division by nonzero constant
    - Maybe useful in practice for approximate methods
  - Reward shaping - we could add to all rewards $$R(s,a,s')$$ in MDP values of potential based shaping function $$F(s,a,s')$$ without changing an optimal policy

$$
F(s,a,s') = \gamma \Phi(s') - \Phi(s)
$$

- s : state
- s' : next state
- Intuition : When no discounting F adds as much as it subtracts from the total return

## Policy: evaluation & improvement

### Model-based setup with value-based approach

- How to find optimal policy?
  - Model & value based
- Model based setup : model of the world is known.
  - i.e. $$p(r, s' | s, a)$$for all r, s', s, a  is known
- Value based approach
  1. Build or estimate a value
  2. Extract a policy from the value

### Policy evaluation: motivation

If you can't measure it, you can't improve it.

- Policy evaluation is also called prediction problem:
  - Predict value function for a particular policy

Bellman expectation equation
$$
\begin{align}
v_\pi(s) & = \sum_a \pi (a|s)\sum_{r,s'}p(r,s'|s,a)[r + \gamma v_\pi(s')] \\
& = \mathbb{E}_\pi[R_t + \gamma v_\pi(S_{t+1})|S = s]
\end{align}
$$

### Policy improvement: an idea

- Once we know what is v(s) for a particular policy, we could improve it by acting greedily w.r.t v(s)

$$
\pi(s') \leftarrow \underset{x}{argmax} q_\pi(s,a) \\
q_\pi(s,a) = \sum_{r,s'}p(r,s'|s,a)[r+\gamma v_\pi(s')]
$$

- q function과 v function의 차이
  - q function은 a가 주어지고 v function은 안주어짐
  - 따라서 v에서는 a를 얻는 확률을 곱해줘야 함
- If $$ q_\pi(s, \pi'(s)) \ge v_\pi(s) $$ for all states
  - then $$v_\pi'(s) \ge v_\pi(s)$$
  - meaning that $$\pi' \ge \pi$$

Optimal value under policy with state s -> Bellman optimality equation(Value iteration)
$$
v_\pi'(s) = \max_a \sum_{r,s'}p(r, s' | s, a)[r + \gamma v_\pi(s')]
$$
Optimal policy with optimal q value
$$
\pi_*(s) \leftarrow \underset{a}{argmax } \text{ }q_*(s,a)
$$
Optimal policy with optimal value (Policy iteration)
$$
\pi_*(s) \leftarrow \underset{a}{argmax} \text{ } \sum_{r, s'} p(r, s' | s, a)[r + \gamma v_*(s')]
$$

## Policy and value iteration

### The idea of policy and value iterations

Generalized policy iteration

1. Evaluation given policy
2. Improve policy by acting greedily w.r.t. it's value function

Policy iteration(Action을 찾음)

1. Evaluate policy until convergence (with some tolerance)
2. Improve policy

Value iteration(Value를 찾음, improvement에서 v를 이용해 a 찾음 - argmax)

1. Evaluate policy only with single iteration
2. Improve policy

### Value iteration (VI) vs Policy iteration(PI)

- VI is faster per cycle - $$O(|A||S|^2)$$
- VI requires many cycles
- PI is slower per cycle - $$O(|A||S|^2 + |S|^3)$$
- PI requires few cycles
- PI가 한 싸이클 당 비용이 크지만 적은 횟수만에 수렴하는 경우가 큼
- 따라서 PI가 경험적으로 더 빠르게 수렴 함

No silver bullet -> experiment with # of steps spent in policy evaluation phase to find the best algorithm for the task at hand