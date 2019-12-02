# Chapter 6 Temporal-Difference Learning: part 2

## Importance sampling from week 2

You will be able to use importance to estimate the expected value of a target distribution using samples from a different distribution.

- 다른 distribution의 샘플을 이용하여 target distribution의 기댓값을 예측하는 방법
- 현재 하나의 distribution를 가지고 있고, 예상하고 싶은 기댓값의 distribution는 x값에 대한 값만 알고 있지 distribution는 알고 있지 않음
  - sample : $$x \text{ ~ } b$$ -> Sample x from distribution b
  - estimate : $$\mathbb{E}_\pi[X]$$ -> We want to estimate the expected value of X
    - but with respect to the target distribution Pi.
    - Sample은 b에서 뽑았는데 예측하고 싶은 distribution은 Pi
  - Because x is drawn from b, we cannot simply use the sample average to compute the expectation under Pi.
  - 샘플을 b에서 뽑았기 때문에 Pi distribution을 예측하는데 사용 할 수없음
  - Sample average will give us $$\mathbb{E}_b$$

Derivation of importance Sampling
$$
\begin{alignat}{2}
\mathbb{E}_\pi[X] & \doteq \sum_{x \in X} x\pi(x) \\
& = \sum_{x \in X} x\pi(x) \frac{b(x)}{b(x)} \text{(} b(x) \text{ is prob of observed outcome x under b)} \\
& = \sum_{x \in X} x \frac{\pi(x)}{b(x)}b(x) \\
& = \sum_{x \in X} x\rho(x)b(x)	\text{ (} \rho(x) \text{ is sampling ratio)} \\
& \text{Treat }x\rho(x) \text{as a new random variable, } b(x) \text{ is the probability of observing x. Then,} \\
& = \mathbb{E}_b[X\rho(X)]


\end{alignat}
$$


- $$\frac{\pi(x)}{b(x)}$$ is called the ***importance sampling ratio*** = $$\rho(x)$$

Then how do we use it to estimate?

- We use $$\mathbb{E}[X] \approx \frac{1}{n}\sum_{i=1}^n x_i$$
- $$x_i$$ is drawn from $$b$$ not $$\pi$$
- $$\rho(x) = \frac{\pi(x)}{b(x)}$$

$$
\begin{alignat}{2}
\mathbb{E}_b[X\rho(X)] & = \sum_{x \in X}x\rho(x)b(x) \\
& \approx \frac{1}{n}\sum_{i = 1}^n x_i\rho(x_i) \text{ (Note :  }x_i \sim b \text{)} \\
& \approx \mathbb{E}_\pi [X]
\end{alignat}
$$



## 6.4 Sarsa: On-policy TD Control

- $$V(s)$$ : State to state
- $$Q(s,a)$$ : State-action to state-action
- In Sarsa, the agent needs to know its next state action pair before updating its value estimates.
- Sarsa learns slowly for the first couple of episodes

In this section we present an on-policy TD control method.

***The first step is to learn an action-value function rather than a state-value function.***

- ***In particular, for an on-policy method we must estimate $$q_\pi(s, a)$$*** for the current behavior policy $$\pi$$ and for all states $$s$$ and actions $$a$$. This can be done using essentially the same TD method described above for learning $$v_\pi$$ . 

***Now we consider transitions from state–action pair to state–action pair***, and learn the values of state–action pairs. Formally these cases are identical: they are both Markov chains with a reward process. The theorems assuring the convergence of state values under TD(0) also apply to the corresponding algorithm for action values:
$$
Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$
This update is done after every transition from a nonterminal state $$S_t$$.

- If $$S_{t+1}$$ is terminal, then $$Q(S_{t+1} , A_{t+1})$$ is defined as zero. This rule uses every element of the quintuple of events, $$(S_t , A_t , R_{t+1} , S_{t+1} , A_{t+1} )$$, that make up a transition from one state–action pair to the next. This quintuple gives rise to the name ***Sarsa*** for the algorithm.
- It is straightforward to ***design an on-policy control algorithm based on the Sarsa prediction method***.

## 6.5 Q-learning: Off-policy TD Control

- Q-learning is a sample-based version of value iteration which iteratively applies the Bellman optimality equation.
  - Applying the Bellman's Optimality Equation strictly improves the value function, unless it is already optimal.
  - For the same reason, Q-learning also converges to the optimal value function as long as the aging continues to explore and samples all areas of the state action space.
  - ***Q-learning takes the max*** over next action values. So it only changes when the agent learns that one action is better than another.
  
- In contrast, Sarsa uses the estimate of the next action value in its target.

- Sarsa is a sample-based algorithm to solve the Bellman equation for action values, that each depend on a fixed policy.
  - How can we make Sarsa perform better?
  - A step size parameter value of 0.5 might be a bit high for this experiment
  - These large updates might be causing Sarsa trouble exploratory actions
  
- Agent estimates its value function according to expected returns under their target policy. They actually behave according to their behavior policy.
  
  - ***Behavior policy*** : the policy that controls current actions
    - What are the target in behavior policies?
    - Q-learning's target policy is always greedy with respect to its current values.(deterministic)
    - However behavior policy can be anything that continues to visit all state action pairs during learning(non-deterministic)
  - ***Target policy*** : the policy being evaluated and/or learned
  - ***When the target policy and behavior policy are the same, the agent is learning on-policy, otherwise, the agent is learning off-policy.***
  
- ***Sarsa : On-policy algorithm***
  
  - In Sarsa, the agent bootstraps off of the value of the action it is going to take next, which is sampled from its behavior policy.
    - ***On-policy -> use the current policy***
    - $$A_{t+1} \sim \pi$$
  - Q-learning instead, bootstraps off of the largest action value in its next state
    - ***Off-policy -> Requirement : target policy is different to the behavior policy.***
    - $$a' \sim \pi_* \neq \pi$$
    - Since Q-learning learns about the best action it could possibly take rather than the actions it actually takes, it is learning off-policy
    - 현재 action만 고려하는게 아니라 가능한 모든 action을 고려함
  - But if ***Q-learning learns off-policy***, why don't we see any important sampling ratios?
    - It is because the agent is estimating action values with ***unknown policy***
    - Since the agents target policies greedy, with respect to its action values, all non-maximum actions have probability 0
  
  $$
  \sum_{a'}\pi(a'|S_{t+1})Q(S_{t+1},a') = \mathbb{E}_\pi[G_{t+1}|S_{t+1}] = \underset{a'}{max} \text{ }Q(S_{t+1}, a')
  $$
  
  - Q-learning은 action-value function과 target-policy의 곱으로 주어진 policy에서 어떤 state의 값이든 계산 할 수 있음
  - Q-learning은 expected Sarsa의 special case

***One of the early breakthroughs in reinforcement learning*** was the development of an off-policy TD control algorithm known as Q-learning (Watkins, 1989), defined by
$$
Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \text{ } \underset{a}{max}  \text{ }  Q(S_{t+1}, a) - Q(S_t, A_t)]
$$
In this case, the learned action-value function, Q, directly approximates $$q_*$$, the optimal action-value function, independent of the policy being followed.

- This dramatically simplifies the analysis of the algorithm and enabled early convergence proofs.
- The policy still has an effect in that it determines which state–action pairs are visited and updated. ***However, all that is required for correct convergence is that all pairs continue to be updated***.
  - As we observed in Chapter 5, this is a minimal requirement in the sense that any method guaranteed to find optimal behavior in the general case must require it.
  - Under this assumption and a variant of the usual stochastic approximation conditions on the sequence of step-size parameters, Q has been shown to converge with probability 1 to $$q_*$$ .

## 6.6 Expected Sarsa

- Sarsa

$$
Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$



Consider the learning algorithm that is just like Q-learning except that instead of the maximum over next state–action pairs it uses the expected value, taking into account how likely each action is under the current policy. That is, consider the algorithm with the update rule

- Expected Sarsa

$$
\begin{alignat}{2}
Q(S_t, A_t) & \gets Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \mathbb{E}_\pi[Q(S_{t+1}, A_{t+1} | S_{t+1})] - Q(S_t, A_t)] \\
& \gets Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \sum_a \pi(a|S_{t+1})Q(S_{t+1}, a) - Q(S_t, A_t)]

\end{alignat}
$$

- That means that on every time step, the agent has to average the next state's action values according to how likely they are under the policy
- Expected Sarsa has more stable update target than Sarsa
  - Expected Sarsa update targets are exactly correct
  - And do not change their estimated values away from the true values
- In general, expected Sarsa update targets are much lower variance than Sarsa(But more expensive sampling)
- $$\pi(a|S_{t+1})$$ : 확률
- $$Q(S_{t+1}, a)$$ : 랜덤 변수
- This algorithm moves ***deterministically*** in the same direction as Sarsa moves in expectation, and accordingly it is called ***Expected Sarsa***.
- Expected Sarsa is more complex computationally than Sarsa but, in return, it eliminates the variance due to the random selection of $$A_{t+1}$$

In these cliff walking results Expected Sarsa was used on-policy, ***but in general it might use a policy different from the target policy $$\pi$$ to generate behavior, in which case it becomes an off-policy algorithm***.

- For example, ***suppose $$\pi$$ is the greedy policy while behavior is more exploratory***; then Expected Sarsa is exactly Q-learning.
  - In this sense Expected Sarsa subsumes and generalizes Q-learning while reliably improving over Sarsa. Except for the small additional computational cost, Expected Sarsa may completely dominate both of the other more-well-known TD control algorithms.
- Sarsa and Q-Learning both use the expectation over their target policies in their update targets. This allows them to learn off-policy without importance sampling

