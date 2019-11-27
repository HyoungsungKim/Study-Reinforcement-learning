# Chapter 5 Monte Carlo Methods

***Unlike the previous chapter, here we do not assume complete knowledge of the environment***.

- Monte Carlo methods require only experience—sample sequences of states, actions, and rewards from actual or simulated interaction with an environment.
  - 몬톄 카를로 방법에서는 오직 경험만이 필수 
- Learning from actual experience is striking because it requires no prior knowledge of the environment’s dynamics, yet can still attain optimal behavior.
- Learning from simulated experience is also powerful.
  - Although a model is required, the model need only generate sample transitions, not the complete probability distributions of all possible transitions that is required for dynamic programming (DP).
  - 시뮬레이션을 사용 하려면 모델을 알아야 함
  - 완전한 확률 분포를 알 필요는 없음
  - 모델을 알아야 DP가 가능
  - In surprisingly many cases it is easy to generate experience sampled according to the desired probability distributions, but infeasible(불가능한) to obtain the distributions in explicit form.

Monte Carlo methods are ways of solving the reinforcement learning problem based on averaging sample returns.

- To ensure that well-defined returns are available, here ***we define Monte Carlo methods only for episodic tasks.***
- 몬테카를로 방법은 episodic 한 경우에만 사용 함.
- Only on the completion of an episode are value estimates and policies changed. Monte Carlo methods can thus be incremental in an episode-by-episode sense, but not in a step-by-step (online) sense.

Monte Carlo methods sample and average returns for each ***state–action pair*** much like the bandit methods we explored in Chapter 2 sample and ***average rewards*** for each action.

- ***The main difference is that now there are multiple states, each acting like a different bandit problem*** (like an associative-search or contextual bandit) and the ***different bandit problems are interrelated***.
  - random probability와  random process 차이인가...?
- That is, the return after taking an ***action in one state depends on the actions taken in later states in the same episode.***
  - 지난 states가 현재 state에 영향을 미침
  - Because all the action selections are undergoing learning, the problem becomes non-stationary from the point of view of the earlier state.

To handle the non-stationarity, we adapt the idea of ***general policy iteration (GPI)*** developed in Chapter 4 for DP. Whereas there we computed value functions from knowledge of the MDP, here we learn value functions from sample returns with the MDP.

- The value functions and corresponding policies still interact to attain optimality in essentially the same way (GPI).
- As in the DP chapter, first we consider the prediction problem (the computation of $$v_\pi$$ and $$q_\pi$$ for a fixed arbitrary policy $$\pi$$) then policy improvement, and, finally, the control problem and its solution by GPI. Each of these ideas taken from DP ***is extended to the Monte Carlo case in which only sample experience is available.***

## 5.1 Monte Carlo Prediction

Recall that the value of a state is the expected return—expected cumulative future discounted reward—starting from that state.

- An obvious way to estimate it from experience, then, is simply to average the returns observed after visits to that state.
  - As more returns are observed, the average should converge to the expected value.
  - 경험적으로 평균을 구하는 방법은 관측 될 때마다 평균을 구하면 됨
  - 관측 횟수가 많아질 수록 관측 평균은 실제 평균에 수렴 함.
  - This idea underlies all Monte Carlo methods.

Suppose we wish to estimate $$v_\pi (s)$$, the value of a state `s` under policy $$\pi$$, given a set of episodes obtained by following $$\pi$$ and passing through s. Each occurrence of state s in an episode is called a ***visit to `s`***. Of course, `s` may be visited multiple times in the same episode;

- Let us call the first time it is visited in an episode the first visit to `s`.
- ***The first-visit MC method*** estimates $$v_\pi(s)$$ as the average of the returns following first visits to `s`,
- Whereas ***the every-visit MC*** method averages the returns following all visits to s.
  - These two Monte Carlo (MC) methods are very similar but have slightly different theoretical properties.
  - First-visit MC has been most widely studied, dating back to the 1940s, and is the one we focus on in this chapter.
  - Every-visit MC extends more naturally to function approximation and eligibility traces, as discussed in Chapters 9 and 12.

Can we generalize the idea of backup diagrams to Monte Carlo algorithms?

The general idea of a backup diagram is to show at the top the root node to be updated and to show below all the transitions and leaf nodes whose rewards and estimated values contribute to the update.

- backup diagram : root에 업데이트 될 것들, 밑에는 업데이트에 기여 할 transitions와 reward를 보임
- For Monte Carlo estimation of $$v_\pi$$, the root is a state node, and below it is the entire trajectory(궤도) of transitions along a particular single episode, ending at the terminal state.
- Whereas the DP diagram shows all possible transitions, ***the Monte Carlo diagram shows only those sampled on the one episode.***
- Whereas the DP diagram includes only one-step transitions, ***the Monte Carlo diagram goes all the way to the end of the episode.***

These differences in the diagrams accurately reflect the fundamental differences between the algorithms.

- An important fact about Monte Carlo methods is that the estimates for each state are independent. The estimate for one state does not build upon the estimate of any other state, as is the case in DP.
- Monte Carlo에서는 state가 다른 state에 영향을 주지 않음
- In particular, note that the ***computational expense of estimating the value of a single state is independent of the number of states***. This can make Monte Carlo methods particularly attractive when one requires the value of only one or a subset of states.
- One can generate many sample episodes starting from the states of interest, averaging returns from only these states, ignoring all others.

## 5.2 Monte Carlo Estimation of Action Values

***If a model is not available, then it is particularly useful to estimate action values (the values of state–action pairs) rather than state values.***

- ***With a model, state values alone are sufficient to determine a policy;*** one simply looks ahead one step and chooses whichever action leads to the best combination of reward and next state, as we did in the chapter on DP.
- ***Without a model, however, state values alone are not sufficient.*** 
- One must explicitly estimate the value of each action in order for the values to be useful in suggesting a policy.
  - Thus, one of our primary goals for Monte Carlo methods is to estimate $$q_*$$. To achieve this, we first consider the policy evaluation problem for action values.(exactly $$q_\pi(s,a)$$)

The Monte Carlo methods for this are essentially the same as just presented for state values, except now we talk about visits to a state–action pair rather than to a state.

- A state–action pair s, a is said to be visited in an episode if ever the state s is visited and action a is taken in it.
- ***The every-visit MC method estimates the value of a state–action pair as the average of the returns that have followed all the visits to it.***
- The first-visit MC method averages the returns following the first time in each episode that the state was visited and the action was selected.
  - These methods converge quadratically, as before, to the true expected values as the number of visits to each state–action pair approaches infinity.
  - 이 방법은 2차식으로 무한대로 가기전에 빠르게 수렴 함
- The only complication is that many state–action pairs may never be visited.
  - If $$\pi$$ is a deterministic policy, then in following $$\pi$$ one will observe returns only for one of the actions from each state.
  - 단점은 방문 되지 않는 state-action pairs가 많음
  - 만약 policy가 deterministic하다면, 오직 하나의 action만 각각의 state에서 관측 될 것임
- To compare alternatives we need to estimate ***the value of all the actions*** from each state, not just the one we currently favor.
- ***This is the general problem of maintaining exploration***.
- For policy evaluation to work for action values, we must assure continual exploration.
  - One way to do this is by specifying that the episodes start in a state–action pair, and that every pair has a nonzero probability of being selected as the start.
  - 한가지 방법은 state-action pair에서 episode start를 특정화 하는 것 그리고 모든 pair가 non-zero 확률을 가지고 시작 점으로 선택 되는 것
  - This guarantees that all state–action pairs will be visited an infinite number of times in the limit of an infinite number of episodes. We call this the ***assumption of exploring starts.***
  - 이렇게 되면, 모든 state-action pair가 제한된 매우 큰 수 만큼 방문 될 것임
    - 우리는 이것을 탐험 시작 가정이라고 부름

## 5.3 Monte Carlo Control

We are now ready to consider how Monte Carlo estimation can be used in control, that is, to approximate optimal policies. The overall idea is to proceed according to the same pattern as in the DP chapter, that is, ***according to the idea of generalized policy iteration (GPI).***

- In GPI one maintains both an approximate policy and evaluation an approximate value function.
  - The value function is repeatedly altered to more closely approximate the value function for the current policy, and the ***policy is repeatedly improved with respect to the current value function***.
  - These two kinds of changes work against each other to some extent, as each creates a moving target for the other, but together they cause both policy and value function to approach optimality.

We made ***two unlikely assumptions above in order to easily obtain this guarantee of convergence for the Monte Carlo method.***

- One was that the episodes have exploring starts.
- And the other was that policy evaluation could be done with an infinite number of episodes.

To obtain a practical algorithm we will have to remove both assumptions.

For now we focus on the assumption that policy evaluation operates on an infinite number of episodes. This assumption is relatively easy to remove.

- In fact, the ***same issue arises even in classical DP methods such as iterative policy evaluation, which also converge only asymptotically to the true value function***.
  - In both DP and Monte Carlo cases ***there are two ways to solve the problem.***
    - One is to hold firm to the idea of approximating $$q_{\pi_{k}}$$ in each policy evaluation.
      - Measurements and assumptions are made to obtain bounds on the magnitude and probability of error in the estimates, and then sufficient steps are taken during each policy evaluation to assure that these bounds are sufficiently small.
      - 측정과 가정은 에러의 확률과 값을 범위 안에 들어오도록 하고, 충분한 스텝이 policy evaluation동안 이루어지면 이 범위는 충분히 작아진다
      - This approach can probably be made completely satisfactory in the sense of guaranteeing correct convergence up to some level of approximation. However, it is also likely to require far too many episodes to be useful in practice on any but the smallest problems.
      - 하지만 이 방법 역시 많은 횟수의 반복이 필요 함