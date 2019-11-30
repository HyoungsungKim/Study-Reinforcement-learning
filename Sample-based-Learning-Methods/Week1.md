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
- 같은 에피소드에서 한 스테이트를 여러번 방문 할 수도 있음
- 여기서 처음 방문 했을 때 값을 평균으로 사용 함(처음 방문 했을 때에만 누적 합에 포함시킴) -> First-visit MC method
- [Psudo code](https://ai.stackexchange.com/questions/10812/what-is-the-difference-between-first-visit-monte-carlo-and-every-visit-monte-car)
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
    - There is a second approach to avoiding the infinite number of episodes nominally required for policy evaluation, in which ***we give up trying to complete policy evaluation before returning to policy improvement.***
      - On each evaluation step we move the value function toward $$q_{\pi_k}$$, but we do not expect to actually get close except over many steps.
      - One extreme form of the idea is value iteration, in which only one iteration of iterative policy evaluation is performed between each step of policy improvement. The in-place version of value iteration is even more extreme

## 5.4 Monte Carlo Control without Exploring Starts

How can we avoid the unlikely assumption of exploring starts? There are two approaches to ensuring this, resulting in what we call ***on-policy methods and off-policy methods***.

- On-policy methods attempt to evaluate or improve the policy that is used to make decisions
  - On-policy는 결정을 내리는데 사용했던 정책을 평가하거나 발전 시키는 것을 시도 함
  - The Monte Carlo ES(Exploring Starts - Maintain exploration) method developed above is an example of an on-policy method.
- Whereas off-policy methods evaluate or improve a policy different from that used to generate the data.
  - Off-policy는 data를 생성하는데 사용 했던 정책 차이를 평가하거나 발전 시킴
  - Off-policy methods are considered in the next section.

In ***on-policy*** control methods the policy is generally ***soft***, meaning that $$\pi(a|s) > 0$$ for all $$s \in S$$ and all $$a \in A(s)$$, but gradually ***shifted closer and closer to a deterministic optimal policy.***

- Deterministic : 같은 input이 들어가면 같은 output이 나옴
- $$\epsilon$$-greedy : 같은 input이 들어가도 확률적으로 다른 output이 나올 수도 있음
- $$\epsilon$$-soft : Deterministic optimal policy로 조금씩 이동하는  $$\epsilon$$-greedy

- The on-policy method we present in this section uses $$\epsilon$$-greedy policies, meaning that ***most of the time they choose an action that has maximal estimated action value, but with probability $$\epsilon$$ they instead select an action at random.***
- Greedy한 선택을 하다가 낮은 확률로($$\epsilon$$) 다른 행동 선택
- That is, all non-greedy actions are given the minimal probability of selection, $$\frac{\epsilon}{|A(s)|}$$ , and the remaining bulk of the probability, $$1 - \epsilon + \frac{\epsilon}{|A(s)|}$$, is given to the greedy action.
- $$\frac{\epsilon}{|A(s)|}$$  : Joint probability -> one of actions and it is non-greedy?(Joint probability)
- $$1 - \epsilon \{ {1 - \frac{1}{|A(s)|}} \}$$ :  
  - $$\epsilon \{ {1 - \frac{1}{|A(s)|}} \}$$ : action 중에 하나를 선택했는데 그걸 제외한 모든 action이 non-greedy 할 확률
  - 1에서 action 중에 하나를 선택 했는데 그걸 제외한 모든 action이 non-greedy한 확률을 빼면 하나의 액션이 greedy한 선택일 확률 이 됨(맞나...?)
- ***The $$\epsilon$$-greedy policies are examples of $$\epsilon$$-soft policies,*** defined as policies for which $$\pi(a|s) \ge \frac{\epsilon}{|A(s)|}$$ for all states and actions, for some $$\epsilon > 0$$. Among $$\epsilon$$-soft policies, $$\epsilon$$-greedy policies are in some sense those that are closest to greedy.
- ***Epsilon soft policies Force the agent to continually explore*** that means we can drop the exploring starts requirement from the Monte Carlo control algorithm an Epsilon soft policy assigns nonzero probability to each action in every state ***because of this Epsilon soft agents continue to visit all state action pairs indefinitely.***

Fortunately, GPI does not require that the policy be taken all the way to a greedy policy, only that it be moved toward a greedy policy.

## 5.5 Off-policy Prediction via Importance Sampling

- On-policy : Improve and evaluate the policy being used to select actions
  - 액션 선택에 사용되는 정책을 향상시키거나 평가 하는 것
- Off-policy : Improve and evaluate a different policy from the one used to select actions
  - 선택 된 정책과 다른 정책을 향상 시키고 평가 하는 것

For example, you could learn the optimal policy while following a totally random policy ***we call the policy that the agent is learning the target policy*** because it is the target of the agents learning.

- ***The value function that the agent is learning is based on the target policy.***
  - Target policy : $$\pi(a|s)$$
  - 에이전트는 타겟 폴리시를 기반으로 학습 함
  - one example of a Target policy is the optimal policy.
- ***We call the policy that the agent is using to select actions the behavior policy*** because it defines our agents Behavior.
  - Behavior policy : $$b(a|s)$$
  - The behavior policy is in ***charge of selecting actions*** for the agent.
  - 액션을 선택하는데 들어가는 비용
  - The behavior policies shown here is the uniform random policy. 
- Coupling targeting and behavior
  - Because it provides another strategy for continual exploration
  - If our agent behaves according to the Target policy it might only experience a small number of states. 
  - 만약 타겟 폴리시만 따라간다면 충분한 숫자의 state가 학습이 안된 것
  - If our agent can behave according to a policy that favors exploration, It can experience a much larger number of states. 
  - 만약 우리의 에이전트가 탐험한 정책에 따를 수 있다면  더 많은 숫자의 정책을 경험 할 수 있음

One key rule of off-policy learning is that the ***behavior policy must cover the target policy.***

- In other words, if the target policy says the probability of selecting an action a given State `s` is greater than zero, ***then the behavior policy must say the probability of selecting that action in that state is greater than 0***. There is a key mathematical reason for this that we will discuss in an upcoming video. 
  - $$\pi(a|s) > 0 \text{ where } b(a|s) > 0$$
- ***On-policies the specific case where the target policy is equal to the behavior policy.***
- Off-policy learning is another way to obtain continual exploration.
- Off-policy learning allows learning an optimal policy from suboptimal behavior
- ***The policy that we are learning is the target policy***
- ***The policy that we are choosing actions from is the behavior policy***

All learning control methods face a dilemma:

- They seek to learn action values conditional on subsequent optimal behavior
- But they need to behave non-optimally in order to explore all actions (to find the optimal actions).
- Optimal한 action을 원하지만 optimal한 action만 하면 explore 할 수 없음

How can they learn about the optimal policy while behaving according to an exploratory policy?

- The on-policy approach in the preceding section is actually a compromise—it learns action values not for the optimal policy, but for a near-optimal policy that still explores.
- A more straightforward approach is to use two policies
  - One that is learned about and that becomes the optimal policy
  - One that is more exploratory and is used to generate behavior.
- The policy being learned about is called the ***target policy***, and the policy used to generate behavior is called the ***behavior policy***. In this case we say that learning is from data “off” the target policy, and the overall process is termed off-policy learning.

On-policy methods are generally simpler and are considered first. Off-policy methods require additional concepts and notation, and because the data is due to a different policy, ***off-policy methods are often of greater variance and are slower to converge***. On the other hand, off-policy methods are more powerful and general. They include on-policy methods as the special case in which the target and behavior policies are the same. Off-policy methods also have a variety of additional uses in applications. For example, they can often be applied to learn from data generated by a conventional non-learning controller, or from a human expert.

- In order to use episodes from b to estimate values for $$\pi$$, we require that every action taken under $$\pi$$ is also taken, at least occasionally, under b. ***That is, we require that $$\pi(a|s) > 0$$ implies $$b(a|s) > 0$$***.

Almost all off-policy methods utilize importance sampling, a general technique for estimating expected values under one distribution given samples from another.

- ***We apply importance sampling to off-policy learning by weighting returns*** according to the relative probability of their trajectories occurring under the target and behavior policies, called the ***importance-sampling ratio.***

## 5.10 Summary

The Monte Carlo methods presented in this chapter learn value functions and optimal policies from experience in the form of sample episodes. ***This gives them at least three kinds of advantages over DP methods.***

- First, ***they can be used to learn optimal behavior directly from interaction with the environment, with no model of the environment’s dynamics.***
  - 모델 없이 최적 값을 알 수 있음
- Second, ***they can be used with simulation or sample models.*** For surprisingly many applications it is easy to simulate sample episodes even though it is difficult to construct the kind of explicit model of transition probabilities required by DP methods.
  - 시뮬레이션 하기 좋음
- Third, ***it is easy and efficient to focus Monte Carlo methods on a small subset of the states.*** A region of special interest can be accurately evaluated without going to the expense of accurately evaluating the rest of the state set.
  - State의 부분 집합이 적을 때에도 사용 하기 좋음
- A fourth advantage of Monte Carlo methods, which we discuss later in the book, is that ***they may be less harmed by violations of the Markov property***. This is because they do not update their value estimates on the basis of the value estimates of successor states. In other words, it is because they do not bootstrap.

***In control methods we are particularly interested in approximating action-value functions***, because these can be used to ***improve the policy without requiring a model of the environment’s transition dynamics.***

- 모델없이 policy 향상 시킴

***Maintaining sufficient exploration is an issue in Monte Carlo control methods***. 

- One approach is to ignore this problem by assuming that episodes begin with state–action pairs randomly selected to cover all possibilities.
  - Such exploring starts can sometimes be arranged in applications with simulated episodes, but are unlikely in learning from real experience.
  - 하나의 방법은 시작점을 무작위로 선택하는것. 하지만 현실에 적용하기에는 맞지 않음
- In ***on-policy methods***, the agent commits to always exploring and tries to find the best policy that still explores.
  - On-policy에서는 에이전트가 항상 최고의 policy를 찾으려 탐험 함
  - Episode 동안 학습 할 수 있음
- In ***off-policy methods***, the agent also explores, but learns a deterministic optimal policy that may be unrelated to the policy followed.
  - Off-policy에서도 에이전트는 탐험하지만 on-policy에서 지나지 않았던 deterministic optimal policy를 탐험 함
  - Episode가 다 끝나야 학습이 가능 함

Off-policy prediction refers to learning the value function of a target policy from data generated by a different behavior policy.

- ***Such learning methods are based on some form of importance sampling***, that is, on weighting returns by the ratio of the probabilities of taking the observed actions under the two policies.
  - 이런 학습 방법은 importance sampling에 기반을 둠. 즉 2가지 policies 아래에서 action을 취하는 확률의 비율로 관측된 가중치를 반환 함.

The Monte Carlo methods treated in this chapter differ from the DP methods treated in the previous chapter in two major ways.

- First, they operate on sample experience, and thus ***can be used for direct learning without a model.***
- Second, they do not bootstrap. ***That is, they do not update their value estimates on the basis of other value estimates.***

These two differences are not tightly linked, and can be separated. In the next chapter we consider methods that learn from experience, like Monte Carlo methods, but also bootstrap, like DP methods.

