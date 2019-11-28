# Chapter 4 Dynamic Programming

The term dynamic programming (DP) refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a Markov decision process (MDP).

- Classical DP algorithms are of limited utility in reinforcement learning 
  - Because of their assumption of a perfect model
  - Because of their great computational expense
- But they are still important theoretically. In fact, all of these methods can be viewed as attempts to achieve much the same effect as DP, only with less computation and without assuming a perfect model of the environment.

We usually assume that the environment is a finite MDP.

- That is, ***we assume that its state, action, and reward sets, S, A, and R, are finite***, and that its dynamics are given by a set of probabilities $$p(s',r |s, a),\text{ for all } s\in S, a \in A(s), r \in R, \text{ } s' \in S^+$$
  - $$S^+$$ is S plus a terminal state if the problem is episodic.
- Although DP ideas can be applied to problems with continuous state and action spaces, exact solutions are possible only in special cases.
  - Continuous 한 경우네는 특별한 경우에만 사용 할 수 있음
- A common way of obtaining approximate solutions for tasks with continuous states and actions is to quantize the state and action spaces and then apply finite-state DP methods.

The key idea of DP, and of reinforcement learning generally, is the use of value functions to organize and structure the search for good policies.

- In this chapter we show how DP can be used to compute the value functions defined in Chapter 3.
- As discussed there, we can easily obtain optimal policies once we have found the optimal value functions, $$v_* \text{ or } q_*$$, which satisfy the Bellman optimality equations:

$$
\begin{alignat}{4} v_*(s) & = \underset{a}{\mathbb{E}}[R_{t+1} + \gamma v_*(S_{t+1}) | S_t=s , A_t = a] \\
& = \underset{a}{max} \sum_{s', r} p(s',r|s,a)[r + \gamma v_*(s')], \text{ or } \\
q_*(s,a) & = \mathbb{E} [R_{t+1} + \gamma \text{ } \underset{a'}{max} \text{ } q_*(S_{t+1}, a') | S_t = s, A_t = a ] \\
& = \sum_{s',r'}p(s', r | s, a)[r + \gamma \text{ } \underset{a'}{max} \text{ } q_*(s',a')]
\end{alignat}
$$

- $$\pi(a|s)$$는 optimal 일때 1
- $$\pi, p, \gamma$$ -> Linear System Solver -> $$v_\pi$$
- In practice, $$\pi, p, \gamma%%$$ -> Dynamic Programming -> $$v_\pi$$ : Policy Evaluation
  - It is more suitable for general MDP
- $$p, \gamma$$ -> Dynamic Programming -> $$\pi_*$$ : Control

for all $$s \in S, a \in A(s), \text{ and } s' \in S^+$$. As we shall see, DP algorithms are obtained by turning Bellman equations such as these into assignments, that is, into update rules for improving approximations of the desired value functions.

## 4.1 Policy Evaluation (Prediction)

- ***Policy evaluation*** is the task of determining the state-value function $$v_\pi$$, for a particular policy $$\pi$$ 
  - ***Policy evaluation***은 특정한 policy에서 state-value 함수 $$v_\pi$$를 계산 하는 방법
- ***Control*** is the task of improving an existing policy
  - ***Control***은 존재하는 policy를 증가시키는 방법
- ***Dynamic programming*** techniques can be used to solve both these tasks, if we have access to the dynamics function p
  - ***Dynamic Programming***은 이러한 작업들을 해결하기 위한 방법.

First we consider how to compute the state-value function $$v_\pi$$ for an arbitrary policy $$\pi$$. This is called ***policy evaluation in the DP literature***.

- We also refer to it as the ***prediction problem***. For all $$s \in S$$,
- 이 식은 policy를 순회(iterator) 함

$$
\begin{alignat}{2}
v_\pi(s)  & \doteq \mathbb{E}_\pi [ G_t | S_t = s] \\
& = \mathbb{E}_\pi [R_{t+1} + \gamma G_{t + 1} | S_t = s ] \\
& = \mathbb{E}_\pi [R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s] & \text{(4.3)} \\
& = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a)[r + \gamma v_\pi(s')] & \text{ (4.4)}
\end{alignat}
$$

where $$\pi(a|s)$$ is the probability of taking action a in state s under policy $$\pi$$, and the expectations are subscripted by $$\pi$$ to indicate that they are conditional on $$\pi$$ being followed. 

***If the environment’s dynamics are completely known***(dynamic : (4.4)에서 p 부분), then (4.4) is a system of $$|S|$$ simultaneous linear equations in $$|S|$$ unknowns (the $$v_\pi(s), s \in S$$).

- In principle, its solution is a straightforward, if tedious, computation.
- For our purposes, iterative solution methods are most suitable.
  - $$S^+$$ : Next state
  - Consider a sequence of approximate value functions $$v_0 , v_1 , v_2,...,$$ each mapping $$S^+ \text{ to } \mathbb{R} \text{ (the real numbers)}$$.
  - The initial approximation, $$v_0$$ , is chosen arbitrarily (except that the terminal state, if any, must be given value 0), and each successive approximation is obtained by using the Bellman equation for $$v_\pi$$ (4.4) as an update rule:
    - $$v_0$$ (초기값)을 terminate state 같은 value가 0이 아닌 임의의 값으로 지정
    - 그 다음 값들은 bellman equation으로 계산 함

$$
\begin{alignat}{2}
v_{k+1} & \doteq \mathbb{E}_\pi[R_{t+1} + \gamma v_k(S_{t+1}) | S_t = s] \\
&= \sum_{a} \pi(a|s)\sum_{s',r}p(s',r|s,a)[r + \gamma v_k(s')] \text{ (4.5)}
\end{alignat}
$$

For all $$s \in S$$. Clearly, $$v_k = v_\pi$$ is a fixed point for this update rule because the Bellman equation for $$v_\pi$$ assures us of equality in this case.

- Indeed, the sequence $$\{v_k \}$$ can be shown in general to converge to $$v_\pi$$ as $$k \to \infty$$ under the same conditions that guarantee the existence of $$v_\pi$$. This algorithm is called ***iterative policy evaluation***.

To produce each successive approximation, $$v_{k+1}$$ from $$v_k$$ , iterative policy evaluation applies the same operation to each state s:

- It replaces the old value of s with a new value obtained from the old values of the successor states of s, and the expected immediate rewards, along all the one-step transitions possible under the policy being evaluated.
  - We call this kind of operation an ***expected update***.
  - Each iteration of iterative policy evaluation updates the value of every state once to produce the new approximate value function $$v_{k+1}$$.
  - There are several different kinds of expected updates, depending on whether a state (as here) or a state–action pair is being updated, and depending on the precise way the estimated values of the successor states are combined.
  - ***All the updates done in DP algorithms are called expected updates***
    - Because they are based on an expectation over all possible next states rather than on a sample next state.
    - The nature of an update can be expressed in an equation, as above, or in a backup diagram like those introduced in Chapter 3.

To write a sequential computer program to implement iterative policy evaluation as given by (4.5) ***you would have to use two arrays, one for the old values, $$v_k(s)$$, and one for the new values, $$v_{k+1}$$ (s).***

- 배열을 두개 쓸 경우 하나의 배열을 다 채운뒤에 그 배열을 이용하지만, 한개의 배열을 사용 할 때에는 배열에 채워진 update value를 바로 사용 함 -> 더 빠르게 수렴
-  With two arrays, the new values can be computed one by one from the old values without the old values being changed.
- Of course it is easier to use one array and update the values “in place,” that is, with each new value immediately overwriting the old one. Then, depending on the order in which the states are updated, sometimes new values are used instead of old ones on the right-hand side of (4.5).
  - This in-place algorithm also converges to $$v_{\pi}$$; in fact, ***it usually converges faster than the two-array version,*** as you might expect, because it uses new data as soon as they are available.
  - For the in-place algorithm, ***the order in which states have their values updated during the sweep has a significant influence on the rate of convergence.*** We usually have the in-place version in mind when we think of DP algorithms.
  - Sweep : Updates as being done

Formally, iterative policy evaluation converges only in the limit, but in practice it must be halted short of this. The pseudo code tests the quantity $$\underset{s \in S}{max}  |v_{k+1} (s) - v_{k}(s)|$$ after each sweep and stops when it is sufficiently small.

## 4.2 Policy Improvement

Our reason for computing the value function for a policy is to help find better policies. Suppose we have determined the value function $$v_\pi$$ for an arbitrary deterministic policy $$\pi$$.

- For some state s we would like to know whether or not we should change the policy to deterministically choose an action $$a \neq \pi(s)$$.
  - We know how good it is to follow the current policy from s—that is $$v_\pi (s)$$—but would it be better or worse to change to the new policy? One way to answer this question is to consider selecting `a` in `s` and thereafter following the existing policy, $$\pi$$. The value of this way of behaving is

$$
\begin{alignat}{2}
q_\pi & \doteq \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t = a] \\
&= \sum_{s', r} p(s',r | s,a)[r + \gamma v_\pi(s')]
\end{alignat}
$$

That this is true is a special case of a general result called the ***policy improvement theorem***. Let $$\pi$$ and $$\pi'$$ be any pair of deterministic policies such that, for all $$s \in S $$,

- $$a = \pi(s)$$
- $$v_\pi(s)$$ : ***value*** of state `s` under policy $$\pi$$
- $$q_\pi(s,a)$$ : ***value*** of taking action `a` in state `s`under policy $$\pi$$

$$
q_\pi(s, \pi'(s)) \ge v_\pi(s) \text{ (4.7) }
$$

Then the policy $$\pi'$$ must be as good as, or better than, $$\pi$$. That is, it must obtain greater or equal expected return from all states $$s \in S$$:
$$
v_\pi' \ge v_\pi(s) \text{ (4.8) }
$$

- Moreover, if there is strict inequality of (4.7) at any state, then there must be strict inequality of (4.8) at that state.

## 4.3 Policy Iteration