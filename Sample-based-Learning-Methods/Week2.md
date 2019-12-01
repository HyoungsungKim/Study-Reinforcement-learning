# Chapter 6 Temporal-Difference Learning

If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be temporal-difference (TD) learning. ***TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas.***

- Like Monte Carlo methods, ***TD methods can learn directly from raw experience without a model of the environment’s dynamics.***
- Like DP, ***TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap)***. 

## 6.1 TD Prediction

Both TD and Monte Carlo methods ***use experience to solve the prediction problem.***

- Given some experience following a policy $$\pi$$, both methods(both methods : TD and Monte Carlo) update their estimate V of $$v_\pi$$ for the nonterminal states $$S_t$$ occurring in that experience.
- Roughly speaking, Monte Carlo methods wait until the return following the visit is known, then use that return as a target for $$V (S_t)$$.
- ***A simple every-visit Monte Carlo method suitable for non-stationary environments is***

$$
V(S_t) \gets V(S_t) + \alpha[G_t - V(S_t)]
$$

Where $$G_t$$ is the actual return following time t, and ***$$\alpha$$ is a constant step-size parameter***.

- Let us call this method constant-$$\alpha$$ MC.

Whereas ***Monte Carlo methods must wait until the end of the episode to determine the increment to $$V(S_t)$$ (only then is $$G_t$$ known), TD methods need to wait only until the next time step***. At time t + 1 they immediately form a target and make a useful update using the observed reward $$R_{t+1}$$ and the estimate $$V(S_{t+1} )$$.

- ***The simplest TD method makes the update immediately on transition to $$S_{t+1}$$ and receiving $$R_{t+1}$$.***

$$
V(S_t) \gets V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
$$

- In effect, the target for the Monte Carlo update is $$G_t$$
- In side of bracket : TD error
- Whereas the target for the $$TD$$ update is $$R_{t+1 }+ \gamma V(S_{t+1} )$$. 

This TD method is called $$TD(0)$$, or one-step $$TD$$, because it is a special case of the $$TD(\lambda)$$.

The value estimate for the state node at the top of the backup diagram is updated on the basis of the one sample transition from it to the immediately following state.

- ***We refer to TD and Monte Carlo updates as sample updates*** because they involve looking ahead to a *sample successor* state (or state–action pair), using the value of the successor and the reward along the way to compute a backed-up value, and then updating the value of the original state (or state–action pair) accordingly.
  - Sample updates differ ***from the expected updates of DP methods in that they are based on a single sample successor*** rather than on a complete distribution of all possible successors.
- ***Note that the quantity in brackets in the $$TD(0)$$ update is a sort of error***, measuring the difference between the estimated value of $$S_t$$ and the better estimate $$R_{t+1} + V(S_{t+1})$$. This quantity, called the TD *error*, arises in various forms throughout reinforcement learning:

$$
\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

- ***TD error*** depends on the next state and next reward, it is not actually available until one time step later.
- $$\delta_t$$ is the error in $$V(S_t)$$, available at time $$t + 1$$

## 6.2 Advantages of TD Prediction Methods

TD methods update their estimates based in part on other estimates. ***They learn a guess from a guess***—they bootstrap. Is this a good thing to do? What advantages do TD methods have over Monte Carlo and DP methods

Obviously, ***TD methods have an advantage*** over DP methods in that ***they do not require a model of the environment, of its reward and next-state probability distributions***. The next most obvious advantage of TD methods over Monte Carlo methods is that ***they are naturally implemented in an online, fully incremental fashion***.

- With ***Monte Carlo methods one must wait until the end of an episode***, because only then is the return known
- Whereas with ***TD methods one need wait only one time step.***

***Some Monte Carlo methods must ignore or discount episodes*** on which experimental actions are taken, which can greatly slow learning.

- 몬테 카를로 방법은 episodic 할 때에만 사용 가능 함

***TD methods are much less susceptible to these problems***

- Because they learn from each transition regardless of what subsequent actions are taken.
- TD methods는 각 transition마다 학습을 하기 때문에 MD 방법의 단점에서 자유로움

But are TD methods sound? Certainly it is convenient to learn one guess from the next, without waiting for an actual outcome, but can we still guarantee convergence to the correct answer? Happily, the answer is yes.

***For any fixed policy $$\pi$$, TD(0) has been proved to converge to $$v_\pi$$***.

If both TD and Monte Carlo methods converge asymptotically to the correct predictions, then a natural next question is ***“Which gets there first?”*** In other words, which method learns faster? Which makes the more efficient use of limited data?

- At the current time ***this is an open question*** in the sense that no one has been able to prove mathematically that one method converges faster than the other.
- In fact, it is not even clear what is the most appropriate formal way to phrase this question! In practice, ***however, TD methods have usually been found to converge faster than constant-$$\alpha$$ MC methods on stochastic tasks***

## 6.3 Optimality of TD(0)

Suppose there is available only a finite amount of experience, say 10 episodes or 100 time steps. In this case, a common approach with incremental learning methods is to present the experience repeatedly until the method converges upon an answer. All the available experience is processed again with the new value function to produce a new overall increment, and so on, until the value function converges. We call this ***batch updating*** because updates are made only after processing each complete ***batch of training data.***

- Value update한 value를 이용해서 다시 value를 업데이트함
  - 반복하면 정답에 수렴함.  이 방법을 batch update라고 부름
- ***Under batch updating, TD(0) converges deterministically to a single answer*** independent of the step-size parameter, $$\alpha$$, as long as $$\alpha$$ is chosen to be sufficiently small.
- ***The constant-$$\alpha$$ MC method also converges deterministically under the same conditions, but to a different answer***.
- Understanding these two answers will help us understand the difference between the two methods. ***Under normal updating the methods do not move all the way to their respective batch answers, but in some sense they take steps in these directions.***
- TD(0)와 MC의 값은 다르지만 같은 방향으로 전개 됨

### Example 6.4

illustrates a general difference between the estimates found by batch $$TD(0)$$ and batch Monte Carlo methods.

- Batch Monte Carlo methods always find the estimates that ***minimize mean-squared error*** on the training set
- Whereas batch TD(0) always finds the estimates that would be exactly correct for the ***maximum-likelihood model of the Markov process***.
  - In general, the maximum-likelihood estimate of a parameter is the ***parameter value whose probability of generating the data is greatest***.
    - Maximum-likelihood estimate of a parameter는 생성 확률을 가장 크게 만드는 매개변수 값
  - In this case, ***the maximum-likelihood estimate is the model of the Markov process formed in the obvious way from the observed episodes***: the estimated transition probability from i to j is the fraction(일부) of observed transitions from i that went to j, and the associated expected reward is the average of the rewards observed on those transitions.
  - Given this model, ***we can compute the estimate of the value function that would be exactly correct if the model were exactly correct***.
    - This is called the ***certainty-equivalence estimate*** because it is equivalent to assuming that the estimate of the underlying process was known with certainty rather than being approximated.
    - 추정치를 확신 할 수 있음 -> 모델이 주어지면 추정치가 옳바른 답과 같을 것임
    - In general, batch TD(0) converges to the certainty-equivalence estimate.

***This helps explain why TD methods converge more quickly than Monte Carlo methods***. In batch form, TD(0) is faster than Monte Carlo methods because it computes the true certainty-equivalence estimate. This explains the advantage of TD(0) shown in the batch results on the random walk task (Figure 6.2).

The relationship to the certainty- equivalence estimate may also explain in part the speed advantage of non-batch TD(0) (e.g., Example 6.2, page 125, right graph). ***Although the non-batch methods do not achieve either the certainty-equivalence or the minimum squared-error estimates, they can be understood as moving roughly in these directions.***

***Non-batch TD(0) may be faster than constant-$$\alpha$$ MC*** because it is moving toward a better estimate, even though it is not getting all the way there. At the current time nothing more definite(확고한) can be said about the relative efficiency of online TD and Monte Carlo methods.

- 현재까지는 online TD와 몬테 카를로 방법 보다 상대적으로 효과적인 방법은 없음

Finally, it is worth noting that although the certainty-equivalence estimate is in some sense an optimal solution, ***it is almost never feasible to compute it directly***.

- If $$n = |S|$$ is the number of states, then just forming the maximum-likelihood estimate of the process may require on the order of $$n^2$$ memory, and computing the corresponding value function requires on the order of $$n^3$$ computational steps if done conventionally.
- certainty-equivalence estimate가 optimal solution을 구하는 최적의 방법이지만 메모리 복잡도와 계산 복잡도가 높아서 구현이 쉽지 않음

In these terms it is indeed striking that TD methods can approximate the same solution using memory no more than order n and repeated computations over the training set.

- TD methods 는 n 만큼의 메로리를 쓰면서 트레이닝 셋의 반복으로 certainty-equivalence estimate 값을 추정 할 수 있음.
- Certainty-equivalence estimate를 구현하려면 모델이 필요 함.
- 하지만 TD methods는 모델 없이, 메모리도 조금쓰면서 certainty-equivalence estimate 값에 근접 함 -> 직접 구현하는 것보다 효율적임

On tasks with large state spaces, ***TD methods may be the only feasible way of approximating the certainty-equivalence solution.***