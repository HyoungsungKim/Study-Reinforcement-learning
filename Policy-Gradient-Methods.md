# Chapter 13 Policy Gradient Methods

In this chapter we consider methods that instead learn a parameterized policy that can select actions without consulting a value function. ***A value function may still be used to learn the policy parameter, but is not required for action selection.*** We use the notation $$\theta \in \mathbb{R}^{d'}$$ for the policy's parameter vector.
$$
\pi(a|s, \theta) = Pr(A_t=a | S_t=s, \theta_t = \theta)
$$
In this chapter we consider methods for learning the policy parameter based on the gradient of some scalar performance measure $$J(\theta)$$ with respect to the policy parameter. These methods seek to maximize performance, so their updates approximate ***gradient ascent*** in J:
$$
\theta_{t+1} = \theta_t + \alpha \nabla J(\theta_t)
$$

- All methods that follow this general schema we call ***policy gradient methods***, whether or not they also learn an approximate value function. 
- Methods that ***learn approximations to both policy and value functions are often called actor–critic methods,***
  - where ***`actor` is a reference to the learned policy***
  - ***`critic` refers to the learned value function***, usually a state-value function.

## 13.1 Policy Approximation and its Advantages

In practice, to ensure exploration we generally require that ***the policy never becomes deterministic*** (i.e., that $$\pi(a|s,\theta) \in (0, 1) \text{ for all s, a, }\theta$$).

- In this section we introduce ***the most common parameterization for discrete action spaces*** and point out the advantages it offers over action-value methods.
  - Policy-based methods also offer ***useful ways of dealing with continuous action spaces***
- If the action space is discrete and not too large, then a natural and common kind of parameterization is to form parameterized numerical preferences $$h(s,a,\theta) \in \mathbb{R}$$ for each state-action pair.
  - The actions with the highest preferences in each state are given the highest probabilities of being selected(e.g. softmax)

$$
\pi(a|s, \theta) \doteq \frac{e^{h(s,a,\theta)}}{\sum_b e^{h(s,b,\theta)}}
$$

The action preferences themselves can be parameterized arbitrarily. For example, they might be computed by a deep artificial neural network (ANN), where $$\theta$$ is the vector of all the connection weights of the network.
$$
h(s,a,\theta) = \theta^\intercal x(s,a)
$$

- $$\theta$$ : Weights
- $$x(s,a)$$ : Input of NN(Feature vector)
- One advantage of parameterizing policies according to the soft-max in action preferences is that the ***approximate policy can approach a deterministic policy***
  - whereas with $$\epsilon$$-greedy action selection over action values there is always an $$\epsilon$$ probability of selecting a random action. 
  - If the soft-max distribution included a temperature parameter, then the temperature could be reduced over time to approach determinism, ***but in practice it would be difficult to choose the reduction schedule, or even the initial temperature, without more prior knowledge of the true action values than we would like to assume.***
- A second advantage of parameterizing policies according to the soft-max in action preferences is that ***it enables the selection of actions with arbitrary probabilities.*** In problems with significant function approximation, the best approximate policy may be stochastic.
  - For example, in card games with imperfect information the optimal play is often to do two different things with specific probabilities, such as when bluffing in Poker.
  - Action-value methods have no natural way of finding stochastic optimal policies, whereas policy approximating methods can
  - Policy 기반에서는 주어진 정보가 불완전 해도 최적 값을 찾을 수 있음
- we note that the ***choice of policy parameterization is sometimes a good way of injecting prior knowledge about the desired form of the policy into the reinforcement learning system.*** This is often the most important reason for using a policy-based learning method.

## 13.2 The Policy Gradient Theorem

The episodic and continuing cases define the performance measure, $$J(\theta)$$, differently and thus have to be treated separately to some extent. Nevertheless, we will try to present both cases uniformly, and we develop a notation so that the major theoretical results can be described with a single set of equations.

***In this section we treat the episodic case,*** for which we define the performance measure as the value of the start state of the episode. We can simplify the notation without losing any meaningful generality by assuming that every episode starts in some particular (non-random) state $$s_0$$ .

Then, in the episodic case we define performance as
$$
J(\theta) \doteq v_{\pi_\theta}(s_0)
$$
How can we estimate the performance gradient with respect to the policy parameter when the gradient depends on the unknown effect of policy changes on the state distribution?

- Fortunately, there is an excellent theoretical answer to this challenge in the form of the ***policy gradient theorem***, which provides an analytic expression for the gradient of performance with respect to the policy parameter that does not involve the derivative of the state distribution.

$$
\nabla J(\theta) \propto \sum_s \mu(s) \sum_q q_\pi (s,a) \nabla \pi(a|s, \theta)
$$

## 13.3 REINFORCE: Monte Carlo Policy Gradient

The policy gradient theorem gives an exact expression proportional to the gradient; all that is needed is some way of sampling whose expectation equals or approximates this expression.
$$
\nabla J(\theta)  \propto \sum_s \mu(s) \sum_q q_\pi (s,a) \nabla \pi(a|s, \theta)
$$

- It means $$\mathbb{E}_\pi[\sum_a q_\pi (S_t,a) \nabla \pi(a|s_t,\theta)]$$
- Because $$\mu(s)$$ is distribution of state

As a stochastic gradient method, REINFORCE has good theoretical convergence properties.

- By construction, the expected update over an episode is in the same direction as the performance gradient.
- This assures an improvement in expected performance for sufficiently small learning rate, and convergence to a local optimum under standard stochastic approximation conditions for decreasing learning rate.
- However, as a Monte Carlo method REINFORCE may be of high variance and thus produce slow learning.

## 13.4 REINFORCE with Baseline

The policy gradient theorem can be generalized to include a ***comparison of the action value to an arbitrary baseline b(s):***
$$
\nabla J(\theta)\propto \sum_s \mu(s) \sum_a(q_\pi(s,a) - b(s))\nabla \pi(a|s,\theta)
$$
***The baseline can be any function, even a random variable, as long as it does not vary with `a`;*** the equation remains valid because the subtracted quantity is zero:

Because the baseline could be uniformly zero, this update is a strict generalization of REINFORCE.

- ***In general, the baseline leaves the expected value of the update unchanged, but it can have a large effect on its variance.***
- For example, an analogous baseline can significantly reduce the variance (and thus speed the learning) of gradient bandit algorithms. In the bandit algorithms the baseline was just a number (the average of the rewards seen so far),
- ***but for MDPs the baseline should vary with state.*** In some states all actions have high values and we need a high baseline to differentiate the higher valued actions from the less highly valued ones; in other states all actions will have low values and a low baseline is appropriate.

One natural choice for the baseline is an estimate of the state value, $$\hat{v}(S_t ,w)$$, where $$w \in \mathbb{R}^m$$ is a weight vector learned by one of the methods presented in previous chapters.

## 13.5 Actor-Critic Methods

We do not consider it to be an actor–critic method because its state-value function is used only as a baseline, not as a critic.

- That is, it is not used for bootstrapping (updating the `value estimate` for a state from the estimated values of subsequent states), but only as a baseline for the state whose estimate is being updated.

As we have seen, ***the bias introduced through bootstrapping and reliance on the state representation is often beneficial because it reduces variance and accelerates learning.***

- REINFORCE with baseline is unbiased and will converge asymptotically to a local minimum, but like all Monte Carlo methods it tends to learn slowly (produce estimates of high variance) and to be inconvenient to implement online or for continuing problems.
- ***Temporal-difference methods we can eliminate these inconveniences, and through multi-step methods we can flexibly choose the degree of bootstrapping***.
  - In order to gain these advantages in the case of policy gradient methods ***we use actor–critic methods with a bootstrapping critic.***
  - Bootstrapping : 에피소드가 끝나지 않아도 업데이트
- REINFORCE는 편향되지 않고 로컬 미니멈에 수렴 시킬 수 있음. 하지만 수렴 속도가 느리고, 분산이 크고, 연속 문제에서 불편 함
- 시간차 방법은 이 문제를 해 결할 수 있기 때문에 함께 사용 함.

The main appeal of one-step methods is that they are fully online and incremental, yet avoid the complexities of eligibility traces. They are a special case of the eligibility trace methods, and not as general, but easier to understand. ***One-step actor–critic methods replace the full return of REINFORCE with the one-step return (and use a learned state-value function as the baseline)***

