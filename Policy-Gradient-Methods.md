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
