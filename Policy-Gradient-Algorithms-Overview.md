# Policy Gradient Algorithms Overview

https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

## Policy Gradient

Reward function:
$$
J(\theta) = \sum_{s \in S}d^\pi(s)V^\pi(s) = \sum_{s \in S} d^\pi(s)\sum_{a \in A}\pi_\theta(a|s)Q^\pi(s,a)
$$

- $$d^\pi(s)$$ is the stationary distribution of Markov chain for $$\pi_\theta$$

Using gradient ascent we can move $$\theta$$ toward the direction suggested by the gradient $$\nabla_\theta J(\theta)$$ to find the best $$\theta$$ for $$\pi_\theta$$ that produces the highest return.

- Vanilla policy gradient update has no bias but high variance. 

## Policy Gradient Algorithms

### REINFORCE

REINFORCE(Monte-Carlo policy gradient) relies on estimated return by Monte-Carlo methods using episodes samples to update the policy parameter $$\theta$$. 

- REINFORCE works because the expectation of the sample gradient is equal to the actual gradient.

A widely used variation of REINFORCE is to subtract a baseline value from the return $$G_t$$ to *reduce the variance of gradient estimation while keeping the bias unchanged* (Remember we always want to do this when possible).

### Actor-Critic

Two main components in policy gradient are ***the policy model and the value function***.

- It makes a lot of sense to learn the value function in addition to the policy, since knowing the value function can assist the policy update, such as by reducing gradient variance in vanilla policy gradients, and that is exactly what  the **Actor-Critic** method does.

### Off-Policy Policy Gradient

***Both REINFORCE and the vanilla version of actor-critic methods are on-policy***: training samples are collected according to the target policy. ***Off-policy methods, however, result in several additional advantages:***

- The off-policy approach does not require full trajectories and can reuse any past episodes(experience replay) for much better sample efficiency
- The sample collection follows a behavior policy different from the target policy, bringing better exploration

### A3C

Asynchronous Advantage Actor-Critic is a classic policy gradient methods with a special focus on parallel training.

- In A3C, the critics learn the value function while multiple actors are trained in parallel and get synced with global parameters from time to time. Hence, A3C is designed to work well for parallel training.
- A3C enables the parallelism in multiple agent training. 

### DPG

**Deterministic policy gradient (DPG)** instead models the policy as a deterministic decision: $$a=μ(s)$$.

- We can consider the deterministic policy as a *special case* of the stochastic one, when the probability distribution contains only one extreme non-zero value over one action. 
- However, unless there is sufficient noise in the environment, it is very hard to guarantee enough exploration due to the determinacy of the policy.
  - We can either add noise into the policy (ironically this makes it  nondeterministic!) or learn it off-policy-ly by following a different  stochastic behavior policy to collect samples.

### TPRO

To improve training stability, we should avoid parameter updates that change the policy too much at one step. **Trust region policy optimization (TRPO)** carries out this idea by enforcing a KL divergence constraint on the size of policy update at each iteration.

- The objective function in an off-policy model measures the total advantage over the state visitation distribution and actions
- While the mismatch between the training data distribution and the true policy state distribution is compensated by importance sampling estimator

### PPO

Given that TRPO is relatively complicated and we still want to implement a similar constraint, **proximal policy optimization (PPO)** simplifies it by using a clipped surrogate objective while retaining similar performance.