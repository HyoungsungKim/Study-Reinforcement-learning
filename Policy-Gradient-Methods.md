# Policy Gradient Methods

Sutton CH 13

In this chapter, we consider methods that instead learn a parameterized policy that can select actions without consulting a value function.

- A value function may still be used to learn the policy parameter, but is not required for action selection.
  - Value function은 여전히 policy parameter로 사용 될 수 있지만 action selection에 필수는 아님
- We use the notation $$\theta \in \mathbb{R}^{d^`}$$ d for the policy’s parameter vector.

Thus we can write
$$
\pi(a|s,\theta) = Pr\lbrace{A_t = a | s_t = s, \theta_t = \theta} \rbrace
$$
In this chapter ***we consider methods for learning the policy parameter based on the gradient*** of some scalar performance measure $$J(\theta)$$ with respect to the policy parameter.

- $$J(\theta)$$ : Performance measure for the policy $$\pi_\theta$$

These methods seek to ***maximize performance***, so their updates approximate ***gradient ascent*** in J
$$
\theta_{t+1} = \theta_t + \alpha \nabla J(\theta_t)
$$

- All methods that follow this general schema we call ***policy gradient methods***, whether or not they also learn an approximate value function.
- Methods that learn approximations to both policy and value functions are often called ***actor–critic methods***, where ‘actor’ is a reference to the learned policy, and ‘critic’ refers to the learned value function, usually a state-value function.

>- A continuous task never ends.
>  - Which means you're not given the reward at the end, since there is no end, but every so often during the task.
>  - Reading the internet to learn maths could be considered a continuous task.
>- An episodic task lasts a finite amount of time. 
>  - Playing a single game of Go is an episodic task, which you win or lose.

## 13.1 Policy Approximation and its Advantages

$$\nabla \pi(a|s, \theta) $$ : the column vector of partial derivatives of $$\pi(a|s, \theta)$$ with respect to the components of $$\theta$$

