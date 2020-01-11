# Week 5

## Policy-based RL vs Value-based RL

### Intuition

-  You've probably done some great things in the ***value-based method***
  - The idea behind value-based methods that is easy to explain intuitively
    - Find an optimal action, you want your algorithm to learn how much reward, how much as a discounted do you think you will get. 
  - If it starts from this state, maybe takes this action, and then follows this policy or maybe optimal policy for Q-learning. 
- This approach is very useful, provided that you know this function. But today, we're going to study another family of methods that try to explicitly avoid learning the Q or V or any other value-related functions. 
- The Q-learning doesn't explicitly learn what to do. It instead tries to learn what kind of a value, what Q function you'll get if you do this, and it's kind of hard, especially if you consider this applied to everyday problems.
- ***Your neural network would never be able to approximate old Q values in a way that it has no error***
  - So it would have some approximation error there, and what it actually tries to do, it tries to make an approximation which minimizes the lost function or a mean squared error in this case

#### Conclusion

The problem is, in that your DQN, while trying to minimize square root error, ***will avoid optimal policy and try to converge to something different,*** which is although more accurate than sense of a slow function, it's not what you actually want, you want to play optimally.

- Often computing q-values is harder that picking optimal actions!
  - loss가 낮다고 optimal action을 보장하는건 아님
- We could avoid learning values functions by directly learning agent's policy $$\pi_\theta(s|a)$$

***Why don't we just learn the policy here?***

- We want the optimal policy!
- We use cross entropy
  - It only operates in terms of $$\pi(a|s)$$
  - $$\pi(a|s)$$ : 특정 policy에서 state가 주어졌을때 actㅑon의 확률
  - When state is given, probability of action under policy 

### All kinds of Policies

#### Policies

- In general, two kinds of policies

  - Deterministic policy : $$ a  = \pi_\theta(a|s)$$

    - You learn an algorithm that takes your state and predicts one action.
    - So basically, it doesn't learn anything ***but for the number of action or maybe the value of action if it's continuous.***

  - Stochastic policy : $$a \sim \pi_\theta(a|s)$$

    - You can learn a probabilistic distributions.

    - So you can learn to predict the probabilities of taking each possible action.

    - e.g. Rock-scissor-paper game

      - If you're playing against your recent opponent, is to pick all possible actions at random.

        - If you only have one action, ***if you're using deterministic policy, the opponent is going to adapt and always show the item that will beat your current policy.***

        - The stochastic policy will be able to converge to a probability of one over three, 

          in this case, one-third. -> Much better than deterministic policy

    - Don't need epsilon-greedy policy

      - Because it already decides randomly(Stochastic policy)

How did you find the probabilistic policy in case of continuous(More then 1-dimensions) actions?

- Use normal distribution(Categorical, normal, mixture of normal, whatever)

#### Two approaches

- Values based:
  - Learn value function $$Q_\theta(s,a)$$ or $$V_\theta(s)$$
    - The value based methods rely on, first, learning some kind of value function, V or Q or whatever. 
  - Infer prefer $$\pi(a\s) = \#[a = \arg \max_a Q_\theta(s,a)] $$
    - If you have all perfect Q values, then you can simply find the optimal ones, the maximum Q function in this particular state, and this would be your optimal action.
    - However, if you don't know you have some error in Q values, your policy would be sub-optimal.
- Policy based:
  - Policy based methods don't rely on thess things.
    - Explicitly learn policy $$\pi_\theta(s,a)$$ or $$\pi_\theta(s)) \rightarrow a$$
  - Implicitly maximize reward over policy
    - They try to explicitly learn probability or deterministic policy, and they adjust them to implicitly maximize the expected reward or some other kind of objective. 