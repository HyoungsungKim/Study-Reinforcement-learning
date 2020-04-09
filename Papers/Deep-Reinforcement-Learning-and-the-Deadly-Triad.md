# Deep Reinforcement Learning and the Deadly Triad

We know from reinforcement learning theory that temporal difference learning can fail in certain cases.

Expecially when these three properties are combined, learning can diverge with the value estimates becoming unbounded.

- Approximation
- Bootstrapping
- Off-policy

***However, several algorithms successfully combine these three properties, which indicates that there is at least a partial gap in our understanding***.  In this work, we investigate the impact of the deadly triad in practice

---

In the simplest form of TD learning, the immediate reward is added to the discounted value of the
subsequent state, and ***this is then used as a target to update the previous state’s value***.

- t+1 에서 얻은 값을 사용하여 t에서 얻은 값을 업데이트(backward)
- This means the value estimate at one state is used to update the value estimate of a previous state—this is called bootstrapping.
- Bootstrapping is also commonly used within policy-gradient and actor-critic methods to learn value functions.

When combining TD learning with function approximation, ***updating the value at one state creates a risk of inappropriately changing the values of other states***, including the state being bootstrapped upon.

- TD learning과 approximation을 같이 쓰면 업데이트 할 때 잘못된 값으로 업데이트 될 수도 있음
  - 예를 들어 부트스트래핑 같은 경우 초기 값이 영향을 줄 수도 있음
  - 타겟값을 얻는 네트워크가 업데이트 될 때 우연히 매우 안좋은 값이 들어가면 트레이닝 전체에 악영향을 줄 수가 있음.
- This is not a concern when the agent updates the values used for bootstrapping as often as they are used.
  - However, if the agent is learning off-policy, it might not update these bootstrap values sufficiently often(e.g Q-learning).
  - This can create harmful learning dynamics that can lead to divergence of the function parameters 
- But DQN solved these problems
  - Neural network
  - Experience buffer
  - One-step Q-learning as its learning algorithm, it relies on bootstrapping.
  - Despite combining all these components of the deadly triad, DQN successfully learns to play many Atari 2600 games.

In this paper, we conduct an empirical examination into when the triad becomes deadly. 

## The deadly triad in deep reinforcement learning

Why does DQN works?

### Deadly triad in deep reinforcement learning

- Bootstrapping
  - We can modulate the influence of boot strapping using multi-step returns.
  - Multi-step returns have already been shown to be beneficial for performance in certain variations of DQN
- Function approximation
  - We can modify the generalization and aliasing of the function approximation by changing the capacity of the function space.
  - ***We manipulated this component by changing the size of the neural networks***.
- Off-policy
  - we can change how off-policy the updates are by changing the state distribution that is sampled by experience replay.
  - ***In particular, we can do so by prioritising certain transitions over others***. 
  - Heavier prioritisation can lead to more off-policy updates.

## Building intuition

Off-policy can converge with a small modification to the function class.
$$
v(s_i) = w(\phi(s_i) + u)
$$

- Where us is also learnable.
- If we allow both `w` and `u` to be updated by TD, then values first seemingly diverge, then recover and converge to the optimum.

## Hypothesis

### Hypothesis 1 (Deep divergence)

- Unbounded divergence is uncommon when combining Q-learning and conventional deep reinforcement learning function space

### Hypothesis 2 (Target network)

- There is less divergence when bootstrapping on separate networks
- However, ***target networks do not suffice as a solution to the deadly triad***.
  - When such target networks are applied to the standard with linear function approximation, the weights still diverge, though the divergence is slowed down by the copying period.

### Hypothesis 3 (Overestimation)

- There is less divergence when correcting for overestimation bias
- Standard Q-learning and target Q-learning are known to suffer from an overestimation bias.
  - To prevent this, we can decouple the action selection from the action evaluation in the bootstrap target, by using $$v(s) = q_0(s, \underset{a}{argmax} \,q(s, a))$$.
  - This is known as double Q-learning.
- It has the benefits of reducing overestimation, but not those of using a separate target network for bootstrapping.

### Hypothesis 4 (Multi-step)

- Longer multi-step returns will diverge less easily
- We can use multi-step returns to reduce the amount of bootstrapping.
- If we use multi-step, then bootstrapping are happened less

### Hypotheses 5 (Capacity)

- Larger, more flexible networks will diverge less easily.
- If all values are stored independently in a function approximation, then divergence would not happen

### Hypothesis 6 (Prioritisation)

- Stronger prioritisation of updates will diverge more easily

## Evaluation

### Unbounded divergence in deep RL (Hypothesis 1)

The lack of unbounded divergence in any of the runs suggests that ***although the deadly triad can cause unbounded divergence in deep RL, it is not common when applying deep Q-learning and its variants to current deep function approximators***.

### Examining the bootstrap targets (Hypothesis 2, 3)

Q-learning exhibits by far the largest fraction of instabilities (61%);

- ***Target Q-learning and double Q-learning, which both reduce the issue of inappropriate generalization via the use of target networks, are the most stable***.
- Inverse double Q-learning, which addresses the over-estimation bias of Q-learning but does not benefit from the stability of using a target network, exhibits a moderate rate (33%) of soft-diverging runs.

These results provide strong support for both Hypothesis 2 and 3: ***divergence occurs more easily with overestimation biases, and when bootstrapping on the same network***.

Also ***when more learnable parameters are added to the canonical formulation: values initially grow very large, but then return to more accurate value estimates***.

### Examining the deadly triad (Hypothesis 4, 5 and 6)

- For all four bootstrap methods, ***there is a clear trend that a longer bootstrap length reduces the prevalence of instabilities***.
- For Q-learning: small networks exhibited less instabilities (53%) than larger network (67%).
- Increasing the prioritisation increases instability

## Discussion

A key result is that the instabilities caused by the deadly triad interact with statistical estimation issues induced by the bootstrap method used. As a result, ***the instabilities commonly observed in the standard epsilon-greedy regime of deep Q-learning can be greatly reduced by bootstrapping on a separate network and by reducing the overestimation bias***. 

In our experiments, there were strong performance benefits from longer multi-step returns and from larger networks. Interestingly, while longer multi-step returns also yielded fewer unrealistically high values, larger networks resulted in more instabilities, except when double Q-learning was used. We believe that the general learning dynamics and interactions between (soft)-divergence and control performance could benefit from further study.