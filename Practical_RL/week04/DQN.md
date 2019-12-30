# DQN

## DQN: Bird's eye view

- First successful application learning
  - Directly from raw visual inputs(same as humans do)
  - In a wide variety of environments(Atari games)
- Deep convolution network
- No hand-designed features
- Q-learning with stability tricks
- Epsilon-greedy exploration

## Architecture of DQN (Based on paper)

The architecture was composed of three convolution layers, one on top of another, and one dense layer on top of the last convolution one. 

- Q-values
- Dense
- Conv3
- Conv2
- Conv1

Well, in any practical application, the first thing you should consider to do is to reduce the complexity of the task as much as you could before starting any attempts to solve this task.

- In the reinforcement learning world, it mostly means reducing the space of states as much as you could.
  - Reducing the state's base in the context of Atari games basically consist in gray-scaling, down-sampling, and cropping.
  - All these operations were performed in the original approach. At first, a colored Atari screen image which is 210 wide and 160 pixels high was gray-scaled and downscaled to 110 by 84. 
  - Then it was cropped to a square region of 84 by 84 pixels. And in the end, an interesting transformation was reformed.
    - Initial image (210 * 160 * 3)
    - Gray-scale and downsample (110 * 84 * 1)
    - Crop (84 * 84 * 1)
    - Stack 4 most recent frames (84 * 84 * 4)
  - After each action performed, the four layer denser of observation was changed. The oldest frame was thrown away, and the most recent one that is a new observation was added.
  - ***There is no pooling layer***
    - As you might know, poolings, especially max poolings are ubiquitous(아주 흔한) in neural network architectures designed to work with images.
    - ***But in this architecture, you don't see any pooling layers.***
      - In general, pooling layers don't have any parameters but they consume computational resources and more importantly, they require time. 
      - pooling layer는 파라미터가 존재하지 않지만 컴퓨팅 자원과 시간을 소모함
      - 하지만 pooling layer와 같은 dimensional reduction을 유지하기 위해 convolution layer에서 stride사용 함.

## Stack 4 frames = fourth-order Markov assumption

- Markov assumption

$$
p(r_{t+1}, s_{t+1}|s_0, a_0, ... r_t, s_t, a_t) = p(r_{t+1}, s_{t+1} | s_t, a_t)
$$

- Fourth-order Markov assumption

$$
p(r_{t+1}, s_{t+1} | s_0, a_0, ... , r_t, s_t, a_t) = p(r_{t+1}, s_{t+1} | s_t, a_t, ..., s_{t-3},a_{t-3})
$$

- These tricks works well for velocity, acceleration, and various things that depends on very small amount of recent images. 
  - However, this trick fails for anything that depends on number of frames longer than four. 

## DQN fixes some of the instability problems

***DQN has some tweaks(비틀다) against three problems***

1. Sequential correlated data
   - May hurt convergence and performance
   - Solution : Experience replay
2. Instability of data distribution due to policy change
   - Policy may diverge and / or oscillate
   - Solution : Target networks
3. Unstable gradient
   - Absolute value of $$Q(s,a)$$ may vary much across states
   - Solution : Reward clipping

### (1) Sequential data -> Experience replay

Semi-gradient update for approximate Q-learning
$$
w \leftarrow w + \alpha[R_{t+1} + \gamma \max_{a} \hat{q}(S_{t+1}, a, w) - \hat{q}(S_t, A_t, w)]\nabla \hat{q}(S_{t}, A_{t}, w)
$$

1. Store tuples (S, A, R, S') in a pool
   - We only need (S, A, R, S') to make such update
2. Sample tuple(s) from pool at random
3. Update the model of Q-func
4. Act epsilon-greedy w.r.t Q-func
5. Add new samples to pool
6. Go to 2

Very simple, but why does it help?

- If the pool is sufficiently large, then we effectively de-correlate the data by taking, in the update, different pieces of possibly different trajectories.
  - ***Such experience replay is possible only for off-policy learning.***
  - That is so because ***on-policy models implies that only new, fresh data coming from policy with the latest parameters is considered for learning.***
  - Off-policy에서만 사용 가능 - On-policy에서는 임의로 데이터를 선택 할 수 없음.(Policy에 따라야 함)
  - And our current parameters are different to those used to generate the old samples.
- Experience replay is a very powerful technique. It is used almost everywhere it can be used because of its properties.
- It smooths out learning and prevents oscillations or divergence in the parameters.
- Experience replay not only helps against correlations, but also increases sample efficiency and reduces variance of updates. ***It also is very easy to use for training on distributed architectures. That is, on cluster of machines.***
- Another interesting observation is that the experience replay could be viewed as an ***analog of sample based model of the world.*** This view partially explains the effectiveness of this technique.
  - Experience replay는 sample based model의 연속성(analog)으로도 볼 수 있음
- However, experience replay is not completely free of disadvantages.
  - ***It is very memory intensive.***
  - Also, the random sampling from a pool is not the most efficient method of sampling.
    - We might want to more frequently select the latest experience than the old one.
    - 랜덤 샘플링 방식이 좋은 샘플링은 아님
    - 왜냐하면 오래된 샘플보다는 최근에 추가된 샘플을 추가하고 싶은데 랜덤으로 선택하면 최근에 추가된게 선택된다는 보장이 없기 때문에
- Helps against correlations
- Increase sample efficiency
- Reduce variance of updates
- Computations are easy to parallel
- Analog of sample based model of the world
  - Memory intensive (DQN stored ~ 1 million interactions)
  - Random sampling from pool could be improved
  - Older interactions were obtained under weaker policy

### (2) policy oscillation -> Target networks

***The instability problem is not eliminated completely with experience replay.***

- Targets still depend on the parameters and error made in any target immediately propagates to other estimates.
- This dependence of target on parameters could easily broke the learning by introducing oscillations and positive feedback cycles. 
  - For example, ***we might want to update more the parameters responsible for low Q value estimate that corresponds to high target value.***
  - But by doing so and because of sharing the approximation parameters, we might also increase the value of the target. 
  - Target값에 맞춰 낮은 Q-value를 업데이트 하면 approximation parameters를 공유하기 때문에 target값도 같이 높아져서 Q-value가 target에 수렴 할 수 없음
- The idea against such positive feedback loops proposed in the DQN paper was both simple and effective. ***Let just split the parameters of Q values and targets*** from the parameters of currently learning Q-function approximation

Unwanted source of instability 

- Targets are functions of parameters
- Errors in estimates propagate into other estimate

***Standard*** Q-learning update
$$
w \leftarrow w + \alpha[R_{t+1} + \gamma \max_{a} \hat{q}(S_{t+1}, a, w) - \hat{q}(S_t, A_t, w)]\nabla \hat{q}(S_t, A_t, w)
$$
Q-learning with the ***target networks***
$$
w \leftarrow w + \alpha[R_{t+1} + \gamma \max_{a} \hat{q}(S_{t+1}, a, w^-) - \hat{q}(S_t, A_t, w)]\nabla \hat{q}(S_t, A_t, w)
$$
Copy parameters to target network

- In fact, they can be updated in either of the two ways.
  - (hard) Copy $$w$$ to $$w^-$$ once in a while
    - That is, once in a while, say, every 10,000 time stamps, assign the parameters of current Q network to parameters of target network. 
    - You can think about this type of updates as about creating snapshots of Q network, and updating these snapshots from time to time. 
  - (soft) Maintain $$w^-$$ as moving average of $$w$$
    - Update the weights at every time step but use a very small update rate. 
    - In the simplest form, this idea corresponds to the parameter of the target network being the exponential moving average of the Q-network parameters.

### (3) Unstable gradient -> Reward clipping

***This problem is in part inherent to reinforcement learning because of average changes of action value function.***

- Don't know the scale of reward beforehand

- Don't want numeric problems with large Q-values

- ***Clip the reward to [-1, 1]***

  - less peaky(뼈아픈, 창백한) Q-values
  - good gradients
  - Drawback(단점)) : cannot distinguish between good and great
    - ***Unlike the previous two, it wasn't adopted by future researchers because of its drawbacks.***
    - Nevertheless, sometimes it may be helpful. 

  