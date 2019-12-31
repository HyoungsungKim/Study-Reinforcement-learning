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


## DQN: statistical issues

### Recap: Approximate Q-learning

- Image
  - Gradient descent

$$
w_{t+1} = w_t - \alpha \cdot \frac{\delta{L}}{\delta{w}}
$$

- model $$\theta = params$$ 
  - Objective

$$
L = (Q(s_t, a_t) - [r + \gamma \cdot \max_{a'} Q(s_{t+1}, a')])^2
$$

- $$Q(s,a_0),Q(s,a_1),Q(s,a_2)$$
  - Q-values update

$$
\hat{Q}(s_t, a_t) = r + \gamma \cdot \max_{a} \hat{Q}(s_{t+1}, a')
$$

### We have a problem

$$
s \rightarrow a \rightarrow s' \rightarrow \text{one of }Q(s', a_0), Q(s', a_1), Q(s', a_2)
$$

- For starters, you network will have some approximation error and due to gradient decent, you will have some perturbations(작은 변화) in those $$\text{Q of }s', a_0, a_1, a_2$$ just because network trains between the iteration, it's vary changes lightly.
  - $$Q(s', a_0), Q(s', a_1), Q(s', a_2)$$가 같지 않음
  - What you want to do is want to compute the value of the state, the maximum of action values.
    - $$\max{(Q(s', a_0), Q(s', a_1), Q(s', a_2))}$$
- Another issue here is that the outcomes are often stochastic, and therefore the Q of s prime, a and so on may vary as well.
  - 또 다른 이슈 중 하나는, 출력물은 통계적이기 때문에, $$\text{Q of }s', a$$ 등등의 값 역시 다양하게 나옴
  - Actual value(true value)는 고정

$$
\mathbb{E}_{s'}[\max_{a'}\hat{Q}(s_{t+1}, a')] \geq \max_{a'}[\mathbb{E}_{s'}(\hat{Q}(S_{t+1}, a'))]
$$

- 각 s와 a에 의한 결과로 나온 value들 중 최대 값을 뽑고 expectation 하면 expectation의 최대값보다 항상 크게 나옴
- 따라서 실제로 얻고 싶은 값보다 크게 나옴
- General idea is that, if you use this maximization over samples, you'll get something which is larger than what you actually want.
  - Therefore, for example if you have some particular states in which your value is filtrate, network is still trying to learn it. Then your network will be overoptimistic, it will over-appreciate(과대평가) being in this state although this only happened due to the statistical error.
    - 만약 에러에 의해 원하지 않은 결과가 나오더라도 네트워크는 이 결과를 학습하고, 이 state를 과대 평가 할 것임
    - So, this is the problem which causes actual DQN get as optimistic(낙관적) as it actually explodes.
  - So the Q-values become larger and larger over time on some games and sometimes they never get back. So, they are being optimistic all the time. 
    - 따라서 Q-value가 계속 커져서 네트워크는 모든 상황에 대하여 낙관적이게 됨.
- Solution : Double Q-learning

## Double Q-learning

***One popular solution to this problem of optimism is the so-called double Q-learning***

- Idea : Train two Q-functions separately, $$Q_1$$ and $$Q_2$$
- If you cannot trust one Q function, Double Q-learning could make you learn two of them, to train one another. 
- ***Those are independent*** estimates of the actual value function. 
  - In tabular double Q-learning, those are just two independent tables, and in deep Q network case, those are two neural networks that have several sets of weights
- What happens if you update them one after the other, and you use one Q function to train the other one, and vice versa. 

Objective for $$Q_1$$
$$
\hat{Q}_1(s_t,a_t) = r_t + \gamma \cdot Q_2(s_{t+1}, \arg\max_{a*} Q_1{s_{t+1}, a*})
$$

- $$Q_1$$을 최댓값으로 만드는 action을 $$Q_2$$의 action값으로 사용하고 이 결과를 $$Q_2$$ 업데이트에 사용

Objective for $$Q_2$$
$$
\hat{Q_2}(S_t, a_t) = r_t + \gamma \cdot Q_1(S_{t+1}, \arg\max_{a*} Q_2({s_{t+1}, a*}))
$$
If all Q functions are equal for example, then the maximum of say Q2 is going to be just basically a random action because of how noise works.

- If you take the expectation of Q value of random action from Q1, you'll get exactly the maximum of expectations in the limit of course, if you take all those samples. You do the same thing with Q2.
- Basically, you take the Q2 and you use Q1 to help it to update itself.
- So you maximize by one Q network and take the action value for this maximal action from the other one. 

### Algorithm

```pseudocode
Initialize Q1, Q2
Forever:
	sample s_t, a_t, r_t, s_{t+1} from env
	
	if flip_coin() == 'heads':
		Qhat1(s_t, a_t) = r_t + gamma * Q_2(s_{t+1}, argmax_{a*}Q1(s_{t+1}, a^*))
		Q1(s_t, a_t) <- Q1(s_t, a_t) + alpha[Qhat1(s_t, a_t) - Q1(s_t, a_t)]		
    else:
    	Qhat2(s_t, a_t) = r_t + gamma * Q_1(s_{t+1}, argmax_{a*}Q2(s_{t+1}, a^*))
		Q2(s_t, a_t) <- Q2(s_t, a_t) + alpha[Qhat2(s_t, a_t) - Q2(s_t, a_t)]		
```

Drawback : It takes twice time to train

- Need smart way!

### Smart way

Now consider DQN

- Neural net policy
- Experience replay
- Target networks

Idea 1 : train two networks separately

- Drawback : It takes twice time to train

Solution : ***Use the target network***

Idea 2 : use target network instead of Q2

- ***The older snapshot of your network(Q^{old})*** as the source of independent randomness as the other Q network.
- ***So you only train your Q1,*** maximizing and taking the action value, you just take the action value of your old Q network, the target network that corresponds to an action optimal on the reoccurring Q network.
- 하나의 네트워크만 트레이닝 함. 이 네트워크의 이전의 snapshot을 Q2처럼 사용함
- 업데이트를 100,000 iteration 간격으로 하면 dependency는 더욱 낮아짐

Classic DQN(with target network)
$$
\hat{Q}(s_t, a_t) = r_t + \gamma \cdot \max_{a*} Q^{old}(s_{t+1}, a^*)
$$
Double DQN(with target network) 
$$
\hat{Q}(s_t, a_t) = r_t + \gamma \cdot Q^{old}(s_{t+1}, \arg\max_{a*}Q(s_{t+1}, a^*))
$$

### Recap

DQN
$$
\hat{Q}(s_t, a_t) = r_t + \gamma \cdot \max_{a*} \hat{Q}(s_{t+1}, a')
$$
Double DQN
$$
\hat{Q}(s_t, a_t) = r_t + \gamma \cdot \hat{Q}(s_{t+1}, \arg\max_{a'} \hat{Q}(s_{t+1}, a'))
$$
Double DQN with target network
$$
\hat{Q} = r_t + \gamma \cdot Q^{old}(s_{t+1}, \arg\max_{a'} \hat{Q}(s_{t+1}, a'))
$$

## More DQN tricks

***We have a problem #2***

### Advantage

In atari breakout Q(s, left) and Q(s, right)

- What you might have noticed is the fact that most of the time, especially if the ball in breakout is on the opposite side of the game field, you'll have Q values for actions more or less(more or less : 거의, 약) on the same value.
  - 공이 벽돌에 맞고 튕긴후 정 반대의 위치로 갈때 Q-value가 가장 크게 차이 남.
  - 하지만 대부분의 시간동안에는 left와 right 모두 비슷한 값을 가짐
- This is because in this state, one action, even a stupid action won't change anything.
  - 떨어지는 공의 반대방향으로 이동하는 경우도 있음	
  - Even if you make one bad move while the ball is on the other side of the field, you'll still have plenty of time to adjust to go the right way and fix the issue.
    - 왜냐하면 어리석은 행동을 하더라고 공이 밑으로 떨어지기 전까지 조정하기 위한 충분한 시간이 있기 때문에
  - Therefore, all Q values are more less the same. There's this common parts which all of those action values have. 
- Maybe the ball is approaching your player,  you're going to move at just the right position to catch it and so it bounces off. If you don't, you'll just miss the ball and lose one life. ***So, there are rare cases where those Q values are highly different.***
  - 만약 공이 플레이어쪽으로 향한다면이 때 공을 받기 위해 공쪽으로 향할 것임.
  - 이럴 경우 Q value의 값이 매우 크게 차이남

This brings us to another architecture. ***It's called the dueling deep Q network.***

Decomposition : $$Q(s,a) = V(s) + A(s,a)$$

- V(s) only depend on state
- Neutral A(s,a) - Advantage function
  - The intuition here is the advantage is how much you're action value differs from the state value

Where Q(s,a) are action values, as usual

- V(s) are state values$$(V^*(s) or V_\pi(s))$$
- A(s,a) is the "advantage"
  - Some of the advantage of the suboptimal action is going to be negative.
  - $$A(s,a) = Q(s,a) - V(s)$$
  - Suboptimal을 negative 값으로 줌
  - 만약 value가 하나는 100 하나는 1일 경우 둘다 positive이기 때문에 100을 선택한다는 보장이 없었지만, advantage를 사용 할 경우에 1을 선택하면 -100을 value에서 잃게 되어 적게 잃는 쪽을 선택하게 함.

Usual DQN

- Image -> model -> Q(s, a0), Q(s, a1), Q(s, a2)

Dueling DQN

- Image -> model -> V(s), A(s, a0), A(s, a1), A(s, a2) -> Q(s, a0), Q(s, a1), Q(s, a2)

***The only difference is that you may define those advantages and value functions differently.***

### Dueling DQN

Estimating A(s,a)

Option 1: 최댓값
$$
A(s, a_i) = nn(s)[i]-\max_{j}nn(s)[j]
$$
Option 2: 평균값
$$
A(s, a_i) = nn(s)[i]- \frac{1}{|A|} \sum_j nn(s)[j]
$$
Where nn(s) is neural network activation for advantage  (|A| is cardinality? maybe...)

#### Problem

- The problem is that sometimes it takes to actually make those bold steps, make a few suboptimal, similarly suboptimal action to discover something new which is not near the optimal policy but which is a completely new policy in a way that it approaches the entire decision process.
  - Suboptimal 값과 유사하지만 optimal 값과는 거리가 매우 먼 policy를 발견 할 수도 있음
- The Epsilon greedy strategy is very unlikely to discover this.
  - Epsilon greedy strategy 방법으로는 이걸 발견하기 힘듬
- It is very prone to local optimal convergence. 
  - 즉 local optimal로 수렴하는 경우가 발생 할 수 있음.
- Solution : Bootstrap DQN

### Issue of exploration: Bootstrap DQN

- Each head predict all q-values
- State -> shared network -> haed1, head2, ... head K
- Why? ***Make exploration great again!***
- How? Maintain K separately heads(for example, last 2 layers) and shared body weights

Here is that you have to train a set of key, say 5 or 10 Q value predictors, ***that all share the same main body parts, so they all have same convolutional layers.***

- The way they are trained is at the begin of every episode, you pick on them hard, so you throw a dice and you pick one of those key has.
- 주사위를 던저서(무작위로) 나오는 값으로 Q value predictor 선택 함
- Then you follow its actions and you train the weights of the head and the weights of the corresponding body. Then you basically do this thing for the entire episode.
- This way, you want the heads to going to be slightly better but also your features are going to change as well. Then at the beginning of the next episode, you pick another head. So you re-throw dice again and see what happens and then you follow the policy of this new head. 
- You can expect them to differ, and this difference is going to be systematic. So, they wont just take suboptimal actions(dependency를 줄이는 방법인가...?)

#### Simplified algorithm

- Pick random head
- Play full game
- Train head & body
- Repeat

