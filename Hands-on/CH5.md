# Ch5 Tabular Learning and the Bellman Equation

- Q-learning : Powerful and flexibility
- Q-optimal을 이용해서 Q-learning을 함
- 여기서는 Q-optimal을 Q-learning처럼 설명하고 있음

## Value, state, and optimality

***Value는 value function의 결과이고, value function을 어떻게 정의하느냐에 따라 달라짐.***

In cross-entropy value is
$$
V(s) = \mathbb{E}[\sum_{t=0}^\infin r_t \gamma ^t]
$$
Value is always calculated in the respect of some policy that our agent follows.

- It is trivial

How can we get a optimal policy?

- Bellman equation

### The Bellman equation of optimality

When we act greedily, we do not only look at the immediate reward for the action, ***but at the immediate reward plus the long-term value of the state.***

- *Richard Bellman* proved that with that extension, our behavior will get the best possible outcome.
- In other words, it will be optimal. So, the preceding equation is called the **Bellman equation** of value (for a deterministic case):

What we need to do is to calculate the expected value for every action, instead of just taking the value of the next state.

Example)
$$
V_0(a = 1) = p_1(r_1 + \gamma V_1) + p_2(r_2 + \gamma V_2) + p_3(r_3 + \gamma V_3)
$$
or more formally,
$$
V_0(a) = \mathbb{E}_{a \sim S}[r_{s,a} + \gamma V_s] = \sum_{s \in S} p_{a, 0 \rightarrow s}(r_s,a + \gamma V_s)
$$

- Action이 정해져있는 경우
- $$p_{a,i \rightarrow j}$$ : It means action `a` is issued at state `i` and end up at `j`

For deterministic case
$$
V_0 = \max_{a \in A} \mathbb{E}_{s \sim S}[r_{s,a} + \gamma V_s] = \max_{a \in A} \sum_{s \in S}p_{a, 0 \rightarrow s}(r_{s,a} + \gamma V_s)
$$

- Action이 정해져 있지 않아서 최댓값을 반환하는 action을 선택 할 경우
- The interpretation is still the same:
  - The optimal value of the state is equal to the action, which gives us the maximum possible expected immediate reward plus discounted long-term reward for the next state
  - State의 최적 value는 가장 기댓값이 높은 action을 선택 할 때와 같음
  - 기댓값은 현재의 reward와 미래의 value의 합
- 특이한 점은, 다음 state의 value를 미리 알고 있다고 가정하고 업데이트 함.
- 위의 수식을 보면 $$V_0$$에서 `a`를 선택했을때 기대되는 value를 계산하는데 다음 state의 value가 사용 됨
  - 조금 이상해 보이지만 많이 사용 됨
  - 흠 뭐지...

### Value of action

We defined `value of state` $$V_s$$. Now We will define `value of action` $$Q_{s,a}$$

- 앞에서 정의한 value of state와 다른 value of state 그리고 value of action을 정의 할 것임
- Basically, it equals the total reward we can get by executing action `a` in state `s` and can be defined via $$V_s$$
- The whole family of methods called "Q-learning"
- It is more convenient in practice

$$
Q_{s,a} = \mathbb{E}_{s' \sim S}[r_{s,a} + \gamma V_s'] = \sum_{s' \in S}p_{a,s \rightarrow s'}(r_{s,a} + \gamma V_s')
$$

- `s'` is next state
- `r` is current reward
- 앞에서 정의한 value of state에서는 미래의 reward에 미래의 value를 더하는 방식이었음
- Value of action에서는 현재의 reward에 미래의 value 더하는 방식
- 이 value of action을 이용해서 새로운 value of state를 정의 함

$$
V_s = \max_{a \in A} Q_{s,a}
$$

- Value of state : Q(s,a)에 a를 바꿔가면서 넣을때 얻을 수 있는 가장 큰 value

Define Q(s,a) using Q(s,a)
$$
Q(s,a) = r_{s,a} + \gamma \max_{a' \in A} Q(s', a')
$$

- Q(s,a) : 현재의 reward에 다음 action을 통해 얻을 수 있는 가장 큰 expected value의 합으로 현재 state의 value 구함

## The value iteration method

1. Initialize all value as zero
2. Update using mdp and reward
3. Update for long time

Limitation

- Our state space should be discrete and small enough to perform multiple iterations over all states.
- We rarely know the transition probability for the actions and rewards matrix

## Q-learning for FrozenLake

The most obvious change is to our value table.

- In value iteration, we kept value of the state, so the key in the dictionary was just a state.
- Now we need to store values of the Q-function, which has two parameters
  - State
  - Action

Q-learning은 state와 action을 모두 저장하고 있음. 하지만 value iteration은 state만 저장해 놓고, action이 필요하면 계산해서 얻음

- 즉 Q-optimal은 dynamic programming 같은 느낌
- 따라서 수렴 속도는 Q-optimal이 빠르지만, 메모리 사용량이 큼
- Value iteration에서는 다음 state로 넘어갈때 확률(Probability of state transition)을 알고 있음
- 하지만 Q-learning에서는 확률을 모를때 사용 가능.
  - value iteration은 model-based, Q-optimal은 model-free에 사용 됨
  - value iteration은 주어진 확률을 기반으로 다음 value 계산 함
    - 즉 가장 높은 확률의 action을 선택하고 그에 따른 state의 value 계산
  - Q-optimal은 다음 state의 reward들을 보고  value 계산함. 확률은 모름
    - 다음 state의 value가 가장 높은 걸 선택
- [What is different between q-learning and value iteration](https://stackoverflow.com/questions/28937803/what-is-the-difference-between-q-learning-and-value-iteration)
- 처음에 V(s), q(s,a)를 0으로 초기화 하고 시작!
- [Sutton 2019] 책 63, 64 페이지 참고
  - Bellman optimality equation에서 v optimal은 unique solution을 가지고 있음(어떤 a 를 선택 할지)

## Recap: Bellman optimality equation

### Value iteration

- V(s) : value of state function of bellman optimality equation

$$
V(s) = \max_{a}[R^a_{s'} + \gamma  \sum_{s'} p(s' | s, a)V(s'))]
$$

- Value iteration이라는 느낌은 오지만 기댓값의 최댓값을 선택한다는 느낌은 잘 안드는듯...
- $$R^a_{s'}$$ : 현재 state s에서 다음 state s'으로 action a를 했을 때 ***reward의 기댓값*** -> R(s,a,s')

Other variation
$$
V(s) = \max_a[\sum_{s', r} p(s', r | s, a)(r + \gamma V(s'))]
$$

- Value iteration이라는 느낌은 잘 안들지만 기댓값의 최댓값을 선택한다는 느낌은 쉽게 듬
- $$r$$ : distribution을 통해 알고 있는 reward (Random variable)
- $$p(s', r | s, a)$$ : Probability of s' with r given s, a
- 기댓값이 최대인걸 선택

### Q-learning이라기 보다는 Q optimal인것 같은데...

- Q(s,a) : Value of state-action function of bellman optimality equation

Q optimal makes choosing optimal actions even easier.

Define Q(s,a)
$$
Q(s,a) = R_{s'}^a + \gamma \cdot \sum_{s'}[p(s'|s,a)\max_aQ(s',a')]
$$

- $$R^a_{s'}$$ : 현재 state s에서 다음 state s'으로 action a를 했을 때 ***reward의 기댓값*** -> R(s,a,s')

$$
Q(s,a) = \sum_{s', r}p(s', r|s,a)(r + \gamma \max_a Q(s',a')
$$

- $$r$$ : distribution을 통해 알고 있는 reward (Random variable)
- $$p(s', r | s, a)$$ : Probability of s' with r given s, a
- Q(s,a)의 optimal 구하는 공식임
- The action-value function effectively caches the results of all one-step-ahead searches.