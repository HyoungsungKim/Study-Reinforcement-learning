# Ch5 Tabular Learning and the Bellman Equation

- Q-learning : Powerful and flexibility

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

- Our state space should be discrets