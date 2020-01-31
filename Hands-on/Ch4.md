# Ch4 The Cross-Entropy Method

DQN이나 Actor-Critic보다는 덜 유명하지만 Cross-Entropy 나름대로 장점이 있음

- Simplicity
- Good convergence

## Taxonomy of RL methods

The cross-entropy method falls into the ***model-free and policy-based*** category of methods.

- Model-free or Model-based
  - Model-free
    - Method doesn't build a model of the environment or reward
    - It just directly connects observations to actions (or values that are related to actions)
    - In other words, the agent takes current observations and does some computations on them, and the result is the action that it should take.
    - 에이전트는 현재의 관측을 기반으로 행동 함
    - 현재 연구가 가장 활발하게 이루어지고 있는 분야임.
    - 앞으로 언급할 방법들은 모두 model-free 방법.
  - Model-based
    - Methods try to predict what the next observation and/or reward will be.
    - Based on this prediction, the agent is trying to choose the best possible action to take, very often making such predictions multiple times to look more and more steps into the future.
    - 에이전트는 미래의 결과를 예측하고 행동함
    - 보드게임 같은 룰이 엄격하게 정해진 환경에서 사용되는 방식
- Policy-based or Value-based
  - Policy-based
    - Policy-based methods are directly approximating the policy of the agent
    - That is, what actions the agent should carry out at every step.
    - Policy is usually represented by probability distribution over the available actions.
    - Policy는 distribution over action
      - 마코프 프로세스에서 선택지 고르는 것 생각
      - 하나 하나의 선택지가 policy
    - Policy-based는 agent의 policy를 바로 추정 함(Policy로 결정되는 action의 probability 고려)
  - Value-based
    - Instead of the probability of actions, the agent calculates the value of every possible action and chooses the action with the best value. 
    - Action의 확률을 고려하는게 아니라 value를 고려 함
- On-policy or Off-policy
  - On-policy
    - 현재 policy를 통해서만 학습 가능
  - Off-policy
    - It has the ability of the method to learn on old historical data
    - 과거의 데이터 또는 사전에 정의된 data를 이용해 학습 할 수 있음

Cross-entropy : ***model-free, policy-based and on-policy***

- The core of the cross-entropy method is to throw away bad episodes and train on better ones.

### Limitation of cross-entropy

- For training, our episode have to be finite and, preferably short
  - 짧고 유한한 길이의 에피소드만 학습 가능함
- The total reward for the episodes should have enough variability to separate good episodes from bad ones
  - 좋은 에피소드와 나쁜 에피소드를 구분 하기 위해 다양한 total reward가 존재해야 함
  - 예를 들어 에피소드가 끝날 때까지 얻을 수 있는 reward가 0과 1뿐이라면 cross-entropy 사용 불가능함
- There is no intermediate indication about whether the agent has succeeded or failed
  - 중간에 에이전트가 목표에 도달 했는지 못했는지 알 수가 없음

#### Tweak for cross-entropy

- Larger batches of played episodes
- Discount factor applied to reward
- Keeping 'elite' episode for a longer time
- Decrease learning rate
- Much longer training time