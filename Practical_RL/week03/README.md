# Week3

## Model-based vs Model-free

Model-free setting:

- We don't know actual $$P(s', r |s, a)$$
- If we know V or Q -> We have optimal policy
- We can learn them with dynamic programming
  - What to learn?
  - Q(s,a)
  - Because V(s) is useless without P(s'|s,a)

## Monte carlo method

- Get all trajectories containing particular(s,a)
- Need full trajectory
- Iteration이 끝나야 학습 할 수 있음
- Less reliant on Markov property

## Temporal difference

- Remember we can improve Q(s,a) iteration

### Q-learning

- Only need a small section of the trajectory
- You can train your algorithm even before you've ended the first section

### SARSA

- Q-learning : find a shortest path
  - It does not guarantee maximized reward
- SARSA gets optional reward under correct exploration strategy
- Q-learning policy would be optional without exploration

## On-policy vs Off-policy

### On-policy

- Agent can pick actions
- Most obvious step
- Agent always follows his own policy

### Off-policy

- Agent cannot pick actions
- Learning with exploration playing without exploration
- Learning from expert (Expert is imperfect)
- Learning from session(recorded data)

>SARSA will only work on-policy?
>
>- No
>- When epsilon is close to zero(0), It will work like off-policy
>- Suboptimal 값을 선택 할 수 없게 하면 Q-learning과 같은 방식으로 동작
>- 만약 optimal 값이 주어지면 Q-learning과 같게 동작 함
>
>Cross entropy
>
>- Distribution을 추정 했을 떄 이 distribution이 실제 distribution과 얼마나 차이가 있는지 확인하는 방법(같으면 0)

## Experience replay

- Idea : Store several past interactions(s,a,r,s') train on random suboptimal
- Training curriculum
  - Play 1 step and recored it
  - play N random transitions to train
- Profit : You don't need tot re-visit same(s,a) many times to learn it

> ***Only works with off-policy algorithms!***
>
> - On-policy에서는 현재 state와 action을 기반으로 update하기 때문에 이전에 방문 했던 걸 저장 해놓고 업데이트 하는 방식인 experience replay 방식을 사용 할 수 없음
>   - 따라서 오직 Off-policy에서만 가능 함
