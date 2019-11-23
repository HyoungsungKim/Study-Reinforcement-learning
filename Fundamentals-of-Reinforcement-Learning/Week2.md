# Finite Markov Decision Processes

In this chapter we introduce the formal problem of finite Markov decision processes, or finite MDPs, which we try to solve in the rest of the book.

- This problem involves evaluative feedback, as in bandits, but also an associative aspect—choosing different actions in different situations.
  - MDPs are a classical formalization of sequential decision making, ***where actions influence not just immediate rewards, but also subsequent situations, or states, and through those future rewards.***
- Thus MDPs involve delayed reward and ***the need to trade-off immediate and delayed reward.***
  - Whereas in bandit problems we estimated the value $$q_*(a)$$ of each action `a`
  - In MDPs we estimate the value $$q_*(s, a)$$ of each action `a` in each state `s`, or we estimate the value $$v_*(s)$$ of each state given optimal action selections.
    - Bandit 문제에서는 action `a`에서 $$q_*(a)$$를 측정 함
    - MDP에서는 State `s`에서 action `a`의 $$q_*(s,a)$$를 측정 함 또는  최적 action이 주어졌을 때 $$v_*(s)$$를 계산 함.
  - These state-dependent quantities are essential to accurately assigning credit for long-term consequences to individual action selections.
  - ***MDPs are a mathematically idealized form of the reinforcement learning problem for which precise theoretical statements can be made.***
    - We introduce key elements of the problem’s mathematical structure, such as returns, value functions, and Bellman equations.

## 3.1 The Agent–Environment Interface

***MDPs are meant to be a straightforward framing of the problem of learning from interaction to achieve a goal.***

- The learner and decision maker is called the **agent.**
- The thing it interacts with, comprising(포함하는) everything outside the agent, is called the **environment.**
- 학습하는 사람과 결정을 내리는 사람을 agent라고 함
- 이것 들과 상호 작용 하는 외부의 agent를 environment라고 함
  - ***These interact continually,*** the agent selecting actions and the environment responding to these actions and presenting new situations to the agent.
  - 이들은 끊임없이 상호 작용하고, 에이전트는 선택 함
  - The environment also gives rise to rewards, special numerical values that the agent seeks to maximize over time through its choice of actions.

More specifically, the agent and environment interact at each of a sequence of discrete time steps, $$t = 0, 1, 2, 3, . . ..$$

- At each time step `t`, the agent receives some representation of the environment’s state, $$S_t\in S$$, and on that basis selects an action, $$A_t \in A(s)$$.
- One time step later, in part as a consequence of its action, the agent receives a numerical reward, $$R_{t+1}\in R \subset \mathbb{R}$$, and finds itself in a new state, $$S_{t+1}$$ .
- `t`시간에 에이전트는 어떤 state를 받고, 이걸 기반으로 $$A_t$$를 선택 함
  - 시간이 주어지면 이 시간에 해당하는 state의 action 선택
- 그 다음 시간에, reward를 받고 다음 state에서 이걸 찾음

S -> A -> R -> S -> A -> R -> ...
$$
p(s',r | s, a) \doteq  Pr\{S_t=s',R_t=r | S_{t-1} = s, A_{t-1} = a\}
$$

- State s와 Action a가 주어졌을 때 다음 state가 s'이고 reward가 r일 확률
- The function p defines the dynamics of the MDP

$$
\sum_{s'\in S}\sum_{r \in R} p(s',r|s,a) = 1, \text{for all }s\in S, a \in A(s)
$$

***Some of what makes up a state could be based on memory of past sensations or even be entirely mental or subjective.***

- MDP는 기억성을 가지고 있음
- 따라서 모든 이전 states를 고려 하지 않고, 직접적으로 영향을 주는 states만 고려 하면 됨

***We do not assume that everything in the environment is unknown to the agent.***

- For example, the agent often knows quite a bit about how its rewards are computed as a function of its actions and the states in which they are taken.
- ***But we always consider the reward computation to be external to the agent*** because it defines the task facing the agent and thus must be beyond its ability to change arbitrarily.
  - In fact, in some cases the agent may know everything about how its environment works and still face a difficult reinforcement learning task, just as we may know exactly how a puzzle like Rubik’s cube works, but still be unable to solve it.

The agent–environment boundary represents the limit of the agent’s absolute control, not of its knowledge.

- In practice, the agent–environment boundary is determined once one has ***selected particular states, actions, and rewards,*** and thus has identified a specific decision making task of interest.

***The MDP framework is a considerable abstraction of the problem of goal-directed learning from interaction.***

- MDP framework는 목표 방향 학습 문제의 추상화로 여겨질수도 있다.
- 하지만 목표 방향 학습의 문제는 3개의 signal passing으로 제거 할 수 있다.
- It proposes that whatever the details of the sensory, memory, and control apparatus, and whatever objective one is trying to achieve, ***any problem of learning goal-directed behavior can be reduced to three signals passing back and forth between an agent and its environment:***
  - One signal to represent the choices made by the agent (the actions),
  - One signal to represent the basis on which the choices are made (the states),
  - One signal to define the agent’s goal (the rewards).
- This framework may not be sufficient to represent all decision-learning problems usefully, but it has proved to be widely useful and applicable. 

## 3.2 Goals and Rewards

In reinforcement learning, the purpose or goal of the agent is formalized in terms of a special signal, ***called the reward, passing from the environment to the agent.***

- Informally, the agent’s goal is to maximize the total amount of reward it receives. This means maximizing not immediate reward, but cumulative reward in the long run.

Although formulating goals in terms of reward signals might at first appear limiting, in practice it has proved to be flexible and widely applicable.

- Reward signals 이라는 용어로 목표를 공식화 하는 방식은 초반에 한게가 나타났지만, 실제로는 유연하고 넓게 응용 가능하다고 증명 되어 왔다.
- For example, to make a robot learn to walk, researchers have provided reward on each time step proportional to the robot’s forward motion. In making a robot learn how to escape from a maze, the reward is often 1 for every time step that passes prior to escape;
  - This encourages the agent to escape as quickly as possible. To make a robot learn to find and collect empty soda cans for recycling, one might give it a reward of zero most of the time, and then a reward of +1 for each can collected.
  - One might also want to give the robot negative rewards when it bumps into things or when somebody yells at it. For an agent to learn to play checkers or chess, the natural rewards are +1 for winning, 1 for losing, and 0 for drawing and for all nonterminal positions.

In particular, the reward signal is not the place to impart(전하다, 주다) to the agent prior knowledge about how to achieve what we want it to do.

- Reward signal은 에이전트에게 어떻게 달성 할 수 있는지에 대한 지식을 주는 것이 아님
- For example, a chess-playing agent should be rewarded only for actually winning, not for achieving subgoals such as taking its opponent’s pieces or gaining control of the center of the board. If achieving these sorts of subgoals were rewarded, then the agent might find a way to achieve them without achieving the real goal.
  - For example, it might find a way to take the opponent’s pieces even at the cost of losing the game. The reward signal is your way of communicating to the robot what you want it to achieve, not how you want it achieved.
  - Subgoals들이 있으면 main goal에 집중하지 않고 subgoals에만 머물러 있는 경우가 발생 함.

## 3.3 Returns and Episodes

***We have said that the agent’s goal is to maximize the cumulative reward it receives in the long run.*** How might this be defined formally?

- If the sequence of rewards received after time step `t` is denoted $$R_{t+1}, R_{t+2}, R_{t+3}, . . .,$$ then what precise aspect of this sequence do we wish to maximize?
- ***In general, we seek to maximize the expected return,*** where the return, denoted $$G_t$$ , is defined as some specific function of the reward sequence. In the simplest case the return is the sum of the rewards:

$$
G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... + R_{T},
$$

***When the agent–environment interaction breaks*** naturally into subsequences, which we call `episodes`, such as plays of a game, trips through a maze, or any sort of repeated interaction.

- Each episode ends in a special state called the `terminal state`, followed by a reset to a standard starting state or to a sample from a standard distribution of starting states.
- Even if you think of episodes as ending in different ways, such as winning and losing a game, the next episode begins independently of how the previous one ended.
  - ***Thus the episodes can all be considered to end in the same terminal state, with different rewards for the different outcomes.***
  - Tasks with episodes of this kind are called ***episodic tasks***.
    - In episodic tasks ***we sometimes need to distinguish the set of all nonterminal states,*** denoted `S`, from the set of all states plus the terminal state, denoted $$S^+$$ .
    - The time of termination, `T` , is a random variable that normally varies from episode to episode.

***On the other hand, in many cases the agent–environment interaction does not break naturally into identifiable episodes, but goes on continually without limit.***

- For example, this would be the natural way to formulate an on-going process-control task, or an application to a robot with a long life span.
- We call these ***continuing tasks.*** The return formulation (3.7) is problematic for continuing tasks because the final time step would be $$T = \infty$$, and the return, which is ***what we are trying to maximize, could itself easily be infinite.***
  - (For example, suppose the agent receives a reward of +1 at each time step.) Thus, in this book we usually use a definition of return that is slightly more complex conceptually but much simpler mathematically.

***The additional concept that we need is that of discounting.***

- According to this approach, the agent tries to select actions so that the sum of the discounted rewards it receives over the future is maximized.
- In particular, it chooses $$A_t$$ to maximize the expected discounted return:

$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t + 3} + ... = \sum^\infty_{k= 0} \gamma^k R_{t + k + 1} \text{ (3.8)} 
$$

- Where $$\gamma$$ is a parameter, $$0 \leq \gamma \leq  1$$, called the ***discount rate***.

***The discount rate determines the present value of future rewards***:

- A reward received $$k$$ time steps in the future is worth only $$\gamma^{k-1}$$ times what it would be worth if it were received immediately.
- ***If $$ \gamma < 1$$***, the infinite sum in (3.8) has a finite value as long as the reward sequence $${R_k}$$ is bounded.
- ***If $$\gamma = 0$$ , the agent is “myopic(근시안적인)” in being concerned only with maximizing immediate rewards:***
  - Its objective in this case is to learn how to choose $$A_t$$ so as to maximize only $$R_{t+1}$$.
  - 이 방법은 $$R_{t+1}$$을 최대로 하는 $$A_t$$를 구하기 위해 사용 됨
- If each of the agent’s actions happened to influence only the immediate reward, not future rewards as well, then a myopic agent could maximize (3.8) by separately maximizing each immediate reward.
  - ***But in general, acting to maximize immediate reward can reduce access to future rewards so that the return is reduced.***
  - 감마는 시간이 지날 수록 0으로 수렴하기 때문에 근시안 적인 전략을 가진 에이전트의 리워드는 점점 감소하게 됨
- ***As $$\gamma$$ approaches 1, the return objective takes future rewards*** into account more strongly; the agent becomes more farsighted.
- $$\gamma$$가 1에 가까워 질 수록 에이전트는 더욱 장기적인 전략을 가짐