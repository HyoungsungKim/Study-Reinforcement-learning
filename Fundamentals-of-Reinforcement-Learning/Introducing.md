# Reinforcement Learning 

Week 1

***Reinforcement learning is different from supervised learning,*** the kind of learning studied in most current research in the field of machine learning.

- *Supervised learning* is learning from a training set of labeled examples provided by a knowledgable external supervisor.
- 감독 학습은 외부의 지능이 있는 감독에 의해 라벨링된 트레이닝 셋 예제로부터 학습 함
  - Each example is a description of a situation together with a specification***—the label—***of the correct action the system should take to that situation, which is often to identify a category to which the situation belongs.
  - The object of this kind of learning is for the system to extrapolate(추론하다), or generalize, its responses so that it acts correctly in situations not present in the training set.
  - This is an important kind of learning, ***but alone it is not adequate for learning from interaction.*** In interactive problems it is often impractical to obtain examples of desired behavior that are both correct and representative of all the situations in which the agent has to act. In uncharted(지도에 표시되어 있지 않은) territory—where one would expect learning to be most beneficial—***an agent must be able to learn from its own experience.***
- 지도학습은 라벨링 된 데이터에 적절하게 동작하지 않음
  - 에이전트는 경험으로부터 학습 할 수 있어야 함

***Reinforcement learning is also different from what machine learning researchers call unsupervised learning,*** which is typically about finding structure hidden in collections of unlabeled data.

- 강화학습은 라벨링 되지 않는 데이터의 집합으로부터 특징을 찾는 비지도 학습과도 다름
- The terms supervised learning and unsupervised learning would seem to exhaustively classify machine learning paradigms, but they do not.
- Although one might be tempted to think of reinforcement learning as a kind of unsupervised learning because it does not rely on examples of correct behavior, ***reinforcement learning is trying to maximize a reward signal instead of trying to find hidden structure.***
- 강화학습은 숨겨진 구조를 찾으려 노력하는게 아니라, 보상 신호를 최대화 하기 위해 노력 함
- Uncovering structure in an agent’s experience can certainly be useful in reinforcement learning, but by itself does not address the reinforcement learning problem of maximizing a reward signal.

***We therefore consider reinforcement learning to be a third machine learning paradigm,*** alongside supervised learning and unsupervised learning and perhaps other paradigms.

- 따라서 우리는 강화학습을 세번째 머신러닝 패러다임으로 고려 함.

One of the challenges that arise in reinforcement learning, and not in other kinds of learning, is the ***trade-off between exploration(탐험) and exploitation(이용).***

- ***To obtain a lot of reward, a reinforcement learning agent must prefer actions that it has tried in the past and found to be effective in producing reward.***
- ***But to discover such actions, it has to try actions that it has not selected before.*********
  - 보상을 얻기위해 에이전트는 지금까지 해왔던 action을 시도하고 더 효과적인 보상을 찾음(***Exploration(이용)***)
  - 하지만 더 큰 리워드를 얻기 위해서는 지금까지 하지 않았던 곳을 ***Exploitation(탐험)***해야 함
  - Exploration만 하면 Exploitation이 부족해지고, Exploitation을 하면 Exploration이 부족해짐
- The agent has to exploit what it has already experienced in order to obtain reward, but it also has to explore in order to make better action selections in the future. The dilemma is that neither exploration nor exploitation can be pursued exclusively without failing at the task.
- On a stochastic task, each action must be tried many times to gain a reliable estimate of its expected reward. The exploration–exploitation dilemma has been intensively studied by mathematicians for many decades, yet remains unresolved.
  - 어떻게 나눠야 하는지 아직 풀리지 않은 문제
- Another key feature of reinforcement learning is that ***it explicitly considers the whole problem of a goal-directed agent interacting with an uncertain environment.*** 

Reinforcement learning takes the opposite tack(방향), starting with a complete, interactive, goal-seeking agent.

- All reinforcement learning agents have explicit goals, can sense aspects of their environments, and can choose actions to influence their environments.
  - Moreover, it is usually assumed from the beginning that the agent has to operate despite significant uncertainty about the environment it faces.
- When reinforcement learning involves planning, it has to address the interplay between planning and real-time action selection, as well as the question of how environment models are acquired and improved. ***When reinforcement learning involves supervised learning, it does so for specific reasons that determine which capabilities are critical and which are not.***
- ***For learning research to make progress, important subproblems have to be isolated and studied,*** but they should be subproblems that play clear roles in complete, interactive, goal-seeking agents, even if all the details of the complete agent cannot yet be filled in.

## Elements of Reinforcement Learning

Beyond the agent and the environment, one can identify four main sub elements of a reinforcement learning system:

- A ***policy***
  - ***A policy defines the learning agent’s way of behaving at a given time.***
  - Roughly speaking, ***a policy is a mapping from perceived states of the environment to actions to be taken when in those states.***
  - It corresponds to what in psychology would be called a set of stimulus–response rules or associations. In some cases the policy may be a simple function or lookup table, whereas in others it may involve extensive computation such as a search process.
  - ***The policy is the core of a reinforcement learning agent in the sense that it alone is sufficient to determine behavior.***
  - In general, ***policies may be stochastic,*** specifying probabilities for each action
- A ***reward signal***
  - ***A reward signal defines the goal of a reinforcement learning problem.***
  - On each time step, the environment sends to the reinforcement learning agent a single number called the reward. ***The agent’s sole objective is to maximize the total reward it receives over the long run.***
  - The reward signal thus defines what are the good and bad events for the agent. 
  - ***The reward signal is the primary basis for altering the policy;*** if an action selected by the policy is followed by low reward, then the policy may be changed to select some other action in that situation in the future.
  - In general, ***reward signals may be stochastic functions of the state of the environment and the actions taken.***
- A ***value function***
  - ***A value function specifies what is good in the long run.***
  - Roughly speaking, the ***value of a state*** is the total amount of reward an agent can expect to accumulate over the future, starting from that state.
    - Whereas ***rewards determine the immediate, intrinsic desirability of environmental states***
    - ***Values indicate the long-term desirability of states after taking into account the states that are likely to follow and the rewards available in those states.***
      - Rewards는 즉각적으로 얻을 수 있는 바람직한 상황을 결정함
      - Value는 계정을 보상을 얻을 수 있는 states로 데려간 뒤 장기적인 바람직한 상황을 가르킴
    - To make a human analogy(비유),
      - Rewards are somewhat like pleasure (if high) and pain (if low)
      - Whereas values correspond to a more refined and far-sighted judgment of how pleased or displeased we are that our environment is in a particular state.
      - 사람으로 비유 했을떄 Rewards 즐거움이나 고통을 나타내고,
      - Values는 우리의 환경이 특정 states일 때 조금 더 기쁨이나 슬픔의 정제되고 장기적인 판단을 나타냄
  - ***Rewards are in a sense primary(주요 감각), whereas values, as predictions of rewards, are secondary.***
    - Without rewards there could be no values, and the only purpose of estimating values is to achieve more reward.
      - Rewards 없이 values는 존재 할 수 없음
      - Values의 목적은 rewards 이기 때문에
    - Nevertheless, it is values with which we are most concerned when making and evaluating decisions. Action choices are made based on value judgments.
      - 그럼에도 불구하고 우리는 values를 가장 중요하게 고려해야 함. action은 value를 기반으로 판단하기 때문에
    - We seek actions that bring about states of highest value, not highest reward, because these actions obtain the greatest amount of reward for us over the long run.
      - 우리는 높은 reward가 아니라 높은 value를 추구해야함. 결국 이게 가장 높은 reward를 얻는 방법이기 때문에
    - Unfortunately, ***it is much harder to determine values than it is to determine rewards.***
      - 불행하게도 values를 결정하는게 reward를 결정하는 것 보다 훨씬 어려움
      - Rewards are basically given directly by the environment
      - But values must be estimated and re-estimated from the sequences of observations an agent makes over its entire lifetime.
      - Values는 계속되는 시도를 통해서 얻어야 함
      - ***In fact, the most important component of almost all reinforcement learning algorithms we consider is a method for efficiently estimating values.*** 
- Optionally, a ***model*** of the environment
  - ***Model of the environment.***
    - This is something that mimics the behavior of the environment, or more generally, that allows inferences to be made about how the environment will behave.
    - For example, given a state and action, the model might predict the resultant next state and next reward. ***Models are used for planning, by which we mean any way of deciding on a course of action by considering possible future situations before they are actually experienced.***
    - 모델이 있으면 실수를 경험으로 배우지 않고 더 좋은 방향으로 나아갈 수 있음
    - ***Methods for solving reinforcement learning problems that use models and planning are called model-based methods,*** as opposed to simpler model-free methods that are explicitly trial-and-error learners—viewed as almost the opposite of planning. In Chapter 8 we explore reinforcement learning systems that simultaneously learn by trial and error, learn a model of the environment, and use the model for planning. Modern reinforcement learning spans the spectrum from low-level, trial-and-error learning to high-level, deliberative planning.

## 1.4 Limitations and Scope

Most of the reinforcement learning methods we consider in this book are structured around estimating value functions, but it is not strictly necessary to do this to solve reinforcement learning problems.

- For example, solution methods such as genetic algorithms, genetic programming, simulated annealing, and other ***optimization methods never estimate value functions.***
- Value 측정을 최적화 하지 못하면 강화 학습으로 해결 할 수 없음

These methods apply multiple static policies each interacting over an extended period of time with a separate instance of the environment.

- The policies that obtain the most reward, and random variations of them, ***are carried over to the next generation of policies, and the process repeats.***
  - We call these ***evolutionary methods*** because their operation is analogous to the way biological evolution produces organisms with skilled behavior even if they do not learn during their individual lifetimes.
- ***If the space of policies is sufficiently small, or can be structured*** so that good policies are common or easy to find—or if a lot of time is available for the search—***then evolutionary methods can be effective.*** In addition, evolutionary methods have advantages on problems in which the learning agent cannot sense the complete state of its environment.

***Our focus is on reinforcement learning methods that learn while interacting with the environment, which evolutionary methods do not do.***

- Methods able to take advantage of the details of individual behavioral interactions can be much more efficient than evolutionary methods in many cases.
- Evolutionary methods ignore much of the useful structure of the reinforcement learning problem:
  - They do not use the fact that the policy they are searching for is a function from states to actions;
  - They do not notice which states an individual passes through during its lifetime, or which actions it selects.
  - In some cases this information can be misleading (e.g., when states are misperceived), but more often it should enable more efficient search.
    - Although evolution and learning share many features and naturally work together, ***we do not consider evolutionary methods by themselves to be especially well suited to reinforcement learning problems and, accordingly, we do not cover them in this book.***

# Chapter 2 Multi-Armed Bandit

***The most important feature distinguishing reinforcement learning from other types of learning is that it uses training information that evaluates the actions taken rather than instructs by giving correct actions.***

강화학습이 다른 학습 방법과 비교 했을 때 구별되는 특징은, 강화학습에서 사용되는 트레이닝 정보는 옳은 행동이라고 지도 받은 정보를 사용하는 것이 아니라 행동을 평가한 정보를 사용 함

- This is what creates the need for active exploration, for an explicit search for good behavior.
  - Purely evaluative feedback indicates how good the action taken was, but not whether it was the best or the worst action possible.
    - Evaluative feedback은 행동이 얼마나 좋은지 나타내지만 그것이 최고인지 최악인지는 보이지 않는다
  - Purely instructive feedback, on the other hand, indicates the correct action to take, independently of the action actually taken. This kind of feedback is the basis of supervised learning, which includes large parts of pattern classification, artificial neural networks, and system identification. 
    - Instructive feedback은 옳은 행동을 가르키고, 실제로 행한 행동에는 독립적이다.

In their pure forms, these two kinds of feedback are quite distinct: evaluative feedback depends entirely on the action taken, whereas instructive feedback is independent of the action taken.

##  2.1 A k-armed Bandit Problem

Consider the following learning problem. You are faced repeatedly with a choice among k different options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. Your objective is to maximize the expected total reward over some time period, for example, over 1000 action selections, or time steps.

- This is the original form of the k-armed bandit problem
- 어떻게 이익을 최대화 할 것인가?

In our k -armed bandit problem, each of the k actions has an expected or mean reward given that that action is selected; let us call this the value of that action. We denote the action selected on time step $$t$$ as $$A_t$$ , and the corresponding reward as $$R_t$$ . The value then of an arbitrary action $$a$$, denoted $$q_*(a)$$, is the expected reward given that a is selected:
$$
q_*(a) \doteq \mathbb{E} [R_t | A_t = a]
$$

- $$A_t = a$$가 주어졌을 때 $$R_t$$의 기댓값을 $$q_*(a)$$ 라 정의 한다
  - Time step t에서 a를 action 했을 때 평균 reward
- $$A_t$$ : time step t에서 선택된 action
- $$R_t$$ : time step t에서 한 action으로 얻을 수 있는 Reward

If you knew the value of each action, then it would be trivial to solve the k -armed bandit problem: you would always select the action with highest value.

- 만약 각 action의 값을 알고 있따면 k-armed bandit problem을 해결하는건 매우 쉬움

***We assume that you do not know the action values with certainty,*** although you may have estimates. We denote the estimated value of action a at time step t as $$Q_t(a)$$. We would like $$Q_t (a)$$ to be close to $$q_*(a)$$.

- Time step t에서 action a를 했을 때 측정 값을 $$Q_t(a)$$ 라고 정의 함
- $$Q_t(a)$$를 $$q_t(a)$$에 가깝도록 하는 것이 목표
  - 왜 최댓값이 아니라 평균값에 가깝게 하는게 목표지...?

***If you maintain estimates of the action values, then at any time step there is at least one action whose estimated value is greatest.***

만약 측정을 계속한다면 적어도 하나의 최댓값을 찾을 수 있고, 이 값을 찾을 수 있는 action을 greedy action이라고 함

- We call these ***the `greedy actions`***.
  - ***When you select one of these actions, we say that you are exploiting*** your current knowledge of the values of the actions.
    - 만약 greedy action을 선택했다면 이것은 exploiting(이용) 중이라고 말 할 수 있음
  - ***If instead you select one of the `nongreedy actions`, then we say you are exploring,*** because this enables you to improve your estimate of the nongreedy action’s value.
    - 만약 nongreedy action을 선택했다면 한다면 exploring(탐험)이라고 할 수 있음
- Exploitation is the right thing to do to maximize the expected reward on the one step, but exploration may produce the greater total reward in the long run. 
  - If you have many time steps ahead on which to make action selections, then it may be better to explore the nongreedy actions and discover which of them are better than the greedy action.
  - Reward is lower in the short run, during exploration, but higher in the long run because after you have discovered the better actions, you can exploit them many times. ***Because it is not possible both to explore and to exploit with any single action selection, one often refers to the “conflict” between exploration and exploitation.***
  - Exploiting은 현재 step에서 최대의 reward를 얻을 수 있지만, 장기적으로 봤을때 최대의 reward를 보장하지 않음
  - Conflict of exploitation and exploration

In this book we do not worry about balancing exploration and exploitation in a sophisticated way; we worry only about balancing them at all.

## 2.2 Action-value Methods

We begin by looking more closely at methods for estimating the values of actions and for using the estimates to make action selection decisions, which we collectively call ***action-value methods.***

- Recall that the true value of an action is the mean reward when that action is selected.
- One natural way to estimate this is by ***averaging***(Not expectation) the rewards actually received:

$$
Q_t(a) \doteq \dfrac{sum\text{ }of\text{ }rewards\text{ }when\text{ }a\text{ }taken\text{ }prior \text{ }to\text{ }t}{number\text{ }of\text{ }times\text{ }a\text{ }taken\text{ }prior\text{ }t}
= 
\dfrac{\sum^{t-1}_{i = 1} R_i \cdot \bold{1}_{A_i=a}}{\sum^{t-1}_{i=1}\bold{1}_{A_i = a}}
$$

- ***Action이 a 인것들의 평균 reward***
- t 이전에 취한 a(action)의 rewards의 합 / t 이전에 취한 a(action)의 숫자
- $$\bold{1}_{predicate}$$ denotes the random variable that is $$\bold{1}$$ if `predicate` is true and $$\bold{0}$$ if it is false.
  - If the denominator is zero, then we instead define $$Q_t(a)$$ as some default value, such as
    0
  - 만약 분모가 0이라면 $$Q_t(a)$$ 를 0으로 정의
- As the denominator goes to infinity, by the law of large numbers, $$Q_t(a)$$ converges to $$q_*(a)$$
  - 만약 문보가 무한대로(실험 횟수가 무한대로 가면) 가면 큰 수의 법칙에 의해 $$Q_t(a)$$는 $$q_*(a)$$로 수렴
  - 큰 수의 법칙 : 실험 반복 횟수가 많아지면 결과의 평균은 기댓값에 수렴 함
- We call this the ***sample-average*** method for estimating action values because each estimate is an average of the sample of relevant rewards.
  - This is just one way to estimate action values

***The simplest action selection rule is to select one of the actions with the highest estimated value.***

- That is, one of the greedy actions as defined in the previous section. If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly. We write this greedy action selection method as

$$
A_t \doteq \underset{a}{argmax}Q_t(a)
$$

- $$Q_t(a)$$가 max가 되는 argument a

Greedy action selection always exploits current knowledge to maximize immediate reward;

- It spends no time at all sampling apparently inferior actions to see if they might really be better.
  - Simple alternative is to behave greedily most of the time, but every once in a while, say with small probability $$\epsilon$$, instead select randomly from among all the actions with equal probability, independently of the action-value estimates.
    - Greedy하게 행동하다가 가끔 매우 낮은 확률로 무작위로 선택 함
  - ***We call methods using this near-greedy action selection rule $$\epsilon$$-greedy methods.*** An advantage of these methods is that, in the limit as the number of steps increases, every action will be sampled an infinite number of times, thus ensuring that all the $$Q_t(a)$$ converge to $$q_*(a)$$. 
    - 시험 횟수가 많아질수록(무한대로 갈 수록) 시험 평균이 전체 평균에 수렴 함
  - This of course implies that the probability of selecting the optimal action converges to greater than 1- $$\epsilon$$,  that is, to near certainty. These are just asymptotic guarantees, however, and say little about the practical effectiveness of the methods.
    - 만약 $$\epsilon$$ 이 0이면 계속 greedy하게 행동 하기 때문에 optimal action converges to 1

## 2.3 The 10-armed Testbed

- 입실론 너무 낮으면 optimal action 찾는데 오래 걸림
  - 적당히 낮으면 최적 액션 빨리 찾지만 이게 suboptimal일 수도 있음
  - 너무 낮을때 최적 액션 찾는데 걸리는 시간 크지만 suboptimal 피할 수 있음
- 입실론이 클수록 빠르고 saturation 됨
- 입실론이 1/2 인 노말 가우시안에서 평균 리워드 1

***The greedy method performed significantly worse in the long run because it often got stuck performing suboptimal actions.*** 

- Greedy method는 suboptimal에 빠질 수도 있기때문에 시간이 갈수록 성능이 안좋아짐

The lower graph shows that the greedy method found the optimal action in only approximately one-third of the tasks. In the other two-thirds, its initial samples of the optimal action were disappointing, and it never returned to it.

- 약 33%의 범위에서만 optimal action 탐색 함

***The $$\epsilon$$-greedy methods eventually performed better***

- Because they continued to explore and to improve their chances of recognizing the optimal action.
- The $$\epsilon$$ = 0.1 method explored more, and ***usually found the optimal action earlier***, but it never selected that action more than 91% of the time.
- The $$\epsilon$$  = 0.01 method improved more slowly, ***but eventually would perform better than the $$\epsilon$$ = 0.1 method on both performance measures shown in the figure.*** It is also possible to reduce $$\epsilon$$ over time to try to get the best of both high and low values.

## 2.4 Incremental Implementation

We now turn to the question of how these averages can be computed in a computationally efficient manner, in particular, with constant memory and constant per-time-step computation.

- The obvious implementation would be to maintain a record of all the rewards and then perform this computation whenever the estimated value was needed.

## 2.5 Tracking a Non-stationary Problem

***We often encounter reinforcement learning problems that are effectively non-stationary.***

- In such cases it makes sense to give more weight to recent rewards than to long-past rewards.
- One of the most popular ways of doing this is to use a constant step-size parameter.
- 확률이 계속 변하는 상황의 강화 학습 문제를 자주 마주침
  - 이런 경우 최근의 리워드에 더 높은 가중치를 주는 것이 더 자명 함
  - 유명한 방법 중 하나는 상수 스텝 사이즈 파라미터를 쓰는 것

Sometimes it is convenient to vary the step-size parameter from step to step.

- Let $$\alpha_n(a)$$ denote the step-size parameter used to process the reward received after the `n`th selection of action a.
- As we have noted, the choice $$\alpha_n(a) =n_1$$ results in the sample-average method, which is guaranteed to converge to the true action values by the law of large numbers.
  - 무수히 많은 샘플을 뽑아서 평균을 구하면 기댓값으로 수렴(큰 수의 법칙)
  - But of course convergence is not guaranteed for all choices of the sequence $${\alpha_n(a)}$$.
- A well-known result in stochastic approximation theory gives us the conditions required to assure convergence with probability 1:
  - 하지만 모든 연속적인 선택이 수렴을 보장 하는건 아님
  - 확실한 보장을 위해서는 몇가지 조건이 필요 함

$$
\sum^\infty_{n=1}\alpha_n(a) = \infty \text { and } \sum^\infty_{n=1}\alpha^2_n(a) < \infty
$$

- The first condition is required to guarantee that the steps are large enough to eventually overcome any initial conditions or random fluctuations.
- The second condition guarantees that eventually the steps become small enough to assure convergence.
- ***Although sequences of step-size parameters that meet these convergence conditions are often used in theoretical work, they are seldom used in applications and empirical research.***
- 이론적인 연구에서는 이 조건들이 자주 사용 되지만 어플리케이션이나 경험적인 연구에서는 드물게 사용 됨.

## 2.6 Optimal Initial Values

All the methods we have discussed so far are dependent to some extent on the initial action-value estimates, $$Q_1 (a)$$.

- In the language of statistics, these methods are biased by their initial estimates.
- ***For the sample-average methods,*** the bias disappears once all actions have been selected at least once
- But ***for methods with constant $$\alpha$$,*** the bias is permanent, though decreasing over time.
- 샘플을 뽑아서 평균을 구하는 방식에서는 모든 액션이 적어도 한번은 선택 된다면 초기값(바이어스)가 사라짐
- Non-stationary 한 방법에서는 constant(바이어스)가 점점 감소하지만 영원히 남아있음
  - 1보다 작은 수의 지수이기 때문에 점점 감소함
- In practice, this kind of bias is usually not a problem and can sometimes be very helpful.
  - The downside is that the initial estimates become, in effect, a set of parameters that must be picked by the user, if only to set them all to zero.
  - The upside is that they provide an easy way to supply some prior knowledge about what level of rewards can be expected.

***Initial action values can also be used as a simple way to encourage exploration.***

- Suppose that instead of setting the initial action values to zero, as we did in the 10-armed testbed, we set them all to +5.
  - Recall that the $$q_*(a)$$ in this problem are selected from a normal distribution with mean 0 and variance 1.
- An initial estimate of +5 is thus wildly optimistic. ***But this optimism encourages action-value methods to explore.*** Whichever actions are initially selected, the reward is less than the starting estimates;
  - 초기값이 어떤 값으로 설정되는간에 리워드는 시작 할 때보다 낮기 때문에 explore을 장려 할 수 있음
  - 더 좋은 reward를 못찾으면 처음 상태로 돌아와서 반복하면 되기 때문에
- The learner switches to other actions, being “disappointed” with the rewards it is receiving. The result is that all actions are tried several times before the value estimates converge. The system does a fair amount of exploration even if greedy actions are selected all the time.
- $$\epsilon$$-greedy 방식 보다 더 빠르게 최적 optimal을 찾을 수도 있음(더 적극적으로 탐험 하기 때문에)
  - Initially, the optimistic method performs worse because it explores more, but eventually it performs better because its exploration decreases with time.
- ***We call this technique for encouraging exploration optimistic initial values.***
- 처음에는 초기값이 효과가 있지만 시간이 지날수록 효과가 떨어짐
- Non-stationary 할 때에는 잘 맞지 않음

***We regard it as a simple trick that can be quite effective on stationary problems,*** but it is not well suited to non-stationary problems

- Because its drive for exploration is inherently temporary. ***If the task changes, creating a renewed need for exploration, this method cannot help.***

## 2.7 Upper-Confidence-Bound Action Selection

$$\epsilon$$ - greedy action selection forces the non-greedy actions to be tried, but indiscriminately(무비판적으로), with no preference for those that are nearly greedy or particularly uncertain.

- ***It would be better to select among the non-greedy actions according to their potential for actually being optimal,*** taking into account both how close their estimates are to being maximal and the uncertainties in those estimates.
- One effective way of doing this is to select actions according to

$$
A_t = \underset{a}{argmax} \left\lbrack Q_t(a) + c\sqrt \frac{\ln t}{N_t(a)} \right \rbrack
$$

- $$Q_t(a)$$ : Exploitation part
  - Other term is exploration part
- where $$\ln t$$ denotes the natural logarithm of t (the number that e $$\approx$$ 2.71828 would have to be raised to in order to equal t)
- $$N_t(a)$$ denotes ***the number of times that action a has been selected prior to time t (the denominator in (2.1)),*** 
- The number $$c > 0$$ controls the ***degree of exploration.***(Amount of exploration)
  - ***If $$N_t(a) = 0$$, then `a` is considered to be a maximizing action.***
- ***The idea of this upper confidence bound (UCB) action selection is that the square-root term is a measure of the uncertainty or variance in the estimate of `a’s` value.***
- The quantity being max’ed over is thus a sort of upper bound on the possible true value of action `a`, with `c` determining the confidence level.
- Each time `a` is selected the uncertainty is presumably reduced:
  - $$N_t(a)$$ increments, and, as it appears in the denominator, the uncertainty term decreases.
  - 각 시간마다 a는 불확실성이 감소한다고 추정하고 선택 됨
  - $$N_t(a)$$는 증가하고, 불확실성을 나타내는 square root는 감소 함
  - 쉽게 생각하면 시도를 많이 할 수록 불확실성은 감소 함
- On the other hand, each time an action other than `a` is selected, `t` increases but $$N_t(a)$$ does not;
  - Because `t` appears in the numerator, the uncertainty estimate increases.
  - `t`는 분자에 나오기 때문에 `t`가 증가하면 불확실성은 증가 함
  - 쉽게 생각하면 시간 측정이 길어지면 불확실성이 증가함
- The use of the natural logarithm means that the increases get smaller over time, but are unbounded;
  - All actions will eventually be selected, but actions with lower value estimates, or that have already been selected frequently, will be selected with decreasing frequency over time.
  - 자연 로그를 사용하는 이유는 매우 천천히 증가하지만 발산 함.

UCB often performs well, as shown here, but is more difficult than $$\epsilon$$ - greedy to extend beyond bandits to the more general reinforcement learning settings considered in the rest of this book.

- ***One difficulty is in dealing with non-stationary problems;***
- ***Another difficulty is dealing with large state spaces.*** In these more advanced settings the idea of UCB action selection is usually not practical.
- UCB generally performs better than $$\epsilon$$ - greedy action selection, except in the first `k` steps, when it selects randomly among the as-yet-untried actions.

