# Finite Markov Decision Processes part 2

***k bandit problem에서 정의했던 q랑 헷갈리면 안됨***

- 여기서 q를 다시 정의 함

## 3.4 Unified Notation for Episodic and Continuing Tasks

In this book we consider sometimes one kind of problem and sometimes the other, but often both. It is therefore useful to
establish one notation that enables us to talk precisely about both cases simultaneously.

- To be precise about episodic tasks requires some additional notation. Rather than one long sequence of time steps, ***we need to consider a series of episodes, each of which consists of a finite sequence of time steps.***
  - We number the time steps of each episode starting anew from zero.
  - Therefore, we have to refer not just to $$S_t$$ , the state representation at time $$t$$, but to $$S_{t,i}$$ , the state representation ***at time $$t$$ of episode $$i$$ (and similarly for $$A_{t,i}$$ , $$R_{t,i}$$, $$\pi_{t,i}$$ , $$T_i$$ , etc.).***
    - However, it turns out that when we discuss episodic tasks we almost never have to distinguish between different episodes.
    - ***We are almost always considering a particular single episode, or stating something that is true for all episodes.*** Accordingly, in practice we almost always abuse notation slightly by dropping the explicit reference to episode number. That is, we write $$S_t$$ to refer to $$S_{t,i}$$ , and so on.

## 3.5 Policies and Value Functions

Example) Chess - episodic MDP

- State : Positions of all the pieces on the board
- Actions : legal moves 
- Termination occurs when the game ends in either a win, loss, or draw

Almost all reinforcement learning algorithms involve estimating ***value functions***—functions of states (or of state–action pairs) that ***estimate how good it is for the agent to be in a given state*** (or how good it is to perform a given action in a given state).
$$
\pi(s) = a \text{ (Deterministic policy)}
$$


- The notion of “how good” here is defined in terms of future rewards that can be expected, or, to be precise, in terms of expected return.
  - Of course the rewards the agent can expect to receive in the future depend on what actions it will take.
  - Accordingly, value functions are defined with respect to particular ways of acting, called ***policies***.

Formally, a ***policy is a mapping from states to probabilities of selecting each possible action***.

- If the agent is following policy $$\pi$$ at time $$t$$, then  $$\pi(a|s)$$ is the probability that $$A_t = a$$ if $$S_t = s$$. Like $$p, \pi$$ is an ordinary function; the “|” in the middle of $$\pi(a|s)$$ merely reminds that it defines a probability distribution over $$a \in A(s)$$ for each $$s \in \bold{S}$$.
- Reinforcement learning methods specify how the agent’s policy is changed as a result of its experience.

***The value function of a state `s` under a policy $$\pi$$, denoted $$v_\pi(s)$$, is the expected return when starting in `s` and following $$\pi$$ thereafter.*** For MDPs, we can define $$v_\pi$$ formally by
$$
v_\pi(s) \doteq \mathbb{E}[G_t|S_t=s] = \mathbb{E}_\pi [\sum^\infty_{k=0}\gamma^kR_{t+k+1} | S_t = s]
$$

- We call the function $$v_\pi$$ the ***state-value function*** for policy $$\pi$$ 
- State가 주어졌을때 value

Similarly, ***we define the value of taking action $$a$$ in state $$s$$ under a policy $$\pi$$, denoted $$q_{\pi}(s,a)$$, as the expected return starting from $$s$$, taking the action $$a$$, and thereafter following policy $$\pi$$***:
$$
q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t | S_t=t, A_t=a] = \mathbb{E}_\pi[\sum^\infty_{k=0}\gamma^k R_{t+k+1} | S_t = s ,A_t = a] \\
= \sum_{s'}\sum_rp(s',r|s,a)[r + \gamma \mathbb{E}_\pi[G_{t+1}|S_{t+1} = s'] \text{ (1)}\\
= \sum_{s'}\sum_rp(s',r|s,a)[r + \gamma v_\pi(s')] \\
= \sum_{s'}\sum_rp(s',r|s,a)[r + \gamma \sum_{a'} \pi(a'|s) \mathbb{E}_\pi[G_{t+1}|S_{t+1} = s', A_{t+1} = a'] \text{ (2)}\\
= \sum_{s'}\sum_{r}p(s',r|s,a)[r + \gamma \sum_{a'}\pi(a'|s')q_\pi(s',a')] \text{ (3)}
$$

- Recursive form으로 바꾸려면 state가 필요함
- q는 이미 action이 주어져 있음
- 따라서 (1)식 reward부분에 state를 넣어줌
- We call $$q_\pi$$ the ***action-value function*** for policy $$\pi$$.
- State와 action이 주어졌을 때 value

The value functions $$v_\pi$$ and $$q_\pi$$ can be estimated from experience.

For example, if an agent follows policy $$\pi$$ and maintains an average, for each state encountered, of the actual returns that have followed that state, then the average will converge to the state’s value, $$v_\pi(s)$$ , as the number of times that state is encountered approaches infinity.

- If separate averages are kept for each action taken in each state, then these averages will similarly converge to the action values, $$q_\pi(s, a)$$.
  - 각각의 state에서의 평균이 각각의 평균동안 유지된다면, 이 평균들은 action value에 수렴함
  - We call estimation methods of this kind ***Monte Carlo methods*** because they involve averaging over many random samples of actual returns.

***Bellman equation for $$v_\pi$$.* **
$$
v_\pi(s) \doteq \mathbb{E}_\pi[G_t|S_t=s] \\
= \sum_{a \in A}\pi(a|s)*q_\pi(s,a)\\
= \sum_a \pi(a|s)\sum_{s',r}p(s',a|s,a)[r+\gamma v_\pi(s')]  \text{ for all } s \in \bold{S}
$$


- 앞에 sum 2개 : probability, 그 뒤 항은 random variable 
- Exp = prob * random variable
- ***It expresses a relationship between the value of a state and the values of its successor states.***
- Think of looking ahead from a state to its possible successor states, as suggested by the diagram to the right. Each open circle represents a state and each solid circle represents a state–action pair.
- Starting from state s, the root node at the top, the agent could take any of some set of actions—three are shown in the diagram—based on its policy $$\pi$$. From each of these, the environment could respond with one of several next states, $$s'$$ (two are shown in the figure), along with a reward, $$r$$, depending on its dynamics given by the function $$p$$.
- ***The Bellman equation averages over all the possibilities, weighting each by its probability of occurring.***
  - It states that the value of the start state must equal the (discounted) value of the expected next state, plus the reward expected along the way.(2개의 sum 이후에 나온 term)

## 3.6 Optimal Policies and Optimal Value Functions

Solving a reinforcement learning task means, roughly, finding a policy that achieves a lot of reward over the long run.

- For finite MDPs, we can precisely define an optimal policy in the following way.
  - Value functions define a partial ordering over policies.
  - ***A policy $$\pi$$ is defined to be better than or equal to a policy $$\pi'$$ if its expected return is greater than or equal to that of $$\pi'$$ for all states.***
  - In other words, $$\pi \geq \pi'$$ if and only if $$v_\pi(s) \geq v_{\pi'}(s)$$ for all $$s \in S$$.
    - ***There is always at least one policy that is better than or equal to all other policies.*** This is an optimal policy.
    - Although there may be more than one, we denote all the optimal policies by $$\pi_*$$.
- They share the same state-value function, called the ***optimal state-value function***, denoted $$v_*$$, and defined as

$$
v_*(s) \doteq \underset{\pi}{max} \text{ }v_\pi(s) \text{ for all }s \in \bold{S}
$$

- $$v_\pi(s)$$ 중에서 가장 큰 값
- value를 가장 크게 하는 policy를 가졌을 때 value...?

Optimal policies also share the same ***optimal action-value function***, denoted $$q_*$$, and defined as
$$
q_*(s,a) \doteq \underset{\pi}{max} \text{ } q_\pi(s,a) \text{ for all } s \in \bold{S} \text{ and } a \in A(s)
$$

- $$q_\pi(s,a)$$ 중게 가장 큰 값

For the state–action pair $$(s, a)$$, this function gives the expected return for taking action $$a$$ in state $$s$$ and thereafter following an optimal policy. Thus, we can write $$q_*$$ in terms of $$v_*$$ as follows:
$$
q_*(s,a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1})|S_t = s, A_t = a]
$$
Because $$v_*$$ is the value function for a policy, it must satisfy the self-consistency condition given by the Bellman equation for state values.

- Because it is the optimal value function, however, $$v_*$$’s consistency condition can be written in a special form without reference to any specific policy.
- This is the Bellman equation for $$v_*$$, or the ***Bellman optimality equation***.
- Intuitively, the Bellman optimality equation expresses the fact that ***the value of a state under an optimal policy must equal the expected return for the best action from that state***:

For finite MDPs, the Bellman optimality equation for $$v_*$$ has a unique solution.

- ***The Bellman optimality equation is actually a system of equations, one for each state***,
  - so if there are n states, then there are n equations in n unknowns.
  - If the dynamics p of the environment are known, then in principle one can solve this system of equations for $$v_*$$ using any one of a variety of methods for solving systems of nonlinear equations. One can solve a related set of equations for $$q_*$$.

Once one has $$v_*$$ , it is relatively easy to determine an optimal policy.

- For each states, there will be one or more actions at which the maximum is obtained in the Bellman optimality equation.
- Any policy that assigns nonzero probability only to these actions is an optimal policy. You can think of this as a one-step search.
  - If you have the optimal value function, $$v_*$$ , then ***the actions that appear best after a one-step search will be optimal actions***.
  - Another way of saying this is that any policy that is greedy with respect to the optimal evaluation function $$v_*$$ is an optimal policy.
    - Consequently, it describes policies that select actions based only on their short-term consequences.
    - The beauty of $$v_*$$ is that if one uses it to evaluate the short-term consequences of actions—specifically, the one-step consequences—then a greedy policy is actually optimal in the long-term sense in which we are interested
    - Because $$v_*$$ already takes into account the reward consequences of all possible future behavior. By means of $$v_*$$​, the optimal expected long-term return is turned into a quantity that is locally and immediately available for each state. Hence, a one-step-ahead search yields the long-term optimal actions.

Having $$q_*$$ makes choosing optimal actions even easier.

- With $$q_*$$, the agent does not even have to do a one-step-ahead search:
  - For any state s, it can simply find any action that maximizes $$q_*(s, a)$$.
  - The action-value function effectively caches the results of all one-step-ahead searches. It provides the optimal expected long-term return as a value that is locally and immediately available for each state–action pair.
  - Hence, at the cost of representing a function of state–action pairs, instead of just of states, the optimal action-value function ***allows optimal actions to be selected without having to know anything about possible successor states and their values***, that is, without having to know anything about the environment’s dynamics.

***Explicitly solving the Bellman optimality equation provides one route to finding an optimal policy, and thus to solving the reinforcement learning problem.***

- However, this solution is rarely directly useful. It is akin to an exhaustive search, looking ahead at all possibilities, computing their probabilities of occurrence and their desirabilities in terms of expected rewards.

- This solution relies on at least three assumptions that are rarely true in practice:

  (1) we accurately know the dynamics of the environment;

  (2) we have enough computational resources to complete the computation of the solution;

  (3) the Markov property.

- For the kinds of tasks in which we are interested, one is generally not able to implement this solution exactly because various combinations of these assumptions are violated.

  - For example, although the first and third assumptions present no problems for the game of backgammon(주사위 놀이), the second is a major impediment(장애).
    - Because the game has about $$10^{20}$$ states, it would take thousands of years on today’s fastest computers to solve the Bellman equation for $$v_*$$ , and the same is true for finding $$q_*$$ .

- In reinforcement learning one typically has to settle for approximate solutions.

Many different decision-making methods can be viewed as ways of approximately solving the Bellman optimality equation.

- For example, heuristic search methods can be viewed as expanding the right-hand side of (3.19) several times, up to some depth, forming a “tree” of possibilities, and then using a heuristic evaluation function to approximate $$v_*$$ at the “leaf” nodes. (Heuristic search methods such as $$A_*$$ are ***almost always based on the episodic case***.)
- The methods of dynamic programming can be related even more closely to the Bellman optimality equation. Many reinforcement learning methods can be clearly understood as approximately solving the Bellman optimality equation, using actual experienced transitions in place of knowledge of the expected transitions.

$$v_*(s)$$를 알때 optimal policies

- We can get a $$v_*(s)$$ and using $$v_*(s)$$, we can calculate optimal policies

$$
\pi_*(s) = \underset{a}{argmax}\sum_{s'}\sum_{r}p(s',r|s,a)[r+\gamma v_*(s')]
$$

$$q_*(s,a)$$를 알 때
$$
\pi_*(s) = \underset{a}{argmax} \text{ } q_*(s,a)
$$


## 3.7 Optimality and Approximation

***Clearly, an agent that learns an optimal policy has done very well, but in practice this rarely happens.*** For
the kinds of tasks in which we are interested, optimal policies can be generated only with extreme computational cost.

***The memory available is also an important constraint***. A large amount of memory is often required to build up approximations of value functions, policies, and models.

- In tasks with small, finite state sets, it is possible to form these approximations using arrays or tables with one entry for each state (or state–action pair). This we call the tabular(표로 나타낸) case, and the corresponding methods we call tabular methods.
- In many cases of practical interest, however, there are far more states than could possibly be entries in a table. In these cases the functions must be approximated, using some sort of more compact parameterized function representation.