# Planning with Monte Carlo Tree Search

## Introduction to planning

Monte carlo tree search

- Practical and useful for many purpose
  - Crucial component of alpha go

### Exploration reduces uncertainty

Model-free setting

- Unknown environment dynamics $$p(s', r | s, a)$$
  - We don't know how the environment will respond to our reactions and thus don't know the reward.
  - We know nothing about environment dynamics
    - We want to know about probabilities of the next state and reward, given current state and action.

Model-based setting

- Known model of the world

- We are given a model of the world in either of the two forms

  1. Distributed model : $$p(s', r | s, a)$$
     - It is also called explicit model
     - It provides access to the whole distribution of the next state in the world given current state and action
     - Distribution 모델에서는 모든 정보를 제공함
     - Such distributed model allows access to explicit probabilities of all the possible next states, s' and rewards
       

  2. Sample model : $$(s', r) \sim G(s,a)$$
     - We can obtain only samples of the next state and the reward given current state of action.
     - We do not know explicit probabilities of this samples
     - Sample model is also called generative model in scientific literature

In a model-based learning we know everything?

- ***NO***
- What we know in a model-based setting are immediate rewards that follow arbitrary action might be in arbitrary state
  - 우리가 model-based 에서 알 수 있는건 현재 reward와 이에 따르는 임의의 action, state 뿐임
  - 하지만 우리가 알고싶은건 greedy-reward가 아니라 global-reward임
- Do we need exploration for model-based learning?
  - Still don't know the global values of any action!

### We don't know global value estimates

Model-based에서 최적의 정답을 찾는 것을 planning algorithm이라고 함

Given the model of the world, what is the best action?

1. Immediate rewards are ***insufficient***
2. ***Time constraints*** on recovering sum of future rewards

>Planning
>
>***Transform*** immediate rewards and dynamics ***Into***
>
>- value
>- action-value estimates
>- explicit policy

### Types of planning

- Model-free : learn the policy by trial and error
- Model-based : plan to obtain the policy
- There are two types of planning
  - Background planning(Dynamic Programming)
    - Background planning is an approach of sampling from given environment model, either true or approximate, and learning from such samples with any model-free method. 
    - Learning from simulated experience
    - Improves estimates in arbitrary states
  - Decision time planning(Heuristic search, MCTS)
    - ***Make decisions and perform planning simultaneously sometimes***
    - Focuses on current state
      - Select an optimal decision in current state
      - ***Not trying to improve the policy everywhere*** distinguishes decision-time planning from background planning
      - 모든 선택지를 고려하지 않음
    - Plans action selection for current state only
    - Commit an action only after planning is finished
  - 두 방법의 공통적인 목표는 현재 상태에서 다음 action을 할 때 최적의 선택을 하는 것임

### Guided planning - Heuristic search

1. Exhaustive
   - 가능한 것 다 해봄
   - Dynamic programming
     - Tree의 leaves에서 root까지 true value인 것들을 propagate
   - This algorithm is not practical
     - It may requires too much time to finish planning, even for a single state.
2. Full-width with fixed depth
   - Stop enrolling as soon as we have reached some prespecified depths
   - Then perform back out from the leaves and the depths operate to the root
   - But, it is myopic. Because it does not account for reward that the cure deeper that the depths we have stopped at
     - 특정 깊이의 leaves에서 root로 전파 함
     - 이 방법은 근시안적임. 왜냐하면 더 깊은 곳에 더 좋은 방법이 있을지 모르지만 그건 고려하지 않기 때문에.
3. Full-width with fixed depth, approximate at leaves
   - Approximate value function at leaves and then propagating these approximate values backward to the top.
   - It is much practical
   - We can improve more!
4. ***Heuristic***
   - What we can improve is the selection of nodes which we unroll, expand on each iteration.
     - 각 반복에 대해 노드를 선택하거나 확장시키는 것을 개선 할 수 있음
     - 이전까지는 fixed-depth였음
   - In the previous algorithm, this expansion was uniform.
     - It means we considers everything which we can available
   - However we might not be interested in expanding branches that corresponds to bad values.
     - 하지만 우리는 나쁜 values에 해당하는건 고려하고 싶지 않음
   - To give the expansion in the direction of most promising actions, ***we can use any function which correlates with true returns that our policy's able to achieve.***
     - It is called heuristic function
     - And any planning method using such heuristic function is called a heuristic search. 
       - 예를 들어 대각선이 존재하지 않는 lattice에 존재하는 한 점에서 다른 한점까지의 최단거리는 두 점과 점의 직선 거리가 아님
       - 하지만 이 방법은 점 사이의 최단거리의 heuristic한 방법으로 사용 가능 함
     - This function is used to prove unfruitful branches by estimating the total cost or a worth that can be obtained from any single state 
       - 이 방법은 불필요한 것들을 밝히는데 사용 가능 함
   - In reinforcement learning, the heuristic function corresponds to the value function and unlike the usual heuristic function, ***we might want to refine our value function over time.***

### Guided planning - Heuristic search

Heuristic search : An umbrella term for many algorithms

- Lookahead planning guided by heuristic function
  - ***Select only "Valuable" nodes to expand***

Heuristic search for RL

- Resource are focused only on valuable paths
- Nearest states contribute the most to the return
- Drawbacks
  - Bad planning results when the model is unreliable
  - Dependence on the quality of heuristic

### Obtain value estimates with roll outs

1. While within computational budget:

   a. From state `s` make action `a`

   b. Follow rollout policy until episodes terminates(Monte carlo)

   - Policy used to estimate the value by Monte Carlo trajectory sampling is called rollout policy. 

   c. Obtain a return $$G_i$$

2. Output $$\hat{q}(s,a) = \frac{1}{n}\sum_{i=1}^n G_i$$

Pros

- No function approximation
- Trials are independent  -> Parallelization
  - Each rollout is totally independent of the others, which should also perform many rollouts ***simultaneously on many CPU***

Cons

- Dependence on # on returns averaged
  - The precision of the estimates obviously depends on the number of returns average. (Trade-off)

### Using rollouts for action selection

Greedy policy : Policy improvement as in Value iteration
$$
\pi(s) \leftarrow argmax_a \hat{q}(s,a)
$$

- If we want our policy prime to be optimal or close to optimal, we should make our rollout policy as good as we can. 
  - No estimation of q*
  - No further policy improvement
- Therefore, Good rollout policy is needed!
- Good rollout policy may require much of valuable time
  - May result in less reliable estimate of $$\hat{q}(s,a)$$
  - If rollout policy is good, that may mean it requires much time to decide what action to make
  - And here emerges the second instance of the ***time versus accuracy trade-off.***
- Drawbacks
  - This algorithm does not preserve any estimates between successive states. This may not be the best thing to do. 
    - 이 알고리즘은 연속적인 states 사이의 예측을 보존하지 않음
  - Additionally, this algorithm does not care about uncertainty in its action-value estimates. 
    - 추가적으로 이 알고리즘은 action-value 측정의 불확실성을 고려하지 않음

### Recap

- Planning : model in, policy out
- Crucial constraint in planning : time
- Accurate planning : guided by heuristic
- Variant of heuristic : MC estimate
- Simplest policy : greedy with respect to heuristic

Not yet covered

- Preserving estimates between successive time steps
- Dealing explicitly with uncertainty in estimates
- Guiding search with heuristic

## Monte Carlo Tree Search

Monte Carlo Tree Search is nothing but a family of decision-time planning algorithms which may be viewed as distant relatives of heuristic search. 

- Algorithm of Monte Carlo tree search family heavily relies on Monte Carlo rollout estimates of action or action-value function. 

### Monte Carlo Tree Search at a glance

- Amazingly successful decision-time algorithms
- Distant relatives of heuristic search
- Usually require much of computation time
- Estimate action-values with MC rollouts
- ***Preserve value estimates between transitions***
  - This slightly reduces the computation time needed for the algorithm to work well.
- The ***main difference*** compared to the previously discussed naive version of planning with Monte Carlo estimates 
  - Maintain two policies
    - Tree policy : respect uncertainty, guides the search
      - The tree policy is a policy which actually determines the direction of the search while respecting uncertainty of the estimates.
    - Rollout policy  : estimates the return

### Scheme of Monte Carlo Tree Search

Selection -> Expansion -> Simulation -> Backup -> ...(Repeat while time remains)

- Monte Carlo tree search, like the naive algorithm we have discussed previously, uses Monte Carlo rollout to estimate the action values.
- However, unlike the previously discussed alternatives, Monte Carlo Tree search algorithms additionally build a tree. 
  - This tree consists of the initial segments, states and actions of trajectories that have obtained higher return previously.
  - In some sense, these beginnings of trajectories are the most interesting to us because they are most likely to lead us to the best possible trajectory. 
  - 이전의 몬테카를로 방법과 다르게 경로를 tree형태로 저장해 놓음

Typical Monte Carlo tree search algorithm can be divided into four main phases

- Selection
- Expansion
- Simulation
- Backup