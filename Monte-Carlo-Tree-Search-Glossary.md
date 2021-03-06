# A Survey of Monte Carlo Tree Search Methods

Based on http://ccg.doc.gold.ac.uk/ccg_old/papers/browne_tciaig12_1.pdf

## Overview

A great benefit of MCTS is that the values of intermediate states do not have to be evaluated, as for depth-limited minimax search, which greatly reduces the amount of domain knowledge required. ***Only the value of the terminal state at the end of each simulation is required***.

## Bandit-based methods

### Regret

The policy should aim to minimize the player’s regret, which is defined after `n` plays as:
$$
R_N = \mu^*n - \mu_j \sum^K_{j=1}\mathbb{E}[T_j(n)]
$$

- $\mu^*$ : The best possible expected reward
- $E[T_j(n)]$ : The expected number of plays for arm `j` in the first n trials.

In other words, ***the regret is the expected loss due to not playing the best bandit***. It is important to highlight the necessity of attaching non-zero probabilities to all arms at all times, in order to ensure that the optimal arm is not missed due to temporarily promising rewards from a sub-optimal arm.

### Upper Confidence Bounds(UCBs)

For bandit problems, it is useful to know the upper confidence bound (UCB) that any given arm will be optimal. ***The simplest UCB policy proposed by Auer et al. [13] is called UCB1***, which has an expected logarithmic growth of regret uniformly over `n` (not just asymptotically) without any prior knowledge regarding the reward distributions (which have to have their support in[0,1]). The policy dictates to play arm `j` that maximizes

Upper confidence indices : Allow the policy to estimate the expected reward of a specific bandit once its index is computed.

- However, It is difficult to compute

Upper confidence bounds(UCBs) 

- Simplest ***UCB(It is called as UCB1) policy has an expected logarithmic growth of regret uniformly over n*** (Not just asymptotically) without any prior knowledge regarding the reward distributions.

$$
UCB1 = \bar{X}_j + \sqrt{\frac{2\ln{n}}{n_j}}
$$

- Maximize UCB1
- $\bar{X}_j$ : Average reward from arm j
  - It encourages the exploitation
- $n_j$ : The number of times arm j was played
- $n$ : The overall number of plays so far. (The number of times the current parent node has been visited)
  - $\sqrt{\frac{2\ln{n}}{n_j}}$ : It encourages the exploration
  - To maximize this fraction number, denominator has to be 0
  - It means agents have to minimize the number of try

## Monte Carlo Tree Search(MCTS)

### Algorithm

1. Selection
   - Starting at the root node, a child selection policy is recursively applied to descend ***through the tree until the most urgent expandable node is reached***.
   - A node is ***expandable if it represents a nonterminal state and has unvisited(i.e., unexpanded) children***.
   - 확장 가능한 노드에 도달 할 때까지 트리를 탐색 함
   - 여기서 확장 가능한 노드란, Terminal state가 아니고, 방문되지 않은 child node
2. Expansion
   - One (or more) child nodes are added to expand the tree, according to the available actions.
   - 확장 가능한 노드에서 action을 했을 때 얻은 결과(노드)
3. Simulation
   - A simulation is run from the new node(s) according to the default policy to produce an outcome.
   - 정해진 policy에 따라서 실행하는 것
4. Backpropagation
   - The simulation result is “backed up”(i.e., backpropagated) through the selected nodes to update their statistics.
   - 시뮬레이션의 결과가 역전파(Backpropagation) 되는 것

#### Selecting action

1. max child: select the root child with the highest reward;
2. robust child: select the most visited root child;
3. ***max-robust child***: select the root child with both the highest visit count and the highest reward;if none exists, then continue searching until an acceptable visit count is achieved;
4. secure child: select the child which maximizes a lower confidence bound

## Upper Confidence Bounds for Trees(UCT)

