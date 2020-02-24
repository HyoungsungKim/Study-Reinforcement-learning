# Dueling Network Architecture

https://arxiv.org/abs/1511.06581

## 2. Background

### Advantage Function

Basic function
$$
Q^\pi(s,a) = \mathbb{E}[R_t|s_t = s, a_t = a, \pi]\\
V^\pi(s) = \mathbb{E}_{a \text{~}\pi(s)}[Q_\pi(s,a)] \\
$$
Advantage function
$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s) \\
\mathbb{E}_{a\text{~}\pi(s)}[A^\pi(s,a)] = 0
$$


- Intuitively, the value function V measures the how good it is to be in a particular state `s`
- The Q function, However, measures the value of choosing a particular subtract the value of the state from the Q function to obtain a relative measure of the importance of each action

$$
\begin{align}
\mathbb{E}_{a\text{~}\pi(s)}[A^\pi(s,a)] & = \mathbb{E}_{a\text{~}\pi(s)}[Q^\pi(s,a) - V^\pi(s)] \\
& = V^\pi(s) - V^\pi(s) = 0
\end{align}
$$



## 3. The Dueling Network Architecture

We use two sequences (or streams) of fully connected  layers. The streams are constructed such that ***they have the capability of providing separate estimates of the value and advantage functions.***

- Finally, the two streams are combined to produce a single output Q function.

$$
a^* = \underset{a' \in A}{argmax} Q(s, a')
$$

If follows
$$
Q(s,a^*) = V(s) \text { it means } A(s, a^*) = 0
$$

- When `a` is scalar Q = V

Let us consider layer architecture. There are two output and we can denote each output as 

- $$V(s;  \theta, \beta)$$
- $$A(s,a;\theta, \alpha)$$
  - $$\theta$$ : Parameters of convolution later
  - $$\alpha, \beta$$ : Parameters of fully connected layer
  - In short, to get a value(V) we need 3 parameter

Using definition of advantage(A = Q - V)
$$
Q(s,a;\theta, \alpha, \beta) = V(s; \theta, \beta) + A(s,a;\theta, \alpha)
$$

- Q value is matrix form so we need to replicate the scalar $$V(s; \theta, \beta)\text{ } |A|$$ times(when dimension is `A`)
- It would be wrong to conclude that $$V(s;\theta,\beta)$$ is a good estimator of the state-value function, or likewise that $$A(s,a;\theta,\alpha)$$ provides a reasonable estimate of the advantage function.
- This lack of identifiability is mirrored by poor practical performance when this equation is used directly.
- To address this issue of identifiability, we can force the advantage function estimator to have zero advantage at the chosen action.
- For more information : https://ai.stackexchange.com/questions/8128/difficulty-in-understanding-identifiability-in-the-dueling-network-architecture

$$
Q(s,a;\theta, \alpha, \beta) = V(s;\theta, \beta) + (A(s,a;\theta,\alpha) - \max_{a' \in |\mathcal{A}|}A(s,a';\theta,\alpha))
$$

- When Q is max, then Q = V and A = 0
- Therefore, we can identify when `a` is optimal
- When `a` is optimal, Q == V. In other words, A - max(A) is 0 (minimum)

Alternative form
$$
Q(s,a,;\theta,\alpha, \beta) = V(s;\theta,\beta) + (A(s,a;\theta,\alpha) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a',\theta, \alpha))
$$

- We can recognize when V is good estimator.
- When `a` is optimal, A - mean(A) is minimum

## Result

In the experiments, we demonstrate that the dueling architecture can more quickly identify the correct action during policy evaluation as redundant or similar actions are added to the learning problem.

- Value에 어드벤티지를 더해서 어떤  value를 선택하는게 좋은지 판단 할 수 있음.
- 기존에 Q value를 구하기 위해 V와 A를 더하는 방법 은 존재했음
- 뉴런네트워크의 발달로 optimal한 V와 A  얻을 수 있음

The results  show  that both architectures converge at about the same speed. However, when we increase the number of actions, the dueling architecture performs better than the traditional Q-network.

- In the dueling network, the stream $$V(s;\theta,\beta )$$ learns a general value that is shared across many similar actions at `s`,  hence leading to faster convergence. 