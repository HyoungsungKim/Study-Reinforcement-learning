# Meta-RL

Based on https://lilianweng.github.io/lil-log/2019/06/23/meta-reinforcement-learning.html

## key component

### 1. A model with memory

A recurrent neural network maintains a hidden state. Thus, ***it could acquire and memorize the knowledge about the current task*** by updating the hidden state during rollouts.

- Without memory, meta-RL would not work.
- Adjusting the weights of a recurrent network is slow but it allows the model to work out a new task fast with its own RL algorithm implemented in its internal activity dynamics.

### 2. Meta-learning Algorithm

A meta-learning algorithm refers to how we can update the model weights to optimize for the purpose of solving an unseen task fast at test time. In both Meta-RL and $RL^2$ papers, the meta-learning algorithm is the ordinary gradient descent update of LSTM with hidden state reset between a switch of MDPs.

### 3. A Distribution of MDPs

While the agent is exposed to a variety of environments and tasks during training, it has to learn how to adapt to different MDPs.

---

Inductive bias: “a set of assumptions that the learner uses to predict outputs given inputs that it has not encountered”

- As a general ML rule, a learning algorithm with weak inductive bias will be able to master a wider range of variance, but usually, will be less sample-efficient.
- Therefore, to narrow down the hypotheses with stronger inductive biases help improve the learning speed.

In meta-RL, we impose certain types of inductive biases from the *task distribution* and store them in *memory*. 

## Meta-Learning Algorithms for Meta-RL

https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html#optimization-based

The gradient-based optimization is neither designed to cope with a small number of training samples, nor to converge within a small number of optimization steps.

- Is there a way to adjust the optimization algorithm so that the model can be good at learning with a few examples
  - This is what optimization-based approach meta-learning algorithms intend for.

### LSTM Meta-Learner

The goal of the meta-learner is to efficiently update the learner’s parameters using a small support set so that the learner can adapt to the new task quickly.

Notation

- $M_\theta$ : model
- $\theta$ : weight of model
- $R_\Theta$ : meta-learner
- $\Theta$ : weight of meta-learner
- $\mathcal{L}$ : loss function

### Why LSTM?

