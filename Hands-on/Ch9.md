# Ch9 Ways to Speed up RL

## The baseline

We will use 2 characteristics

- The number of frames
- The wall clock time

## The computation graph in PyTorch

RL code is normally much more complex than traditional supervised learning models, so the RL model that we are currently training is also being applied to get the actions that the agent needs to perform in the environment.

So, in DQN, a neural network (NN) is normally used in three different situations:

- When we want to calculate Q-values predicted by the network to get the loss in respect to reference Q-values approximated by the Bellman equation
  - loss 값 구할 때 레퍼런스에서 계산 된 Q 값 빼는경우
- When we apply the target network to get Q-values for the next state to calculate a Bellman approximation
- When the agent wants to make a decision about the action to perform

***It is very important to make sure that gradients are calculated only for the first situation.***

Creating the graph still uses some resources (in terms of both speed and memory), which are wasted because PyTorch creates this computation graph even if we don't call `backward()` on some graph. To prevent this, one very nice thing exists: the decorator `torch.no_grad()`.

```python
@torch.no_grad()
def fun_a(t):
    return t*2
```

`@torch.no_grad()`쓰면 불필요한 계산이 줄어들어서 훨씬 빠르게 training함

## Several environments

The first idea that we usually apply to speed up deep learning training is ***larger batch size***. It's applicable to the domain of deep RL, but you need to be careful here.

- ***In the normal supervised learning case, the simple rule "a large batch is better" is usually true:*** you just increase your batch as your GPU memory allows, and a larger batch normally means more samples will be processed in a unit of time thanks to enormous GPU parallelism.

The RL case is slightly different. During the training, two things happen simultaneously:

- Your network is trained to get better predictions on the current data
- Your agent explores the environment

Environment수를 늘리면 리워드가 빠르게 증가함. 하지만 너무 많으면 오히려 수렴 속도가 늦어짐. env의 수는 3 언저리가 좋음.

## Play and train in separate processes



