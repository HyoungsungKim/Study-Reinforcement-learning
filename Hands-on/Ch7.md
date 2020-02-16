# Ch7 Higher-Level RL Libraries

## The agent

We have used epsilon-greedy behavior to explore the environment, but this doesn't change the picture much.

- Our agent could predict a probability distribution over the actions -> `policy agents`

## Experience source

At a high level, experience source classes take the agent instance and environment and provide you with step by step data from the trajectories.

## Experience replay buffers

The replay buffer normally has a maximum capacity, so old samples are pushed out when the replay buffer reaches the limit.

Problem:

- How to efficiently sample from a large buffer
- How to push old samples from the buffer
- In the case of a prioritized buffer, how priorities need to be maintained and handled in the most efficient way 

All replay buffers provide the following interface:

- A python iterator interface to walk over all the sample in the buffer
- The method `populate(N)` to get  `N` samples from the experience source and put them into the buffer
- The method `sample(N)` to get the batch of N experience objects

So, the normal training loop for DQN looks like an infinite repetition of the following steps:

1. Call `buffer.populate(1)` to get a fresh sample from the environment
2. `batch = buffer.sample(BATCH_SAMPLE)` to get the batch from the buffer
3. Calculate the loss on the sampled batch
   - ***Don't get a loss until buffer get a enough data***
4. Backpropagate
5. Repeat until convergence 

