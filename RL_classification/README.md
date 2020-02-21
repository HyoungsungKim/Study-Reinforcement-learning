# Experiment : Classification using Reinforcement learning

## TL;DR

- It is possible to do classification
- However It is hard to get a better result than ordinary classification SL

### Experiment idea

- If we use RL for classification with less dataset, Is it possible to get a better result?
- No, It is not
- I did and RL classification accuracy become close to SL's accuracy. However cannot get a better result. Even it takes more time to converge to SL's accuracy.

### ETC

- I used MNIST dataset for test and it converges very fast even less dataset
- Therefore, I selected just 50 MNIST data
  - It converges about 80% accuracy in SL
  - I got about 75% accuracy in RL

## Result

- If dataset is enough, RL can get a good accuracy(More than 95% using MNIST)
  - However SL can get a similar result with less training time



