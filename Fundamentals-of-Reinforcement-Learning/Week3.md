# Finite Markov Decision Processes part 2

## 3.4 Unified Notation for Episodic and Continuing Tasks

In this book we consider sometimes one kind of problem and sometimes the other, but often both. It is therefore useful to
establish one notation that enables us to talk precisely about both cases simultaneously.

- To be precise about episodic tasks requires some additional notation. Rather than one long sequence of time steps, ***we need to consider a series of episodes, each of which consists of a finite sequence of time steps.***
  - We number the time steps of each episode starting anew from zero.
  - Therefore, we have to refer not just to $$S_t$$ , the state representation at time $$t$$, but to $$S_{t,i}$$ , the state representation ***at time $$t$$ of episode $$i$$ (and similarly for $$A_{t,i}$$ , $$R_{t,i}$$, $$\pi_{t,i}$$ , $$T_i$$ , etc.).***
    - However, it turns out that when we discuss episodic tasks we almost never have to distinguish between different episodes.
    - ***We are almost always considering a particular single episode, or stating something that is true for all episodes.*** Accordingly, in practice we almost always abuse notation slightly by dropping the explicit reference to episode number. That is, we write $$S_t$$ to refer to $$S_{t,i}$$ , and so on.

## 3.5 Policies and Value Functions

Almost all reinforcement learning algorithms involve estimating ***value functions***—functions of states (or of state–action pairs) that ***estimate how good it is for the agent to be in a given state*** (or how good it is to perform a given action in a given state).

- The notion of “how good” here is defined in terms of future rewards that can be expected, or, to be precise, in terms of expected return.
  - Of course the rewards the agent can expect to receive in the future depend on what actions it will take.
  - Accordingly, value functions are defined with respect to particular ways of acting, called ***policies***.

Formally, a ***policy is a mapping from states to probabilities of selecting each possible action***.

- If the agent is following policy $$\pi$$ at time $$t$$, then  $$\pi(a|s)$$ is the probability that $$A_t = a$$ if $$S_t = s$$. Like $$p, \pi$$ is an ordinary function; the “|” in the middle of $$\pi(a|s)$$ merely reminds that it defines a probability distribution over $$a \in A(s)$$ for each $$s \in \bold{S}$$.
- Reinforcement learning methods specify how the agent’s policy is changed as a result of its experience.

The value function of a state `s` under a policy $$\pi$$, denoted $$v_\pi(s)$$, is the expected return when starting in `s` and following $$\pi$$ thereafter. For MDPs, we can define $$v_\pi$$ formally by
$$
v_\pi(s) \doteq \mathbb{E}[G_t|S_t=s] = \mathbb{E}_\pi [\sum^\infty_{k=0}\gamma^rR_{t+k+1} | S_t = s]
$$

- We call the function $$v_\pi$$ the ***state-value function*** for policy $$\pi$$ 
- State가 주어졌을때 value

Similarly, ***we define the value of taking action $$a$$ in state $$s$$ under a policy $$\pi$$,*** denoted $$q_{\pi}(s,a)$$, as the expected return starting from $$s$$, taking the action $$a$$, and thereafter following policy $$\pi$$ :
$$
q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t | S_t=t, A_t=a] = \mathbb{E}_\pi[\sum^\infty_{k=0}\gamma^k R_{t+k+1} | S_t = s ,A_t = a]
$$

- We call $$q_\pi$$ the ***action-value function*** for policy $$\pi$$.
- State와 action이 주어졌을 때 value

The value functions $$v_\pi$$ and $$q_\pi$$ can be estimated from experience.

For example, if an agent follows policy $$\pi$$ and maintains an average, for each state encountered, of the actual returns that have followed that state, then the average will converge to the state’s value, $$v_\pi(s)$$ , as the number of times that state is encountered approaches infinity.

- If separate averages are kept for each action taken in each state, then these averages will similarly converge to the action values, $$q_\pi(s, a)$$.
  - 각각의 state에서의 평균이 각각의 평균동안 유지된다면, 이 평균들은 action value에 수렴함
  - We call estimation methods of this kind ***Monte Carlo methods*** because they involve averaging over many random samples of actual returns.