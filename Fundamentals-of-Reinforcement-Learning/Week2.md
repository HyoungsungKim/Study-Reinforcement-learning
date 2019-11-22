# Finite Markov Decision Processes

In this chapter we introduce the formal problem of finite Markov decision processes, or finite MDPs, which we try to solve in the rest of the book.

- This problem involves evaluative feedback, as in bandits, but also an associative aspect—choosing different actions in different situations.
  - MDPs are a classical formalization of sequential decision making, ***where actions influence not just immediate rewards, but also subsequent situations, or states, and through those future rewards.***
- Thus MDPs involve delayed reward and ***the need to trade-off immediate and delayed reward.***
  - Whereas in bandit problems we estimated the value $$q_*(a)$$ of each action `a`
  - In MDPs we estimate the value $$q_*(s, a)$$ of each action `a` in each state `s`, or we estimate the value $$v_*(s)$$ of each state given optimal action selections.
    - Bandit 문제에서는 action `a`에서 $$q_*(a)$$를 측정 함
    - MDP에서는 State `s`에서 action `a`의 $$q_*(s,a)$$를 측정 함 또는  최적 action이 주어졌을 때 $$v_*(s)$$를 계산 함.
  - These state-dependent quantities are essential to accurately assigning credit for long-term consequences to individual action selections.
  - MDPs are a mathematically idealized form of the reinforcement learning problem for which precise theoretical statements can be made. We introduce key elements of the problem’s mathematical structure, such as returns, value functions, and Bellman equations.
  - We try to convey the wide range of applications that can be formulated as finite MDPs. As in all of artificial intelligence, there is a tension between breadth of applicability and mathematical tractability. In this chapter we introduce this tension and discuss some of the trade-offs and challenges that it implies. Some ways in which reinforcement learning can be taken beyond MDPs are treated in Chapter 17.