#  Course outline
<section id="outline">

<a href="#/intro">Introduction</a>

<a href="#/background">Statistics and machine learning background</a>

<a href="#/intro_rl">Introduction to reinforcement learning</a>

<a href="#/mdp">Markov Decision Processes</a>

<a href="#/value">Value function methods</a>

<a href="#/policy-optimization">Policy optimization methods</a>

<a href="#/model-based">Model-based reinforcement learning</a>

<a href="#/alphago">AlphaGo to AlphaZero</a>


#  Introduction
<section id="intro">


## Where are we today?

AlphaGo was a milestone achievement for reinforcement learning
- generalized across other perfect information, deterministic games

Lots of research
- testing using Atari or Mujocco

Fleeting glimpses of application in industry
- Google data centres


## Where are we today

Modern trends
- deep neural networks as function approximators
- improvements on old algorithms (Q-Learning -> DQN)
- new algorithms (MCTS)
- access to compute

Challenges
- hidden information
- instability across random seeds
- sample efficiency
- access to simulators


## State of the art

Atari
- Performance - Hessel et. al (2017) Rainbow - [paper](https://arxiv.org/abs/1710.02298)
- Sample efficiency - Kaiser et. al (2019) SimPLe - [paper](https://arxiv.org/pdf/1903.00374.pdf)

AlphaZero
- DeepMind - built on top of MCTS

Open AI Five
- built on top of PPO

World Models
- agent's learning inside dream environments


## Atari

<img src="assets/images/montezuma.png"
	width="35%"
	height="35%">
<figcaption>Montezuma's Revenge - known for being a difficult exploration problem, state of the art is a [combination of imitation and reinforcement learning](https://arxiv.org/abs/1812.03381)</figcaption>


## Atari

The Atari Learning Environment (ALE) is a key reinforcement learning benchmark
- learning from pixels
- discrete space
- stack four frames to make Markovian

The 2013 DQN paper has seven superhuman agents
- since 2013 many improvements have been made to DQN
- the state of the art today on Atari is the model free algorithm Rainbow, which combines these improvements


## AlphaZero

<img src="assets/images/ag_learning_nets.png"
	width="35%"
	height="35%">
<figcaption></figcaption>


## AlphaZero

Generalization of superhuman performance across Go and Chess
- model based
- planning using Monte Carlo Tree Search
- deep convolutional residual networks

AlphaGo -> AlphaGoZero -> AlphaZero
- removed dependence on human expert data
- reduced searching
- less compute


## Open AI Five

<img src="assets/images/open_ai_five.png"
	width="35%"
	height="35%">


## Open AI Five

Semi-professional level
- hidden infomation team game
- learning from n-d arrays
- simplifed version of the full game
- large & continuous action space

Finals on April 13th 2019


## World Models

TODO picture

Really cool use of auto encoders, LSTMs, and evolutionary methods.

High quality interactive [blog post](https://worldmodels.github.io/)

Model based reinforcement learning
- auto-encoder to transform pixel into a latent space
- agent makes decisions based on latent space
- lstm memory that makes predictions on next latent space
- covariance matrix adaptation evolution strategy (CMA-ES) to optimize parameters of a simple controller


## Notable people*

Rich Sutton - University of Alberta / DeepMind
- co-author of the Bible of reinforcement learning

David Silver - DeepMind
- lead author on the AlphaGo papers

John Schulman - Berkeley / Open AI
- lead author TRPO, PPO

Sergey Levine - Berkeley / Open AI / Google
- robotics, teaches CS294

<figcaption>*notable due to research and visibility through teaching</figcaption>


## About me

Adam Green - [adam.green@adgefficiency.com](adam.green@adgefficiency.com) - [adgefficiency.com](http://adgefficiency.com)

- chemical engineer by training
- four years as an energy engineer
- two years as an energy data scientist

Enjoy building computational models


## This course

Developed over two years
- 10 courses, ~90 students

Originally a two day introduction to reinforcement leaning at Data Science Retreat in Berlin
- students had familiarity with supervised learning
- no familiarity with reinforcement learning

Goals for today
- landscape of reinforcement learning - terminology, key algorithms
- challenges in modern reinforcement learning
- where to go next


## Where to go next

Open AI's Spinning Up in Deep RL - [lecture]() - [notes]()

Sutton & Barto - An Introduction to Reinforcement Learning (2nd Edition) - [textbook pdf](http://incompleteideas.net/book/RLbook2018trimmed.pdf)

David Silver's UCL lectures [YouTube](https://www.youtube.com/watch?v=2pWv7GOvuf0)

CS294 - Berkley Deep Reinforcement Learning - [course notes](http://rail.eecs.berkeley.edu/deeprlcourse/) - [lecture videos](https://www.youtube.com/playlist?list=PLkFD6_40KJIxJMR-j5A1mkxK26gh_qg37)
Environments - [Open AI gym](https://github.com/openai/gym)

Agents - [Open AI baselines](https://github.com/openai/baselines) - [TF Agents]() - [Dopamine]()



## Nomenclature

Experience

|   |   |
|---|---|
|$s$ |state     |
|$s'$|next state|
|$a$ |action    |
|$r$ |reward    |

Measurements of total reward

|   |   |
|---|---|
|$ \mathbf{E}[f(x)] $  | expectation of $f(x)$ |
|$\gamma$ |  discount factor [0, 1) |
|$G_t$ | discounted return after time t|
|$ V_\{\pi} (s)$| value function |
|$ Q_\{\pi} (s,a)$| action-value function |


## Nomenclature

Taking actions

- sampling action from a stochastic policy

$$a \sim \pi(s)$$

- deterministic policy

$$a = \pi(s)$$

- optimal policy

$$\pi^\star$$

- function parameters (weights)

$$\theta , \omega$$


# Statistics and machine learning background
<section id="background">

what are the advantages and disadvantages of lookup tables?

which of bias/variance is over/under fitting?

how is reinforcement learning different from supervised learning?


## Expectations

The mean - weighted average of all possible values

`$$expectation = probability * magnitude$$`

`$$\mathbf{E} [f(x)] = \sum p(x) \cdot f(x)$$`

Expectations allow us to approximate by sampling

- to approximate the average time it takes us to get to work
- measure how long it takes us for a week and average each day

`$$expectation = average(samples)$$`

$$\mathbf{E} [f(x)] = \frac{1}{N} \sum\_{i=0}^{N} f(x\_{i})$$


## Expectations

Expectations are convenient in reinforcement learning

The expectation gives us something to aim at
- reinforcement learning optimizes for total expected reward

The expectation allows us to approximate using samples
- often all we can do is sample
- working with expectations allows us to use samples as approximations for the true expectation


## IID

Fundamental assumption in statistical learning
- independent and identically distributed
- assuming that the training set is independently drawn from a fixed distribution

Independent
- our samples are not correlated/related to each other

Identically distributed
- the distribution across our data set is the same as the 'true' distribution


## Conditionals

Probability of one thing given another

- probability of next state $s'$ given state $s$ & action $a$

$$ P(s'|s,a) $$

- reward received conditioned on taking action $a$ in state $s$, then transitioning to state $s'$

$$ R(r|s,a,s') $$

- sampling an action from the policy - action $a$ conditioned on state $s$

$$ a \sim \pi (s|a) $$


## Lookup tables

State with two dimensions

```python
state = np.array([temperature, pressure])
```

One row per element of the discretized state space

|state |temperature | pressure | estimate |
|---|---|---|---|
|0   |high   |high   |unsafe   |
|1   |low   |high   |safe   |
|2  |high   |low   |safe   |
|3   |low   |low   |very safe   |


## Lookup tables

Advantages

- stability
- each estimate is independent of every other estimate

Disadvantages

- no sharing of knowledge between similar states/actions
- curse of dimensionality
- discretization


## Linear functions

$$ V(s) = 3s_1 + 4s_2 $$

Advantages

- less parameters than a table
- can generalize across states

Disadvantages

- the real world is often non-linear


##  Non-linear functions

Most commonly neural networks

Advantages

- model complex dynamics
- convolution for vision
- recurrency for memory / temporal dependencies

Disadvantages

- instability
- difficult to train


# Neural networks


## Neural networks

Used to approximate functions
- forward pass maps from input to output
- forward pass maps from features to target

Learn these functions by creating targets

$$loss = approximation - target$$

- this is supervised learning

Use the loss to find gradients
- loss is a measure of error
- use gradients to change weights in direction that reduces error


## Neural networks

Feedforward / fully connected
- general purpose function approximator

Convolution
- learning from pixels

Recurrent
- learning from sequences

All use features to predict a target


## Bootstrapping

Create targets using bootstrapping

- function creates targets for itself

The Bellman Equation is bootstrapped equation

$$ V(s) = r + \gamma V(s') $$

$$ Q(s,a) = r + \gamma Q(s', a') $$

Bootstrapping causes

- bias - the agent has a chance to fool itself
- instability - weight updates depend on previous weights


## Variance & bias

Variance
- if I did this again, would I get the same result

Bias
- am I getting the correct result

<img src="assets/images/s1_f2.png"
	width="40%"
	height="40%">
<figcaption></figcaption>


## Variance & bias in supervised learning

Model generalization error = bias + variance + noise

Variance = deviation from expected value = overfitting

- error from sensitivity to noise in data set
- seeing patterns that aren't there

Bias = deviation from true value = underfitting

- error from assumptions in the learning algorithm
- missing relevant patterns


## Variance & bias in reinforcement learning

Variance = deviation from expected value
- how consistent is my model / sampling
- can often be dealt with by sampling more - 'sample through' variance
- high variance = sample inefficient

Bias = deviation from true value
- how close to the truth is my model
- approximations or bootstrapping tend to introduce bias
- biased away from an optimal agent / policy


## Random seeds

Used to control randomness in computers

Using the same random seed should result in the same random numbers 

Instability of reinforcement learning across different random seeds is a key message of this course


# Introduction to reinforcement learning

<section id="intro_rl">

*what else could/should I try before reinforcement learning?*

*is my problem a reinforcement learning problem?*

*overview of the reinforcement learning landscape*


## Learning through action

Sequential decision making
- seeing cause and effect
- consequences of actions 
- what to do to achieve goals

Substrate independent
- occurs in silicon and biological brains

Key features
- trial & error search
- delayed reward


## What is reinforcement learning?

Computational approach to learning through action

End to end problem of goal directed agent interaction with an environment

Learning to maximize a reward signal
- map situations to actions
- actions effect the future


## The control problem

Trying to take good actions
- generate and test
- trial and error learning
- evolution

Prediction and control
- supervised and reinforcement
- being able to predict allows us to take good actions


## Control methods

Without gradients

- guess and check
- cross entropy method
- evolutionary methods
- constrained optimization

Gradient based

- optimal control - model based
- model based reinforcement learning
- model free reinforcement learning


## The four grades of competence

Each is a successive application of generate and test with improved competence
- competence = ability to act well


## The four grades of competence

Darwinian
- pre-designed and fixed competence
- no learning within lifetime
- global improvement via local selection

Skinnerian
- also has the ability to adjust behaviour through reinforcement
- learning within lifetime
- hardwired to seek reinforcement

<figcaption>Dennet - From Bach to Bacteria and Back</figcaption>


## The four grades of competence

Popperian
- learns models of the environment
- local improvement via testing behaviours offline
- crows, cats, dogs, dolphins & primates

Gregorian
- builds thinking tools
- arithmetic, democracy & computers
- systematic exploration of solutions
- local improvement via higher order control of mental searches
- only humans

<figcaption>Dennet - From Bach to Bacteria and Back</figcaption>


## The four grades of competence

Darwinian
- fixed competences, tested by selection

Skinnerian
- adjusts behaviour through reinforcement

Popperian
- learns models of the environment, tests using environment models

Gregorian
- builds thinking tools


## Applications of reinforcement learning

Sequential decision making problems

- control physical systems
- interact with users
- solve logistical problems
- play games
- learn sequential algorithms

<figcaption>[David Silver – Deep Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf)</figcaption>

- chess
- optimization of refinery
- calf learning to walk
- robot battery charging
- making breakfast

<figcaption>Sutton & Barto</figcaption>


## Applications of reinforcement learning

Requirement of a reward signal

Ability to simulate through sample inefficiency


<img src="assets/images/mdp.png"
	width="80%"
	height="80%">


## Is my problem a reinforcement learning problem?

What is the action space
- what can the agent choose to do
- does the action change future states

What is the reward function
- does it incentive behaviour

It is a complex problem
- linear programming or cross entropy may offer a simpler solution

Can I sample efficiently / cheaply
- do you have a simulator


## Reinforcement learning is hard

Debugging implementations is hard
- very easy to have subtle bugs that don't break your code

Tuning hyperparameters is hard
- tuning hyperparameters can also cover over bugs!

Instability across random seeds
- results will succeed and fail over different random seeds (same hyperparameters!)
- this means you need to do a lot more experiments

Machine learning is an empirical science
- the ability to do more experiments directly correlates with progress
- this most true in reinforcement learning


##  Biological inspiration

> Of all the forms of machine learning, reinforcement learning is the closest to the kind of learning that humans and other animals do, and many of the core algorithms of reinforcement learning were originally inspired by biological learning systems
<figcaption>Sutton & Barto</figcaption>


##  Dopamine as a value function

Concentration of dopamine associated with the start of the stimulus
<img src="assets/images/s2_f3.png"
	width="30%"
	height="30%">

Expected reward not experienced -> no dopamine
<img src="assets/images/s2_f4.png"
	width="30%"
	height="30%">

<figcaption>Fig. 3. from [Glimcher (2011) Understanding dopamine and reinforcement learning](https://www.pnas.org/content/108/Supplement_3/15647)</figcaption>


## Reward hypothesis

Maximising expected return (which is what agent's do) is making an assumption about the nature of our goals

> Goals can be described by the maximization of expected cumulative reward

- do humans optimize for expected value?
- what about multi-modal distributions?

The Reward Engineering Principle

> As reinforcement-learning based AI systems become more general and autonomous, the design
> of reward mechanisms that elicit desired behaviours becomes both
> more important and more difficult.

<figcaption>The Reward Engineering Principle - [Dewey (2014) Reinforcement Learning and the Reward Engineering Principle](http://www.danieldewey.net/reward-engineering-principle.pdf)</figcaption>


## Context within machine learning

<img src="assets/images/s2_f1.png"
	width="70%"
	height="70%">
<figcaption></figcaption>


## Contrast with supervised learning

Supervised learning
- given a dataset with labels
- test on unseen data

Reinforcement learning
- need to generate data by taking actions
- need to label data


## Contrast with supervised learning

Freedom from the constraint of a dataset
- hard to access high quality datasets
- easy to access high quality simulators

Democratization

Requirement of a dataset is replaced with requirement of a simulator
- sample inefficiency limits learning from small number of samples


## Reinforcement learning data

Reinforcement learning involves

- generating data by taking actions
- creating targets for data

One sample of data = experience/transition tuple $(s,a,r,s')$

- no implicit target

The dataset we generate is the agent's memory

- list of experienced transitions

$$[(s\_{0}, a_0, r_1, s_1), $$
$$(s_1, a_1, r_2, s_2), $$
$$...$$
$$(s_n, a_n, r_n, s_n)] $$



##  Deep reinforcement learning

Deep learning
- neural networks with multiple layers

Deep reinforcement learning
- using multiple layer networks to approximate policies or value functions
- feedforward
- convolutional
- recurrent


## A rough map of the reinforcement learning landscape

<img src="assets/images/s2_f2.png"
	width="100%"
	height="100%">


<img src="assets/images/spinning_taxonomy.png"
	width="100%"
	height="100%">
<figcaption> Open AI Spinning Up</figcaption>


# Four central challenges


## Exploration versus exploitation

Do I go to the restaurant in Berlin I think is best – or do  I try something new?

- exploration = finding information
- exploitation = using information

Agent needs to balance between the two
- we don't want to waste time exploring poor quality states
- we don't want to miss high quality states


## Exploration versus exploitation

How stationary are the environment state transition and reward functions?

How stochastic is my policy?

Design of reward signal versus exploration required

Time step matters

- too small = rewards are delayed = credit assignment harder
- too large = coarser control


## Data quality

Reinforcement learning breaks both assumptions in IID

Independent sampling

- all the samples collected on a given episode are correlated (along the state trajectory)
- our agent will likely be following a policy that is biased (towards good states)

Identically distributed

- learning changes the data distribution
- exploration changes the data distribution
- environment can be non-stationary


##  Credit assignment

The reward we see now might not be because of the action we just took

Reward signal can be
- delayed - benefit/penalty of action only seen much later
- sparse - experience with reward = 0

Can design a more dense reward signal for a given environment
- reward shaping
- changing the reward signal can change behaviour


## Sample efficiency

How quickly a learner learns

How often we reuse data
- do we only learn once or can we learn from it again
- can we learn off-policy

How much we squeeze out of data
- i.e. learn a value function, learn a environment model

Requirement for sample efficiency depends on how expensive it is to generate data
- cheap data -> less requirement for data efficiency
- expensive / limited data -> squeeze more out of data


## Four challenges

Exploration vs exploitation
- how good is my understanding of the range of options

Data
- policy is biased in sampling experience
- distribution of experience changes as we learn

Credit assignment
- which action gave me this reward

Sample efficiency
- learning quickly, squeezing information from data


# Markov Decision Processes
<section id="mdp">

what's wrong with discretizing an action space?

what is the credit assignment problem?

what are the two main tasks of an agent?

why is on-policy learning less sample efficient?


## Markov Decision Processes (MDPs)

Framework from dynamical systems theory
- optimal control of partially observed MDP's

Mathematical framework for sequential decision making
- framework within which agent's live

Markov property
- additional information will not improve our decision
- can make decisions using only the current state
- don't need the history of the process

$$ P(s\_{t+1} | s\_{t}, a\_{t}) = P(s\_{t+1}|s\_t,a\_t...s\_0,a\_0)$$


<img src="assets/images/mdp.png"
	width="80%"
	height="80%">


## Formal definition of a MDP

An MDP is defined as a tuple

$$ (\mathcal{S}, \mathcal{A}, \mathcal{R}, P, R, d_0, \gamma) $$

- set of states $\mathcal{S}$
- set of actions $\mathcal{A}$
- set of rewards $\mathcal{R}$
- state transition function $ P(s'|s,a) $
- reward transition function $ R(r|s,a,s') $
- distribution over initial states $d_0$
- discount factor $\gamma$



```python
done = False
while not done:

	#  select an action based on the observation
	action = agent.act(observation)

	#  take a step through the environment
	next_observation, reward, done, info = env.step(action)

	#  store the experienced transition
	agent.remember(observation, action, reward, next_observation, done)

	#  improve the policy
	agent.learn()
```


## Environment

Real or virtual
- virtual environments allow cheap sampling of experience

Each environment has a state space and an action space
- discrete or continuous


## Environment

Episodic
- finite horizon, with a variable or fixed length

Non-episodic
- infinite horizon

Both exist in the same mathematical framework
- discounting to keep returns finite
- a post episode 'absorbing state' turns episodic into infinite horizon


## Discretiziation

Requires some prior knowledge

Lose the shape of the space

Too coarse
- non-smooth control output

Too fine
- curse of dimensionality
- computational expense


## Environment model

Our agent can learn an environment model

Predicts environment response to actions
- predicts $s', r$ from $s, a$

```python
def model(state, action):
    # do stuff
    return next_state, reward
```

Sample vs. distributional model

Model can be used to simulate trajectories for planning


## State and observation

```python
state = np.array([temperature, pressure])
```
- ground truth of the system

``` python
observation = np.array([temperature + noise, pressure + noise, noise])
```

- information the agent sees about the system
- used to choose actions and to learn

Observation can be made more Markov by
- concatenating state trajectories together
- using function approximation with a memory element (LSTMs)


## Reward and return

Reward ($r$) is a scalar
- delayed
- sparse

Return ($G_t$) is the total discounted future reward

$$G\_t = r\_{t+1} + \gamma r\_{t+2} + \gamma^2 r\_{t+3} + ... = \sum\_{k=0}^{\infty} \gamma^k r\_{t+k+1}$$

Why do we discount future rewards?


## Discounting

Future is uncertain
- stochastic environment

Matches human thinking
- hyperbolic discounting

Finance
- time value of money

Makes the maths works
- return for infinite horizon problems finite
- discount rate is $[0,1)$
- can make the sum of an infinite series finite
- geometric series


## Agent

Goal to maximize total reward
- often optimize for the expectation of future discounted reward

Our agent has a lot to do!
- decide what actions to take
- learn how to take better actions

Reinforcement learning is the interaction of these two processes
- prediction versus control
- general policy iteration
- data generated by actions used in learning


<img src="assets/images/mdp2.png"
	width="90%"
	height="90%">


## Policy

$$\pi(s) = \pi(s,a;\theta) = \pi_\theta(s|a)$$

Rules to select actions

Any rule is valid
- act randomly
- always pick a specific action

We want the optimal policy - $\pi_{*}(s)$
- the policy that maximizes future reward


## Policy

```python
def random_policy():
	return env.action_space.sample()
```

Parametrized directly
- policy gradient methods

Generated from a value function
- value function methods

Deterministic or stochastic
- often explore using a stochastic policy


## On versus off policy learning

On policy
- learn about the policy we are using to make decisions
- limited to data generated using our policy

Off policy
- evaluate or improve one policy while using another to make decisions
- use data generated by other policies
- less stable


<img src="assets/images/on_off_policy.png"
	width="80%"
	height="80%">


## Why would we want to learn off-policy?

We can learn about policies that we don't have
- learn the optimal policy from data generated by a random policy

We can reuse data
- on-policy algorithms have to throw away experience after the policy is improved
- off policy learning increases the diversity of the datasets we can learn from

> Maybe the lesson we need to learn from deep learning is large capacity learners with large and diverse datasets - Sergey Levine




# Value functions
<section id="value">

what do value functions predict?

how do we use the Bellman equation to act?

how do we use the Bellman equation to learn?


## Richard Bellman

<img src="assets/images/bellman.png"
	width="20%"
	height="20%">

Invented dynamic programming in 1953

Also introduced the curse of dimensionality
- number of states $\mathcal{S}$ increases exponentially with number of dimensions in the state


> I was interested in planning, in decision making, in thinking. But planning, is not a good word for various reasons. I decided therefore to use the word, ‘programming.’ I wanted to get across the idea that this was dynamic, this was multistage, this was time-varying...

<figcaption>[On the naming of dynamic programming](ttp://arcanesentiment.blogspot.com.au/2010/04/why-dynamic-programming.html)


Value function

- $V_\pi(s)$
- how good is this state
- $V_{\pi}(s) = \mathbf{E}[G_t | s_t]$
- expected return when in state $s$, following policy $\pi$

Action-value function

- $Q_\pi(s,a)$
- how good is this action
- $Q_{\pi}(s,a) = \mathbf{E}[G_t | s_t, a_t]$
- expected return when in state $s$, taking action $a$, following policy $\pi$


## Value functions are oracles

Prediction of the future
- predict return

Always conditioned on a policy
- return depends on future actions

We don't know this function
- agent must learn it
- once we learn it – how will it help us to act?


<img src="assets/images/gen_policy_value_func.png"
	width="50%"
	height="50%">

``` python
def greedy_policy(state):

    #  get the Q values for each state_action pair
    q_values = value_function.predict(state)

    # select action with highest Q
    action = np.argmax(q_values)

    return action
```


## Prediction versus control

Prediction / approximation
- predicting return for given policy

Control
- the optimal policy
- the policy that maximizes expected future discounted reward

Prediction helps us to do control


## Approximation

To approximate a value function we can use one of the methods we looked at in the first section
- lookup table
- linear function
- non-linear function

Tables and linear functions are appropriate with some functions
- depends on agent and environment

Modern reinforcement learning is based on using neural networks
- convolution for learning from pixels
- feedforward for learning from n-d arrays


## Approximation methods

Let's look at three different methods for approximation

1. dynamic programming
2. Monte Carlo
3. temporal difference

We are creating targets to learn from
- we are labelling our data


## Dynamic programming

Imagine you had a perfect environment model

- the state transition function $P(s'|s,a)$
- the reward transition function $R(r|s,a,s')$

Can we use our perfect environment model for value function approximation?


##  Bellman Equation

Bellman's contribution is remembered by the Bellman Equation

$$ V\_{\pi}(s) = r + \gamma V\_{\pi}(s') $$

$$ Q\_{\pi}(s,a) = r + \gamma Q\_{\pi}(s', a') $$

- value of being in state $S$ = reward $r$ + discounted value of next state $s'$
- bootstrapped


<img src="assets/images/dp_example.png"
	width="80%"
	height="80%">

We can perform iterative backups of the expected return for each state
- probabilities here depend both on the environment and the policy


## Dynamic programming backup

The return for all terminal states is zero

$$V(s_2) = V(s_4) = 0$$

We can then express the value functions for the remaining two states

$$V(s\_3) = P\_{34}[r\_{34} + \gamma V(s\_4)]$$

$$V(s\_3) = 1 \cdot [5 + 0.9 \cdot 0] = 5 $$

$$V(s\_1) = P\_{12}[r\_{12} + \gamma V(s\_2) + P\_{13}[r\_{13} + \gamma V(s\_3)]$$

$$V(s\_1) = 0.5 \cdot [1 + 0.9 \cdot 0] + 0.5 \cdot [2 + 0.9 \cdot 5] = 3.75 $$


## Dynamic programming

Our value function approximation depends on
- our policy - what actions we pick
- the environment - where our actions take us and what rewards we get
- our current estimate of $V(s')$

A dynamic programming update is expensive
- our new estimate $V(s)$ depends on the value of all other states
- choosing which states to update can help (asynchronous dynamic programming)


## Policy iteration

Iterate between

- improving value function accuracy
- improving the policy

## Value iteration

Do it all in a single step

$$V(s) = max\_{a} \sum_{s',r} P(s',r|s,a)[r+\gamma V(s')]$$


## Dynamic programming summary

Requires a perfect environment model
- access to the full distribution over next states and rewards

Full backups
- update based on the probability distribution over all possible next states
- expectation based backups

Bootstrapped
- Bellman Equation to update our value function


## Monte Carlo

Monte Carlo methods
- finding the expected value of a function of a random variable

No environment model
- learn from sampled transitions

No bootstrapping
- we take the average of the true discounted return

Episodic only
- need to measure the experienced discounted return


## Monte Carlo

Estimate the value of a state by averaging the true discounted return observed after each visit to that state

As we run more episodes, our estimate should converge to the true expectation

Low bias & high variance - why?


## Monte Carlo

High variance
- we need to sample enough episodes for our averages to converge
- can be a lot for stochastic or path dependent environments

Low bias
- we are using actual experience
- no chance for a bootstrapped function to mislead us


## Lookup table Monte Carlo
```python
returns = defaultdict(list)

episode = run_full_episode(env, policy)

for experience in episode:
    returns = episode.calc_return(experience)

    returns[experience.state].append(return)

value_estimate = np.mean(returns[state])
```


## Monte Carlo can focus

Computational expense of estimating the value of state $s$ is independent of the number of states $\mathcal{S}$
- because we use experienced state transitions
- can focus on high value states in large state spaces (ignore the rest of the space)

<img src="assets/images/mc_example.png"
	width="80%"
	height="80%">


## Monte Carlo

Learn from actual or simulated experience
- no environment model

No bootstrapping
- use true discounted returns sampled from the environment

Episodic problems only
- no learning online

Ability to focus on interesting states and ignore others

High variance & low bias


## Temporal difference

Learn from samples of experience
- like Monte Carlo
- no environment model

Bootstrap
- like dynamic programming
- learn online

Episodic & non-episodic problems


## Temporal difference error

$$ \text{error} = r + \gamma V(s') - V(s) $$

Update rule for a table

$$ V(s\_t) \leftarrow V(s\_{t}) + \alpha [R\_{t+1} + \gamma \cdot V(s\_{t+1}) - V(s\_t)] $$

Update rule for a neural network

$$ \nabla_{\theta} (r + \gamma Q(s',a;\theta) - (Q(s,a;\theta))^{2}$$


<img src="assets/images/td_example.png" width="80%" height="80%">

$$ V(s\_1) \leftarrow V(s\_1) + \alpha [ r\_{23} + \gamma V(s\_3) - V(s\_1) ] $$


## How do we use the Bellman Equation?

Create targets for learning

- train a neural network by minimizing the difference between the network output and the correct target
- improve our approximation of a value function we need to create a targets for each sample of experience
- minimize a loss function

$$ error = target - approximation $$

For an experience sample of $(s, a, r, s')$

$$ error = r + Q(s',a) - Q(s,a) $$


## Q-Learning

Function approximation and data labelling are a means to an end
- control is what we really want
- optimal actions

Q-Learning
- off-policy control
- based on the action-value function $Q(s,a)$

Why might we want to learn $Q(s,a)$ rather than $V(s)$?


## $V(s)$ versus $Q(s,a)$

Imagine a simple MDP

$$ \mathcal{S} = \{s_1, s_2, s_3\} $$
$$ \mathcal{A} = \{a_1, a_2\} $$

Our agent finds itself in state $s_2$

We use our value function $V(s)$ to calculate the value of all possible next states

$$V(s_1) = 10$$
$$V(s_2) = 5$$
$$V(s_3) = 20$$

Which action should we take?


## $V(s)$ versus $Q(s,a)$

Now imagine we had

$$Q(s\_{2}, a\_1) = 40$$
$$Q(s\_{2}, a\_2) = 20$$

It's now easy to pick the action that maximizes expected discounted return

$V(s)$ tells us how good a state is
- need additional info (state transition probabilities) to select an action

$Q(s,a)$ tells us how good an action is
- select best action by $argmax$ across the action space


## SARSA

SARSA = state, action, reward, next state, next action

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s', a') - Q(s,a)] $$

- on-policy control
- we use every element from our experience tuple $(s,a,r,s')$
- and also $a'$ - the next action selected by our agent

Why is the value function learnt by SARSA on-policy?
- we learn about the action $a'$ that our agent choose to take


## Q-Learning

Off-policy control

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \underset{a}{\max} Q(s', a) - Q(s,a)] $$

- use every element from our experience tuple $(s,a,r,s')$

We take the maximum over all possible next actions
- we don't need to know what action our agent took next (i.e. $a'$)
- learn the optimal value function while following a sub-optimal policy

Don't learn $Q_{\pi}$ - learn $Q^*$ (the optimal policy)


## SARSA versus Q-Learning

<img src="assets/images/sarsa_ql.png" height="60%" width="60%">


## Q-Learning

Select optimal actions by $argmax$ across the action space

$$action = \underset{a}{argmax}Q(s,a)$$
- the $argmax$ limits Q-Learning to discrete action spaces only

Issues with the argmax
- positively biased (see DDQN)
- aggressive


## Lets talk about the $argmax$

Small changes in $Q(s,a)$ estimates can drastically change the policy

$$Q(s_1, a_1) = 10 $$
$$Q(s_1, a_2) = 11 $$

- then we do some learning and our estimates change

$$Q(s_1, a_1) = 12 $$
$$Q(s_1, a_2) = 11 $$

- now our policy is completely different!

For a given approximation of $Q(s,a)$ acting greedy is deterministic
- how then do we explore the environment?


## $\epsilon$-greedy exploration

A common exploration strategy is the epsilon-greedy policy

```python
def epsilon_greedy_policy():
    if np.random.rand() < epsilon:
        #  act randomly
        action = np.random.uniform(action_space)

    else:
        #  act greedy
        action = np.argmax(Q_values)

    return action
```

$\epsilon$ is decayed during experiments to explore less as our agent learns (i.e. to exploit)


## Exploration strategies

Boltzmann (a softmax)
- temperature being annealed as learning progresses

Bayesian Neural Network
- a network that maintains distributions over weights -> distribution over actions
- this can also be performed using dropout to simulate a probabilistic network

Parameter noise
- adding adaptive noise to weights of network

<figcaption>[Action-Selection Strategies for Exploration](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf)</figcaption>

<figcaption>[Plappert et al. (2018) Paramter Space Noise for Exploration](https://arxiv.org/pdf/1706.01905.pdf)</figcaption>


<img src="assets/images/action_selection_exploration.png" height="70%" width="70%">
<figcaption>[Action-Selection Strategies for Exploration](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf)</figcaption>


## Deadly triad

Emergent phenomenon that produces instability.  Driven by:

1. off-policy learning - to learn about the optimal policy while following an exploratory policy
2. function approximation - for scalability and generalization
3. bootstrapping - computational & sample efficiency

It's not clear what causes instability
- dynamic programming can diverge with function approximation - on-policy divergence
- prediction can diverge
- linear functions can be unstable

<figcaption>Sutton & Barto</figcaption>


Up until 2013 the deadly triad limited the use of neural networks with Q-Learning

Then came DeepMind & DQN...


# DQN

why so significant?

which two key innovations stabilized learning?

what is the shape of the output layer?


## DQN

In 2013 a small London startup published a paper
- an agent based on Q-Learning
- superhuman level of performance in three Atari games

In 2014 Google purchased DeepMind for around £400M

This is for a company with
- no product
- no revenue
- no customers
- a few world class employees


<img src="assets/images/2013_atari.png" height="70%" width="70%">
<img src="assets/images/2015_atari.png" height="70%" width="70%">


## Significance

Deep reinforcement learning
- convolution

End to end
- learning from raw pixels

Ability to generalize
- same algorithm, many games


## Reinforcement learning to play Atari

State
- last four screens concatenated together - allows infomation about movement
- grey scale, cropped & normalized

Reward
- game score
- clipped to [-1, +1]

Actions
- joystick buttons (a discrete action space)


<img src="assets/images/atari_results.png" height="50%" width="50%">


<img src="assets/images/atari_func.png" height="90%" width="90%">


<img src="assets/images/atari_sea.png" height="70%" width="70%">


## Two key innovations in DQN

Experience replay

Target network

- both improve learning stability

<img src="assets/images/stability.png" height="70%" width="70%">


Experience replay

<img src="assets/images/exp_replay.png" height="70%" width="70%">


## Experience replay

Experience replay helps to deal with our non-iid dataset
- randomizing the sampling of experience -> more independent
- brings the batch distribution closer to the true distribution -> more identical

Data efficiency
- we can learn from experience multiple times

Allows seeding of the memory with high quality experience

Can only do this because we can learn off-policy!


## Biological basis for experience replay

Hippocampus may support an experience replay process in the brain

- time compressed reactivation of recently experienced trajectories during off-line periods
- provides a mechanism where value functions can be efficiently updated through interactions with the basal ganglia

<figcaption>Mnih et. al (2015)</figcaption>


## Target network

Parameterize two separate neural networks (identical structure) - two sets of weights $\theta$ and $\theta^{-}$

Original Atari work copied the online network weights to the target network every 10k - 100k steps

Can also use a small factor tau ($\tau$) to smoothly update weights at each step


## Target network

Changing value of one action changes value of all actions & similar states
- bigger networks less prone (less aliasing aka weight sharing)

$$L(\theta\_{i}) = [r + \gamma  \cdot max\_{a'} Q(s',a;\theta\_{i}^{-}) - Q(s,a;\theta_{i})]$$

Stable training
- no longer bootstrapping from the same function, but from an old & fixed version of $Q(s,a)$
- reduces correlation between the target created for the network and the network itself


DQN algorithm

<img src="assets/images/DQN_algo.png" height="70%" width="70%">
<figcaption>Mnih et. al (2015)</figcaption>


## Timeline

- 1986 - Backprop by Rumelhart, Hinton & Williams in multi layer nets
- 1989 - Q-Learning (Watkins)
- 1992 - Experience replay (Lin)
- 2010 - Tabular Double Q-Learning
- ---
- 2010's - GPUs used for neural networks
- 2013 - DQN
- 2015 - Prioritized experience replay
- 2016 - Double DQN (DDQN)
- 2017 - Distributional Q-Learning
- 2018 - Rainbow


# Beyond DQN


<img src="assets/images/unified_view.png" height="60%" width="60%" align="top">
<figcaption> [Sutton - Long-term of AI & Temporal Difference Learning](https://www.youtube.com/watch?v=EeMCEQa85tw)</figcaption>


<img src="assets/images/effect_bootstrap.png" height="60%" width="60%" align="top">
<figcaption> [Sutton - Long-term of AI & Temporal Difference Learning](https://www.youtube.com/watch?v=EeMCEQa85tw)</figcaption>


## Eligibility traces

Family of methods between Temporal Difference & Monte Carlo

Eligibility traces allow us to assign TD errors to different states
- can be useful with delayed rewards or non-Markov environments
- requires more computation
- squeezes more out of data

Allow us to trade-off between bias and variance


## n-step returns

In between TD and MC exist a family of approximation methods known as **n-step returns**

<img src="assets/images/bias_var.png" height="60%" width="60%" >
<figcaption>Sutton & Barto</figcaption>


## Forward and backward view

We can look at eligibility traces from two perspectives

The forward view
- helpful for understanding the theory

The backward view
- can be put into practice


The forward view

- assign future returns to the current state

<img src="assets/images/forward_view.png" height="80%" width="80%" >
<figcaption>Sutton & Barto</figcaption>


The backward view

- assign TD error to previous states

<img src="assets/images/backward_view.png" height="80%" width="80%" >
<figcaption>Sutton & Barto</figcaption>


## The backward view

The backward view approximates the forward view
- forward view is not practical (requires knowledge of the future)

We need to remember which states we visited in the past
- it requires an additional variable in our agents memory
- eligibility trace $e_{t}(s)$


<img src="assets/images/traces_grid.png" height="70%" width="70%" >
<figcaption>Sutton & Barto</figcaption>

- one step method would only update the last $Q(s,a)$
- n-step method would update all $Q(s,a)$ equally
- eligibility traces updates based on how recently each $Q(s,a)$ was experienced


## Prioritized Experience Replay

Naive experience replay
- randomly samples experience

Some experience is more useful for learning than others
- we can measure how useful experience is by the temporal difference error

$$ error = r + \gamma Q(s', a) - Q(s,a) $$

TD error is an indication of how useful a transition is for learning
- measures surprise


## Prioritized Experience Replay

Non-random sampling introduces two problems

Loss of diversity
- we will only sample from high TD error experiences
- solve by making the prioritization stochastic

Introduce bias
- non-independent sampling
- correct bias using importance sampling


## DDQN

DDQN = Double Deep Q-Network
- first introduced in a tabular setting in 2010
- reintroduced in the content of DQN in 2016

Tabular Q-Learning used two different approximations of $Q(s,a)$
- one to quantify the value of actions
- one to select actions

Double Q-Learning aims to overcome the maximization bias of Q-Learning


## Maximization bias

Imagine a state where $Q(s,a) = 0$ for all $a$

Our estimates are normally distributed above and below 0

<img src="assets/images/max_bias.png" height="70%" width="70%" >


## DDQN

Use the target network as a different approximation of $Q(s,a)$

Original DQN target
$$ r + \gamma \underset{a}{\max} Q(s,a;\theta^{-}) $$

DDQN target
$$ r + \gamma Q(s', \underset{a}{argmax}Q(s',a; \theta); \theta^{-}) $$

- select the action according to the online network
- quanitfy the value that action using the target network

Both Q networks still have noise - but the noise is decorrelated


<img src="assets/images/2015_DDQN_results.png" height="90%" width="90%" >
<figcaption>van Hasselt et. al. (2015)</figcaption>


## Rainbow

What if we combined all the improvements together?

All the various improvements to DQN address different issues
- DDQN - overestimation bias
- prioritized experience replay - sample efficiency
- duelling - generalize across actions
- multi-step bootstrap targets - bias variance trade-off
- distributional Q-learning - learn categorical distribution of $Q(s,a)$
- noisy DQN - stochastic layers for exploration


<img src="assets/images/rainbow_fig1.png" height="60%" width="60%">
<figcaption>Median human-normalized performance across 57 Atari games. We compare our integrated agent (rainbow colored) to DQN (grey) and six published baselines. Note that we match DQN’s best performance after 7M frames, surpass any baseline within 44M frames, and reach substantially improved final performance. Curves are smoothed with a moving average over 5 points.</figcaption>
<figcaption>van Hasselt et. al. (2017)</figcaption>


<img src="assets/images/rainbow_hyper.png" height="60%" width="60%">
<figcaption>van Hasselt et. al. (2017)</figcaption>


# Policy optimization
<section id="policy-optimization">

how do I parametrize a continuous action space?

how is the score function used to find the policy gradient?

why use baselines?

what do the actor and critic do?


## Policy gradients

Value function methods generate a policy from a learnt value function

$$ a = \underset{a}{argmax} Q(s,a) $$

Policy optimization parametrizes and improves the policy directly

$$ a \sim \pi(a_t|s_t;\theta) $$


<iframe width="560" height="315" src="https://www.youtube.com/embed/PtAIh9KSnjo?rel=0&amp;showinfo=0&amp;start=2905" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


## Motivations for policy gradients

High dimensional action spaces
- robotics

Optimize what we care about
- optimize return 

Less sample efficient
- usually on-policy learning


## Motivation - high dimensional action spaces

Q-Learning requires a discrete action space to argmax across

Lets imagine controlling a robot arm in three dimensions in the range [0, 90] degrees

This corresponds to approx. 750,000 actions a Q-Learner would need to argmax across

We also lose shape of the action space by discretization


<img src="assets/images/disc_cont_act.png" height="70%" width="70%">


## Motivation - optimize return directly

When learning value functions our optimizer is working towards improving the predictive accuracy of the value function
- our gradients point in the direction of predicting return

This isn't what we really care about - we care about maximizing return

Policy methods optimize return directly
- changing weights according to the gradient that maximizes future reward
- aligning gradients with our objective (and hopefully a business objective)


## Motivation - simplicity

Sometimes it's easier to pick an action
- rather than to quantify return for each action, then pick action

<img src="assets/images/motivation_simple.png" height="60%" width="60%">


## Parametrizing policies

<img src="assets/images/discrete_policy.png" height="50%" width="50%">


## Parametrizing policies

<img src="assets/images/cont_policy.png" height="80%" width="80%">


## Policy gradients without equations

We have a parametrized policy
- a neural network that outputs a distribution over actions

How do we improve it - how should we change our parameters?
- take actions that get more reward
- favour probable actions

Reward function is not known
- so we can't just take a gradient wrt reward


## Policy gradients with a few equations

Our policy is a probability distribution over actions

How do we improve it?
- take actions that get more reward
- favour probable actions

Reward function is not known
- but we can find the gradient of expected reward
- find gradients to increase return

$$\nabla\_{\theta} \mathbf{E}[G\_t] = \mathbf{E}[\nabla\_{\theta} \log \pi(a|s) \cdot G\_t]$$


## The score function

The score function allows us to get the gradient of the expectation of a function
- i.e. expected reward
- derived using the log-likelihood ratio trick

Expectations are averages
- this allows us to use sample based methods (no need for full transition probabilities as in dynamic programming)
- i.e. we can approximate the RHS using samples

$$\nabla\_{\theta} \mathbf{E}[f(x)] = \mathbf{E}[\nabla\_{\theta} \log P(x) \cdot f(x)]$$

$$\nabla\_{\theta} \mathbf{E}[G\_t] = \mathbf{E}[\nabla\_{\theta} \log \pi(a|s) \cdot G\_t]$$


## Deriving the score function

<img src="assets/images/score_derivation.png" height="80%" width="80%">


- start with the definition of expectation

$$\mathbf{E}\_{x}[f(x)] = \sum\_{x} p(x) \cdot f(x)$$

- take the gradient with respect to our neural network weights

$$\nabla\_{\theta} \mathbf{E}\_{x}[f(x)] = \nabla\_{\theta} \sum_{x} p(x) \cdot f(x)$$

- swap sum and gradient

$$\nabla\_{\theta} \mathbf{E}\_{x}[f(x) = \sum\_{x} \nabla\_{\theta}  p(x) \cdot f(x)$$

- multiply and divide by $p(x)$

$$\nabla\_{\theta} \mathbf{E}\_{x}[f(x)] = \sum\_{x} \frac{p(x)}{p(x)} \cdot \nabla\_{\theta} p(x) \cdot f(x)$$

<figcaption>[Deep Reinforcement Learning: Pong from Pixels - Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/)>


- using the fact that $\nabla\_{\theta} log(p(x)) = \frac{\nabla\_{\theta} p(x)}{p(x)}$

$$\nabla\_{\theta} \mathbf{E}\_{x}[f(x)] = \sum\_{x} p(x) \cdot \nabla\_{\theta} log (p(x)) \cdot f(x)$$

- using the definition of an expectation $\sum\_{x} p(x) \cdot f(x) = \mathbf{E}\_{x} [f(x)]$

We end up with the ability to find the gradient of a function we don't have access too

$$\nabla\_{\theta} \mathbf{E}\_{x}[f(x)] = \mathbf{E}\_{x}[\nabla\_{\theta} log(p(x)) \cdot f(x)]$$

- but we must be able to sample from it - to approximate the expectation on the RHS

<figcaption>[Deep Reinforcement Learning: Pong from Pixels - Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/)


## Policy gradient intuition

$$\nabla\_{\theta} \mathbf{E}[G\_t] = \mathbf{E}[\nabla\_{\theta} \log \pi(a|s) \cdot G\_t]$$

$\log \pi(a_t|s_t;\theta)$
- how probable was the action we picked
- we want to reinforce actions we thought were good

$ G_t $
- how good was that action
- we want to reinforce actions that were actually good


## Training a policy

We use the score function to get a gradient

```python
gradient = log(probability of action) * expected_return

gradient = log(policy) * expected_return
```

The score function limits us to on-policy learning
- we need to calculate the log probability of the action taken by the policy



## REINFORCE

Williams (1992) - Simple statistical gradient-following algorithms for connectionist reinforcement learning

Different methods to approximate the return $G_t$
- REINFORCE uses a Monte Carlo estimate
- we use actual sampled discounted reward

Using a Monte Carlo approach comes with all the problems we saw earlier
- high variance
- no online learning
- requires episodic environment


## Baseline

We can introduce a baseline function

$$\log \pi(a_t|s_t;\theta) \cdot (G_t - B(s_t; w))$$

- this reduces variance (smaller gradients) without introducing bias
- a natural baseline is the value function (weights $w$)

This also gives rise to the concept of advantage

$$A\_{\pi}(s\_t, a\_t) = Q\_{\pi}(s\_t, a\_t) - V\_{\pi}(s\_t)$$

- how much better this action is than the average action


## Actor Critic

Actor Critic brings together value functions and policy gradients

We parametrize two functions
- actor = policy
- critic = value function

We update our actor (i.e. the behaviour policy) in the direction suggested by the critic

$$\nabla\_{\theta} \mathbf{E}[G] = \mathbf{E}[\nabla\_{\theta} log(\pi_{\theta}(a|s) \cdot Q(s,a)]$$

```python
while not done:
	action = agent.act(observation)
	next_observation, reward, done, info = env.step(action)

	agent.update_critic()

	agent.update_policy()
```

## Advantage Actor Critic (A2C)

Advantage

$$A\_{\pi}(s, a) = Q\_{\pi}(s, a) - V\_{\pi}(s)$$

- how much better this action is than the average action

We don't have to learn an additional network

$$Q\_{\pi}(s,a) = r + \gamma V\_{\pi}(s')$$

$$A\_{\pi}(s,a) = r + \gamma V\_{\pi}(s') - V\_{\pi}(s)$$


## Asynchronous Advantage Actor Critic (A3C)

Multiple agents learning in parallel
- update based on fixed length's of experience (say 20 timesteps)
- share parameters between value and policy networks
- asynchronous updates of parameters

Different policies can be run in each environment
- exploration


## Natural Policy Gradients, TRPO and PPO

All three of these papers build on the same idea - that we want to constrain policy updates to get more stable learning

- Natural Policy gradients - rely on a computationally intense second order derivative method (inverse of the Fisher Infomation matrix)
- TRPO - uses the KL-divergence to hard constrain policy updates (avoids calculating the Fisher Infomation matrix, but uses Conjugate Gradient to solve a constrained optimization problem)
- PPO - uses clipped probability ratios to constrain policy updates

By constraining the policy update, we can learn off-policy


# Model based reinforcement learning
<section id="model-based">


## All modern reinforcement learning is model based

Poor sample efficiency means simulation is required

A simulator is an environment model!

Lots of research aimed at developing model-free algorithms

Model-based is less well developed but has advantages
- for a perfect, distributional model can use dynamic programming
- learning a model is a key challenge


## Why learn and use environment models?

We use low dimensional mental models to represent the world around us.

Our brain learns abstract representations of spatial and temporal infomation.  Evidence also suggests that perception itself is governed by an internal prediction of the future, using our mental models.

This predictive model can be used to perform fast reflexive behaviours when we face danger.


## Kinds of models

Sample versus distributional

Local versus global


## If you have a perfect model


## If you need to learn the model


## Monte Carlo Tree Search


## Ha & Schmidhuber (2018) World Models

[paper](https://arxiv.org/pdf/1803.10122.pdf) - [blog post](https://worldmodels.github.io/) - [blog post appendix](https://worldmodels.github.io/#appendix)

Learning within a dream - agent learns using a generative environment model
- vision that compresses pixels into a latent space
- LSTM memory that makes predictions of the latent space
- low dimensional controller that maps from vision + LSTM to action

Solved a previously unsolved car racing task (plus VisDoom)
- continuous action space
- learning from pixels

| Component | Model |Parameter Count |
|---|---|---|
|Vision| VAE|4,348,547|
|Memory|MDN-RNN|422,368|
|Controller|CMA-ES|867|

## Compression

Compression used to
- reduce the dimensionality of the observation into a latent representation ($z$)
- compress the latent representation over time

Allows a compact linear policy to be used for control 
- the policy parameters are learnt using an evolutionary algorithm


<img src="assets/images/wm_image_generation.png" height="90%" width="90%">
<figcaption>Actual and reconstructed observations produced by the autoencoder.  Note the reconstructed observation is not used by the controller - it is shown here to compare the quality with the actual observation.</figcaption>


<img src="assets/images/wm_flow.png" height="80%" width="80%">
<figcaption></figcaption>


<img src="assets/images/wm_fig1.png" height="90%" width="90%">
<figcaption>The agent consists of three components - Vision (V), Memory (M), and Controller (C)</figcaption>


<img src="assets/images/wm_learning.png" height="100%" width="100%">
<figcaption>Agent can learn both from the real environment and from inside it's own dream</figcaption>


## Vision - Variational Auto Encoder

A network that compresses and reproduces the observation of the environment
- the reconstructed observation is not used
- the compressed version of the observation (aka the latent representation) is used as input to both the memory and the controller


## Memory - Mixed Density Recurrent Network

Predicts the latent representation of the observation
- not the raw observation
- compresses over time
- a predictive model of the future vectors that V is expected to produce.

Mixed density
- stochastic
- outputs probability density
- outputs a mixture of Gaussians


<img src="assets/images/wm_memory.png" height="90%" width="90%">
<figcaption>Outputs the parameters of a mixture of Gaussians used to sample a prediction of the compressed representation of state $z$</figcaption>


## Controller - CMA-ES

Controller is linear
- single layer that maps memory hidden state and latent representation of observation to action
- allows simpler optimization

CMA-ES = covariance matrix adaptation evolution strategy 
- derivative free
- good for up to a few thousand parameters


## Performance

<img src="assets/images/wm_performance.png" height="80%" width="80%">


# AlphaGo to AlphaZero
<section id="alphago">


<iframe width="560" height="315" src="https://www.youtube.com/embed/8tq1C8spV_g?rel=0&amp;showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


## IBM Deep Blue

First defeat of a world chess champion by a machine in 1997

<img src="assets/images/DeepBlue.png" height="60%" width="60%">


## Deep Blue versus AlphaGo

Deep Blue
- hand-crafted by programmers & chess grandmasters
- big lookup table

AlphaGo
- learnt from human moves & self play
- reduced search width and depth using neural networks


## Why Go?

Long held as the most challenging classic game for artificial intelligence
- massive search space
- more legal positions than atoms in universe
- difficult to evaluate positions & moves
- sparse & delayed reward

Played and studied for thousands of years
- best human play is high level
- datasets of online matches


## Components of the AlphaGo agent

Three policy networks $\pi(s)$
- fast rollout policy network – linear function
- supervised learning policy – 13 layer convolutional NN
- reinforcement learning policy – 13 layer convolutional NN

One value function $V(s)$
- convolutional neural network

Combined together using Monte Carlo tree search


## Learning

<img src="assets/images/ag_learning.png" height="110%" width="110%">


## Monte Carlo Tree Search

Value & policy networks combined using MCTS

Basic idea = analyse most promising next moves

Planning algorithm
- simulated (not actual experience)
- roll out to end of game (a simulated Monte Carlo return)


## Monte Carlo Tree Search

<img src="assets/images/MCTS_one.png" height="110%" width="110%">


## Monte Carlo Tree Search in AlphaGo

![fig](assets/images/MCTS_two.png)


## Monte Carlo Tree Search in AlphaGo

![fig](assets/images/MCTS_AG_one.png)


## Monte Carlo Tree Search in AlphaGo

![fig](assets/images/MCTS_AG_two.png)


## Monte Carlo Tree Search in AlphaGo

![fig](assets/images/MCTS_AG_three.png)


## AlphaGo Zero





## AlphaZero






## AlphaGo, in context – Andrej Karpathy

Convenient properties of Go
- fully deterministic
- fully observed
- discrete action space
- access to perfect simulator
- relatively short episodes
- evaluation is clear
- *huge datasets of human play*
- energy consumption (human ≈ 50 W) 1080 ti = 250 W

<figcaption>[Medium blog post - originally written about AlphaGo, most applies to AlphaZero](https://medium.com/@karpathy/alphago-in-context-c47718cb95a5)</figcaption>



# Practical concers


## Deep RL doesn't work yet


## Deep RL that matters
