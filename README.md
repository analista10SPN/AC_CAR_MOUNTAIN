# Actor-Critic Implementation for Continuous Mountain Car Problem: A Modern Approach with PPO and GAE

#### By: Ariel Gonzalez Batista

#### Date: November 22, 2024

## Abstract

This study presents a modern Actor-Critic implementation for solving the Mountain Car Continuous control problem. Our approach integrates Proximal Policy Optimization (PPO) and Generalized Advantage Estimation (GAE) to achieve consistent convergence while effectively avoiding local minima traps. The implementation demonstrates robust learning behavior, achieving solution convergence in approximately 200 episodes with a 90% success rate. Through sophisticated reward shaping and careful architectural decisions, our solution overcomes the inherent challenges of sparse rewards and continuous action spaces.




https://github.com/user-attachments/assets/e78c4ae2-766b-4773-96a2-dd492733e3ab


## Index

1. **Introduction and Problem Statement**

   - 1.1. Reinforcement Learning Overview
   - 1.2. Actor-Critic Design
   - 1.3. The Continuous Mountain Car Problem
   - 1.4. Continuous Mountain Car State, Actions and Rewards

2. **Theoretical Framework**

   - 2.1. Actor-Critic Foundation
   - 2.2. Modern Enhancements

3. **Implementation Architecture**

   - 3.1. Network Design
   - 3.2. Architecture Design Rationale
   - 3.3. Training Process
   - 3.4. Exploration Strategy
   - 3.5. Advantage Estimation
   - 3.6. Training Loop Structure
   - 3.7 Implementation Parameters and Algorithm Overview

4. **Results Analysis**

   - 4.1. Training Dynamics
   - 4.2. Critic Learning Convergence
   - 4.3. Policy Evolution
   - 4.4. Performance Comparison
   - 4.5. Local Minima Analysis

5. **Discussion and Future Work**

   - 5.1. Architectural Insights
   - 5.2. Limitations
   - 5.3. Future Improvements

6. **References**

## 1. Introduction and Problem Statement

### 1.1 Reinforcement Learning Overview

Reinforcement learning (RL) is a computational approach to learning from interaction, where an agent learns to make decisions by interacting with an environment. The agent receives observations (states), takes actions, and obtains rewards, aiming to maximize its cumulative reward over time. Within the RL framework, Actor-Critic methods represent a powerful class of algorithms that combine the advantages of both value-based and policy-based methods.

### 1.2 Actor-Critic Design

Actor-Critic architectures decompose the learning process into two components: an actor that learns a policy for action selection, and a critic that evaluates the actor's choices through value estimation. This separation allows for variance reduction in policy gradients while maintaining the benefits of direct policy optimization. Our implementation builds upon this foundation, incorporating modern enhancements like PPO and GAE to improve learning stability and efficiency.

### 1.3 The Continuous Mountain Car Problem

The Mountain Car Continuous environment represents a fundamental challenge in reinforcement learning control problems. An underpowered car, positioned in a valley, must learn to leverage gravitational forces through oscillatory movements to reach a target position on a hill. The continuous nature of both state and action spaces introduces significant complexity to the learning task.

### 1.4 Continuous Mountain Car State, Actions and Rewards.

The environment's state space $\mathcal{S}$ consists of a two-dimensional continuous vector:

- Position $x \in [-1.2, 0.6]$: Represents the car's horizontal position
- Velocity $v \in [-0.07, 0.07]$: Represents the car's velocity

The action space $\mathcal{A}$ is a single continuous dimension:

- Force $a \in [-1.0, 1.0]$: Represents the engine force applied to the car

The reward function $R(s, a, s')$ is defined as:

- Base reward: $-0.1$ per timestep (encouraging faster solutions)
- Goal reward: $+100$ for reaching $x \geq 0.45$

We decided to use shaped rewards:

- Position improvement: $+10 \cdot (x_{t+1} - x_t)$ when moving uphill
- Velocity magnitude: $+5 \cdot (|v_{t+1}| - |v_t|)$ for building momentum
- Valley penalty: $-0.1$ when $|x| < 0.1$ to discourage staying at the bottom

![alt text](1_Y8gcUv2_XvKcMQxm7XDJrg.png)
_Figure 1: Mountain Car environment's key components and dimensions._

The environment presents three fundamental challenges. First, the reward structure is inherently sparse, providing significant feedback only upon reaching the goal position (x ≥ 0.45). Second, the continuous action space requires the agent to learn precise control policies rather than selecting from discrete actions. Third, the physical constraints of the car's limited engine power necessitate the development of momentum-building strategies, creating a complex relationship between immediate actions and delayed rewards.

## 2. Theoretical Framework

### 2.1 Actor-Critic Foundation

The Actor-Critic architecture represents a sophisticated approach to reinforcement learning that combines policy gradient methods with value function approximation. At its core, Actor-Critic methods utilize deep neural networks to approximate both the policy and value functions, leveraging the networks' capacity to learn complex mappings from high-dimensional state spaces to actions and values.

The framework decomposes the learning process into two interrelated components: an actor network that learns a policy distribution over actions, and a critic network that evaluates state values to guide policy improvement. These neural networks, typically implemented as multilayer perceptrons with several hidden layers, enable the agent to learn rich representations of the state space and develop sophisticated control strategies.

The foundational objective of our implementation maximizes the expected return:

$J(\theta) = E_{\pi_\theta}[\sum_{t=0}^{\infty} \gamma^t r_t]$

The policy gradient theorem provides the fundamental update direction:

$\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)A(s,a)]$

In our continuous action space, the policy $\pi_\theta(a|s)$ is represented by a Gaussian distribution with learned mean and standard deviation. This formulation enables the expression of a continuous range of actions while maintaining exploration through controlled stochasticity.

### 2.2 Modern Enhancements

Our implementation incorporates two critical modern enhancements to the traditional Actor-Critic framework: Proximal Policy Optimization (PPO) and Generalized Advantage Estimation (GAE). These techniques are particularly crucial for the Mountain Car problem due to its challenging exploration requirements and delayed rewards.

#### Proximal Policy Optimization (PPO)

Traditional policy gradient methods can be unstable due to large policy updates that drastically change the behavior of the agent. This is particularly problematic in the Mountain Car environment where successful policies require precise timing of oscillatory actions. PPO addresses this by introducing a constrained optimization objective:

$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]$

Where the probability ratio $r_t(\theta)$ measures how much the current policy has changed from the old policy:

$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

The PPO mechanism provides crucial stability through several key mechanisms. First, the clipping parameter $\epsilon$ (set to 0.2) prevents destructively large policy updates by limiting how much the policy can change in a single step. This is implemented through a dual-objective optimization where the final loss is computed as the maximum of the clipped and unclipped objectives:

```python
actor_loss1 = -ratio * advantages
actor_loss2 = -torch.clamp(ratio, 0.8, 1.2) * advantages
actor_loss = torch.max(actor_loss1, actor_loss2).mean()
```

Furthermore, by constraining updates to be within $[1-\epsilon, 1+\epsilon]$ of the old policy, PPO creates an adaptive trust region that preserves successful oscillatory behaviors once discovered. This trust region mechanism allows for gradual refinement of action timing while preventing catastrophic forgetting of effective strategies. The clipped objective ensures that exploration remains stable by preventing sudden collapse to deterministic policies, excessive growth of action variances, and loss of learned momentum-building strategies.

#### Generalized Advantage Estimation (GAE)

The Mountain Car problem presents a significant challenge in credit assignment due to the delayed nature of rewards - success requires a precise sequence of actions over many timesteps. GAE provides a sophisticated solution for estimating the advantage function $A(s,a)$, which measures how much better an action is compared to the average:

$A^{GAE}(\lambda) = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$

which can be computed recursively as:

$A^{GAE}_t(\lambda) = \delta_t + \gamma\lambda A^{GAE}_{t+1}(\lambda)$

The temporal difference error $\delta_t$ captures immediate rewards and value changes:

$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

The GAE framework introduces a crucial bias-variance trade-off through its $\lambda$ parameter, set to 0.95 in our implementation. Higher $\lambda$ values give more weight to actual returns, which proves beneficial for learning long-term momentum strategies, while lower values rely more on value estimates, effectively reducing variance in updates. This is implemented recursively in our code:

```python
delta = rewards[t] + gamma * next_val - values[t]
gae = delta + gamma * gae_lambda * gae  # Computed recursively
```

GAE's approach to credit assignment proves particularly effective by propagating rewards back through time while carefully weighing immediate and future consequences of actions. This mechanism accounts for the full sequence of momentum-building actions necessary for successful task completion. The framework also significantly improves value function learning by providing more stable advantage estimates for policy updates, better handling of delayed rewards, and reduced variance in value function gradients. This is reflected in our value function update:

```python
returns = gae + values[t]  # Using GAE for value function targets
critic_loss = (returns - values).pow(2).mean()
```

#### Integration Benefits

The synergistic combination of PPO and GAE creates a robust learning framework specifically suited to address the key challenges of the Mountain Car environment. PPO's policy update constraints work in concert with GAE's sophisticated advantage estimation to enable stable learning of complex oscillatory strategies. This combination proves particularly effective at handling the sparse reward structure while maintaining consistent exploration.

## 3. Implementation Architecture

### 3.1 Network Design

The architecture implements a unified ActorCritic class where both policy and value functions are implemented as separate neural networks within the same class, each with their own parameters and specialized outputs.

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(self._init_weights)
```

The network architecture employs several key design elements:

First, orthogonal initialization with a gain of 1.414 ensures proper gradient flow through the ReLU activations:

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=1.414)
        module.bias.data.zero_()
```

Second, the actor branch outputs both mean and log standard deviation for the action distribution:

```python
def get_action_dist(self, state):
    actor_output = self.actor(state)
    mean, log_std = torch.chunk(actor_output, 2, dim=-1)
    mean = torch.tanh(mean)  # Bound mean to [-1, 1]
    std = torch.clamp(log_std.exp(), min=0.2, max=1.0)
    return torch.distributions.Normal(mean, std)
```

The tanh activation on the mean ensures actions remain within the environment's bounds, while standard deviation clamping prevents both premature convergence and excessive exploration.

The critic branch provides state-value estimates through a single output node:

```python
def get_value(self, state):
    return self.critic(state)
```

This unified class architecture offers several advantages. The separate networks maintain full independence in learning their respective features while being organized in a single cohesive implementation. The parameter count remains manageable while maintaining sufficient capacity for the control task.

### 3.2 Architecture Design Rationale

The decision to implement the Actor and Critic as separate networks within a unified class stems from several theoretical and practical considerations. Our approach maintains complete independence between these components while organizing them in a clean, unified implementation.

The mathematical foundation for this separation becomes apparent when we examine the distinct architectures and optimization objectives of each network.

The actor network implements a Gaussian policy with state-dependent mean and standard deviation:

$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$

Where the network outputs both $\mu_\theta(s)$ and $\log \sigma_\theta(s)$ through separate parameters:

$[\mu_\theta(s), \log \sigma_\theta(s)] = W_3(\text{ReLU}(W_2(\text{ReLU}(W_1s + b_1)) + b_2)) + b_3$

The actor optimizes the PPO objective:

$J_{actor}(\theta_\pi) = \mathbb{E}t[\min(r_t(\theta\pi)A_t, \text{clip}(r_t(\theta_\pi), 1-\epsilon, 1+\epsilon)A_t)] + \alpha H(\pi_\theta)$

Where:

$\frac{r_t(\theta_\pi) = \pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio
$A_t$ is the advantage estimate
$H(\pi_\theta)$ is the policy entropy
$\alpha = 0.02$ is the entropy coefficient

The critic network maintains its own independent architecture to estimate state values:

$V_{\theta_V}(s) = W'_3(\text{ReLU}(W'_2(\text{ReLU}(W'_1s + b'_1)) + b'_2)) + b'_3$

Optimizing the value estimation error:
$J_{critic}(\theta_V) = \mathbb{E}[(V_{\theta_V}(s) - R_t)^2]$

Where $R_t$ is the GAE-computed return:

$R_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$
$\delta_t = r_t + \gamma V_{\theta_V}(s_{t+1}) - V_{\theta_V}(s_t)$

In the Mountain Car environment, these networks learn fundamentally different mappings:

The actor's policy mapping must capture precise control decisions:

$a_t = \mu_\theta(s_t) + \sigma_\theta(s_t)\epsilon, \epsilon \sim \mathcal{N}(0,1)$

With exploration noise during early training:

$a_t = \text{clip}(\mu_\theta(s_t) + \sigma_\theta(s_t)\epsilon + \eta_t, -1, 1)$
Where $\eta_t \sim \mathcal{N}(0, \epsilon_t)$ and $\epsilon_t = \max(\epsilon_{min}, \epsilon_0 \cdot \alpha^t)$

The critic's value mapping estimates long-term returns:

$V_{\theta_V}(s_t) \approx \mathbb{E}{\pi\theta}[\sum_{k=0}^{\infty} \gamma^k r_{t+k}|s_t]$

The separate optimization of these networks is reflected in their different learning rates:

$\theta_\pi \leftarrow \theta_\pi - 3\times10^{-4} \nabla_{\theta_\pi} J_{actor}$
$\theta_V \leftarrow \theta_V - 1\times10^{-3} \nabla_{\theta_V} J_{critic}$

### 3.3 Training Process

The training implementation maintains the distinction between actor and critic updates:

```python
# Actor update
dist = model.get_action_dist(states)
new_log_probs = dist.log_prob(actions).sum(-1)
ratio = torch.exp(new_log_probs - old_log_probs.detach())

# PPO clipped objective
actor_loss1 = -ratio * advantages
actor_loss2 = -torch.clamp(ratio, 0.8, 1.2) * advantages
actor_loss = torch.max(actor_loss1, actor_loss2).mean()
entropy_loss = -0.02 * dist.entropy().mean()

total_actor_loss = actor_loss + entropy_loss

# Critic update
values = model.get_value(states).squeeze()
critic_loss = (returns - values).pow(2).mean()
```

Separate optimizers for actor and critic components maintain independent learning rates:

```python
actor_optimizer = optim.Adam(model.actor.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)
```

This separation allows for different learning dynamics between the actor and critic networks, with each optimizing its parameters independently.

### 3.4 Exploration Strategy

The implementation employs a sophisticated exploration strategy that combines Gaussian noise with decay mechanisms. This approach ensures thorough environment exploration while gradually transitioning to exploitation:

```python
exploration_noise = 0.5  # Initial noise magnitude
noise_decay = 0.995     # Decay rate per episode
min_noise = 0.1        # Minimum noise floor

# During action selection
if episode < 100:
    action += torch.randn_like(action) * exploration_noise
action = torch.clamp(action, -1, 1)

# Noise decay after each episode
exploration_noise = max(min_noise, exploration_noise * noise_decay)
```

The exploration mechanism in our implementation operates through a sophisticated multi-layered approach designed to ensure thorough environment exploration. The foundation of our exploration strategy lies in the intrinsic stochasticity of the Gaussian policy, where the learned state-dependent standard deviation parameter naturally induces exploratory behavior throughout training. This base exploration is augmented by an additional external noise injection mechanism during the initial training phase. Specifically, during the first 100 episodes, Gaussian noise is added to the sampled actions, enabling broader state-space coverage and preventing premature convergence to suboptimal strategies. The magnitude of this supplementary exploration follows an exponential decay schedule while maintaining a minimum noise floor:

$\epsilon_t = \max(\epsilon_{min}, \epsilon_0 \cdot \alpha^t)$

Where:

- $\epsilon_0 = 0.5$ (initial noise)
- $\alpha = 0.995$ (decay rate)
- $\epsilon_{min} = 0.1$ (minimum noise)
- $t$ is the episode number

### 3.5 Advantage Estimation

The implementation utilizes Generalized Advantage Estimation (GAE) to compute reliable policy gradients. The GAE calculation incorporates both immediate rewards and value estimates:

```python
with torch.no_grad():
    next_value = 0 if done else model.get_value(torch.FloatTensor(state)).item()
    returns = []
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        next_val = next_value if t == len(rewards)-1 else values[t+1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * gae

        returns.insert(0, gae + values[t])
        advantages.insert(0, gae)
```

This formulation balances bias and variance in advantage estimates:

$A^{GAE}(\lambda) = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$

Where:

- $\gamma = 0.99$ (discount factor)
- $\lambda = 0.95$ (GAE parameter)
- $\delta_t$ is the TD-error at step t

### 3.6 Training Loop Structure

The training loop comprises four essential phases, each serving a specific purpose in the learning process. First, the trajectory collection phase gathers sequences of state-action-reward experiences using the current policy. This phase is crucial for sampling diverse environment interactions and maintaining exploration.

Following collection, the experience processing phase computes advantages and returns using GAE. This computation provides the foundation for policy improvement by estimating the quality of actions relative to the current value estimates. The GAE mechanism balances between immediate rewards and long-term value estimates, crucial for the Mountain Car problem where immediate rewards may not fully reflect action utility.

The policy update phase then executes multiple PPO optimization steps, specifically four epochs per episode. These multiple passes over the collected experience allow for thorough policy refinement while the PPO clipping mechanism ensures updates remain within trust regions. This careful balance between exploitation of current experience and conservative policy updates proves essential for stable learning.

The implementation considers the environment solved when an episode achieves a reward threshold of 96.5, indicating consistent successful goal achievement. This criterion ensures the learned policy reliably solves the Mountain Car task before training concludes.

### 3.7 Implementation Parameters and Algorithm Overview

The implementation uses the following hyperparameters:

Network Architecture:

- Hidden layer dimensions: 128 units
- Activation function: ReLU
- Weight initialization: Orthogonal with gain 1.414
- Actor output: 2 values (mean and log_std)
- Critic output: 1 value (state value)

Training Parameters:

- Maximum episodes: 1000
- Maximum steps per episode: 999
- Discount factor (γ): 0.99
- GAE parameter (λ): 0.95
- PPO clip range (ε): 0.2
- Actor learning rate: 3e-4
- Critic learning rate: 1e-3
- PPO epochs per update: 4
- Entropy coefficient: 0.02
- Gradient clip norm: 0.5

Exploration Parameters:

- Initial noise (ε₀): 0.5
- Noise decay rate (α): 0.995
- Minimum noise (ε_min): 0.1
- Policy std bounds: [0.2, 1.0]

#### Pseudocode

```
Initialize:
    ActorCritic model with separate actor and critic networks
    Metrics tracker
    Initial exploration noise ε = 0.5
    Experience buffers D = {}

for episode = 1 to MAX_EPISODES do
    Reset environment, get normalized initial state
    D.clear()  # Clear episode buffers

    # Collect trajectory
    while not done do
        # Get actions and values
        μ, σ = actor(state)
        value = critic(state)
        action = sample from N(μ, σ)

        # Add exploration noise in early training
        if episode < 100:
            action += noise_sample * ε
        action = clip(action, -1, 1)

        # Environment interaction
        next_state, reward = execute(action)

        # Shape rewards
        reward += uphill_bonus + velocity_bonus - valley_penalty

        # Store transition
        D ← (state, action, reward, value, log_prob)
        state = next_state

    # Process episode data
    advantages = compute_gae(D.rewards, D.values)
    returns = compute_returns(D.rewards, D.values)
    normalize(advantages)

    # PPO training loop
    for update = 1 to 4 do
        # Update actor
        ratio = new_policy / old_policy
        clip_ratio = clip(ratio, 0.8, 1.2)
        actor_loss = min(ratio * advantages, clip_ratio * advantages)
        update_actor(actor_loss + entropy_bonus)

        # Update critic
        value_loss = mse(returns, critic(states))
        update_critic(value_loss)

    # Decay exploration
    ε = max(0.1, ε * 0.995)

    # Log progress
    update_metrics()
    if solved:
        return model

return model
```

## 4. Results Analysis

### 4.1 Training Dynamics

The training process exhibits distinct phases characterized by the evolution of various performance metrics. Analysis of these patterns provides insight into the learning dynamics and the effectiveness of our architectural choices.

![alt text](training_metrics_episode_990.png)
_Figure 2: Visualization of training metrics showing episode rewards, lengths, success rate, and learning curves over 1000 episodes._

Episode rewards demonstrate a clear progression through three phases. The initial exploration phase (episodes 0-50) shows high variance and predominantly negative rewards as the agent learns basic environment dynamics. The rapid learning phase (episodes 50-200) displays a sharp increase in average reward, culminating in consistent goal achievement. The final optimization phase (episodes 200+) exhibits stable performance with occasional variations due to the stochastic policy.

### 4.2 Critic Learning Convergence

The critic's learning progression in our Actor-Critic architecture reveals fascinating patterns that shed light on both the learning dynamics and the Mountain Car environment's inherent challenges. The logarithmic scale visualization of the critic loss demonstrates a complex convergence pattern characterized by distinct phases and persistent oscillations:

![alt text](image-2.png)
_Figure 3: Log-scale visualization of critic loss._

The loss pattern shows an initial phase of high-magnitude errors (10² range) followed by a general downward trend, but with notable characteristics that reflect the environment's dynamics. The persistent oscillations and occasional spikes in loss, even after apparent convergence, can be attributed to several key factors.

In the Mountain Car environment **state values are highly interdependent**. When the policy improves and finds new ways to build momentum, it can suddenly make previously learned value estimates inaccurate, leading to spikes in critic loss.

The stochastic nature of our policy, combined with the exploration noise schedule, means **the agent periodically explores new state-action pairs**. These exploration phases can lead to temporary increases in critic loss as the value function adapts to newly discovered trajectories.

The sharp spikes in loss around episodes 200 and 600 likely correspond to moments when the **policy escapes local optima (like small oscillation patterns) and discovers more effective strategies**. These transitions force rapid readjustment of value estimates across many states.

This pattern of convergence with persistent fluctuations is actually desirable in our case. A completely stable critic might indicate the policy has stopped exploring and improving. The ongoing adjustments in our critic loss suggest continuous policy refinement and adaptation, essential for maintaining robustness in the Mountain Car task's complex dynamics.

### 4.3 Policy Evolution

Analysis of the actor network's output layer weights provides insight into policy development. The weight trajectories show distinct clustering patterns that correspond to learned control strategies:

![alt text](image-3.png)
_Figure 4: Evolution of actor network output layer weights showing convergence patterns._

The weight evolution exhibits three characteristic behaviors:

First, initial weight values show high volatility as the network explores the action space. Second, weights gradually organize into distinct clusters, indicating specialization in different aspects of control policy. Finally, weight values stabilize with minor adjustments, reflecting policy refinement.

The emergence of clustered weight patterns aligns with the physics of the mountain car problem. Some weights specialize in acceleration phases, while others control deceleration timing, creating a coordinated oscillatory strategy.

### 4.4 Performance Comparison

Our implementation demonstrates significant improvements over the reference approach described by [Steinbach (2019)](https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c). While both solutions eventually solve the mountain car problem, our architecture achieves several key advantages.

The most notable improvement lies in convergence speed, reaching stable performance in approximately 200 episodes compared to the reference implementation's 300+ episodes. This acceleration stems from the combination of sophisticated reward shaping and modern policy optimization techniques.

### 4.5 Local Minima Analysis

The implementation exhibits a ~90% success rate in escaping local minima, with approximately 1 in 10 training runs falling into suboptimal policies. This behavior emerges from the interplay between exploration noise and reward shaping:

$\text{Exploration Noise} = \max(\epsilon_{min}, \epsilon_0 \cdot 0.995^t)$

The noise decay schedule balances exploration and exploitation, though in some cases, early random trajectories can lead to premature convergence to oscillatory patterns that fail to reach the goal state.

## 5. Discussion and Future Work

The success of our implementation stems from several key design decisions. The separate actor and critic networks prove particularly effective for the Mountain Car environment, where policy and value estimation can independently develop specialized feature representations. This separation not only allows for focused learning but also provides stability through independent optimization.

The actor branch's Gaussian policy output, combined with tanh-bounded means and clamped standard deviations, provides controlled exploration within the environment's action bounds. Meanwhile, the critic branch's value estimation benefits from the same feature extraction pathway, enabling efficient learning of state-value relationships.

The effectiveness of this design is further enhanced by several modern techniques. The PPO clipping mechanism prevents destructive policy updates, particularly crucial for the Mountain Car's problem nature, where aggressive updates could destabilize both policy and value learning. Orthogonal initialization with carefully tuned gains ensures proper gradient flow through the ReLU activations, while separate learning rates for actor and critic components (3e-4 and 1e-3 respectively) accommodate their different learning dynamics.

### 5.2 Limitations

Despite strong overall performance, several limitations merit discussion. The 10% failure rate in escaping local minima suggests room for improvement in exploration strategies. The reward shaping mechanism, while effective, introduces additional hyperparameters that require careful tuning.

### 5.3 Future Improvements

Several promising directions for future work emerge from our analysis:

The implementation of adaptive exploration strategies could address the local minima challenge. A curriculum learning approach, gradually increasing the difficulty of initial states, might provide more robust policy learning. Integration of ensemble methods could mitigate the impact of random initialization on training outcomes.

## 6. References

1. Schulman, John & Wolski, Filip & Dhariwal, Prafulla & Radford, Alec & Klimov, Oleg. (2017). Proximal Policy Optimization Algorithms. 10.48550/arXiv.1707.06347.

2. Schulman, John & Moritz, Philipp & Levine, Sergey & Jordan, Michael & Abbeel, Pieter. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. 10.48550/arXiv.1506.02438.

3. Fujimoto, S., Van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. International Conference on Machine Learning, 1587-1596. 10.48550/arXiv.1802.09477

4. [Steinbach, A. (2019). Actor-Critic using Deep RL: Continuous Mountain Car in TensorFlow. Medium.](https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c)

5. [Gymnasium Documentation](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/).


