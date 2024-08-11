import numpy as np
import torch as T
import torch.nn.functional as F
from networks import CriticNetwork, ActorNetwork
#from replay_buffer import Replay

class Agent:
    def __init__(self, n_actions, min_action, max_action, actor_dim, critic_dim, agent_id,
                 n_agents, lr_critic=1e-3, lr_actor=1e-4, fc1_dim=64, fc2_dim=64, ckp_dir='tmp/', gamma=0.95, tau=0.01):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.min_action = min_action
        self.max_action = max_action
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau
        #self.mem_cntr = 0

        self.critic = CriticNetwork(lr_critic, critic_dim, fc1_dim, fc2_dim, ckp_dir, name='_critic_')
        self.target_critic = CriticNetwork(lr_critic, critic_dim, fc1_dim, fc2_dim, ckp_dir, name='_target_critic_')
        self.actor = ActorNetwork(lr_actor, actor_dim, fc1_dim, fc2_dim, n_actions, ckp_dir, name='_actor_')
        self.target_actor = ActorNetwork(lr_actor, actor_dim, fc1_dim, fc2_dim, n_actions, ckp_dir, name='_target_actor_')
        #self.replay = Replay(mem_size, actor_dim, critic_dim, n_agents, batch_size)

        self.network_update(self.actor, self.target_actor, tau=1)
        self.network_update(self.critic, self.target_critic, tau=1)


    def network_update(self, src, dest, tau=None):
        # soft update of network parameters
        tau = tau or self.tau
        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)


    def choose_action(self, observation, evaluate=False):
        # add random noise for exploration
        state = T.tensor(observation[np.newaxis, :], dtype=T.float, device=self.actor.device)
        actions = self.actor.forward(state)
        noise = T.randn(size=(self.n_actions,)).to(self.actor.device)
        noise *= T.tensor(1 - int(evaluate)) # evaluation don't require any noise (eval flag)
        # clamp the action to be within the bounds of the action space
        action = T.clamp(actions + noise,
                          T.tensor(self.min_action, device=self.actor.device),
                          T.tensor(self.max_action, device=self.actor.device))
        return action.data.cpu().numpy()[0]

    def learn(self, memory, agent_list):
        # check if replay buffer has enough experiences
        if not memory.ready():
            return 
        
        obs, state, action, reward, obs_, state_, done= memory.sample_buffer()
        states = T.tensor(np.array(state), dtype=T.float, device=self.actor.device)
        states_ = T.tensor(np.array(state_), dtype=T.float, device=self.actor.device)
        rewards = T.tensor(np.array(reward), dtype=T.float, device=self.actor.device)
        dones = T.tensor(np.array(done), device=self.actor.device)

        obss = [T.tensor(obs[idx], device=self.actor.device, dtype=T.float) for idx in range(len(agent_list))]
        obss_ = [T.tensor(obs_[idx], device=self.actor.device, dtype=T.float) for idx in range(len(agent_list))]
        actions = [T.tensor(action[idx], device=self.actor.device, dtype=T.float)for idx in range(len(agent_list))]

        with T.no_grad():
            # calculate target critic value using target networks with no gradients
            # action chosen by the actor target network for new state from the buffer
            # and evaluate actions with target critic
            new_actions = T.cat([agent.target_actor(obss_[i])
                                 for i, agent in enumerate(agent_list)],
                                dim=1)
            critic_value_ = self.target_critic.forward(
                                states_, new_actions).squeeze()
            critic_value_[dones[:, self.agent_id]] = 0.0
            target = rewards[:, self.agent_id] + self.gamma * critic_value_

        # actions sampled from the replay buffer
        old_actions = T.cat([actions[i] for i in range(self.n_agents)], dim=1)
        critic_value = self.critic.forward(states, old_actions).squeeze()
        critic_loss = F.mse_loss(target, critic_value) # update

        # update critic network
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0) # gradient clip to prevent explod
        self.critic.optimizer.step()

        # calculate actor loss and update actor network
        actions[self.agent_id] = self.actor.forward(obs[self.agent_id])
        actions = T.cat([actions[i] for i in range(self.n_agents)], dim=1)
        actor_loss = -self.critic.forward(states, actions).mean() # update actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor.optimizer.step()

        # update target networks
        self.network_update(self.actor, self.target_actor, tau=1)
        self.network_update(self.critic, self.target_critic, tau=1)
    

    def save_models(self):
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()

    def load_models(self):
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()




