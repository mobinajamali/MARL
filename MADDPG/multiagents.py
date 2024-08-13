from agent import Agent

class MultiAgent:
    '''
    construct multi-agent domain from the agent class
    '''
    def __init__(self, actor_dims, critic_dims, n_actions, env, n_agents,
                 lr_actor=1e-4, lr_critic=1e-3, fc1_dim=64, fc2_dim=64, gamma=0.95, tau=0.01,
                 ckp_dir='tmp/', scenario='simple_speaker_listener'):
        self.agents = []
        ckp_dir += scenario
        for i in range(n_agents):
            agent = list(env.action_spaces.keys())[i]
            min_action = env.action_space(agent).low
            max_action = env.action_space(agent).high
            self.agents.append(Agent(actor_dim=actor_dims[i], critic_dim=critic_dims, n_actions=n_actions[i], n_agents=n_agents, 
                                     agent_id=i, lr_actor=lr_actor, lr_critic=lr_critic, tau=tau, fc1_dim=fc1_dim, fc2_dim=fc2_dim, ckp_dir=ckp_dir, 
                                     gamma=gamma, min_action=min_action, max_action=max_action))

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, obs, evaluate=False):
        actions = {}
        for agent_id, agent in zip(obs, self.agents):
            action = agent.choose_action(obs[agent_id], evaluate)
            actions[agent_id] = action
        return actions

    def learn(self, memory):
        for agent in self.agents:
            agent.learn(memory, self.agents)