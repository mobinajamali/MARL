import numpy as np

class Replay:
    '''
    responsible to store a global critic memory transition and local actor
    memory transition for each actor (numpy arrays)
    '''
    def __init__(self, mem_size, actor_dims, critic_dims, n_actions, n_agents, batch_size):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_mem = np.zeros((self.mem_size, self.critic_dims))
        self.new_state_mem = np.zeros((self.mem_size, self.critic_dims))
        self.reward_mem = np.zeros((self.mem_size, self.n_agents))
        self.terminal_mem = np.zeros((self.mem_size, self.n_agents), dtype=bool) # terminal mask

        self.initialize_actor_mem()

    def initialize_actor_mem(self):
        self.actor_state_mem = []
        self.actor_new_state_mem = []
        self.actor_action_mem = []

        for i in range(self.n_agents):
            self.actor_state_mem.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_mem.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_mem.append(np.zeros((self.mem_size, self.n_actions[i])))

    def store_transition(self, obs, state, action, reward, obs_, state_, done):
        index = self.mem_cntr % self.mem_size # position of the first available memory
        self.state_mem[index] = state # concat observation of all the agents
        self.new_state_mem[index] = state_
        self.reward_mem[index] = reward
        self.terminal_mem[index] = done

        for i in range(self.n_agents):
            self.actor_state_mem[i][index] = obs[i]
            self.actor_new_state_mem[i][index] = obs_[i]
            self.actor_action_mem[i][index] = action[i]

        self.mem_cntr += 1

    def sample_buffer(self):
        mem = min(self.mem_cntr, self.mem_size) #highest occupied memory  
        batch = np.random.choice(mem, self.batch_size, replace=False)

        states = self.state_mem[batch]
        states_ = self.new_state_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.terminal_mem[batch]
        
        actor_state_mem = []
        actor_new_state_mem = []
        actor_action_mem = []
        for i in range(self.n_agents):
            actor_state_mem.append(self.actor_state_mem[i][batch])
            actor_new_state_mem.append(self.actor_new_state_mem[i][batch])
            actor_action_mem.append(self.actor_action_mem[i][batch])

        return actor_state_mem, states, actor_action_mem, rewards, actor_new_state_mem, states_, dones
    
    def ready(self):
        return self.mem_cntr > self.batch_size
        