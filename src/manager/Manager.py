from util.config import Config
config = Config()
from util.logger import Logger
logger = Logger()

from pprint import pformat

class Manager(object):

    def reset(self):
        self.agent.reset()
        self.env.reset()
        self.rewards = []
        self.results = []
        self.turns = []
        pass

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.reset()
        pass
    
    def run(self, episode=100, train_after_every=10):
        for e in range(episode):
            result, states, actions, rewards = self.run_episode()
            # add histories
            self.results.append(result)
            self.rewards.append(sum(rewards))
            self.turns.append(len(actions))
            if (e + 1) % train_after_every == 0:
                self.agent.train()
            self.agent.reset()
            self.env.reset()
    
    def run_episode(self):
        states = [self.env.getState()]
        actions = []
        rewards = []
        while True:
            action = self.agent.act(states[-1])
            actions.append(action)
            state_, reward, isFinish = self.env.update(action)
            rewards.append(reward)
            self.agent.record(states[-1], action, reward, state_)
            states.append(state_)
            if isFinish:
                return self.env.success, states, actions, rewards
        
    def success_rates(self):
        s = 0
        ret = []
        for i in range(len(self.results)):
            if self.results[i] == True: # success
                s += 1
            success_rate = 1.0 * s / (i + 1.0)
            ret.append(success_rate)
        return ret

    def average_rewards(self):
        s = 0
        ret = []
        for i in range(len(self.rewards)):
            s += self.rewards[i]
            average_rewards = 1.0 * s / (i + 1.0)
            ret.append(average_rewards)
        return ret

    def average_turns(self):
        s = 0
        ret = []
        for i in range(len(self.turns)):
            s += self.turns[i]
            average_turns = 1.0 * s / (i + 1.0)
            ret.append(average_turns)
        return ret

    def report(self):
        logger.info(f"Success rates: \n{self.success_rates()}")
        logger.info(f"Average rewards: \n{self.average_rewards()}")
        logger.info(f"Average turns: \n{self.average_turns()}")
        # TODO plot graphs