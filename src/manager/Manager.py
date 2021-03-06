from util.config import Config
config = Config()
from util.logger import Logger
logger = Logger()
from manager.Trajectory import Trajectory
from util import getTimeStr

import os
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
    
    def run(self, episode=100, train_after_every=10, save_trajectory_every=None, report_after_every=10):
        logger.info("Manager started, {} episodes to go.".format(str(episode)))
        for e in range(episode):
            result, states, actions, rewards, infos_list = self.run_episode()
            if (save_trajectory_every is not None and (e + 1) % save_trajectory_every == 0) or result == True:
                t = Trajectory(str(self.agent), str(self.env), result, states, actions, rewards, infos_list)
                loc = os.path.join(config.trajectory_save_dir, getTimeStr()+"-episode-{}.pickle".format(str(e)))
                logger.info("Saving {}trajectory {} -> {}".format("success " if result else "", str(e), str(loc)))
                t.dump(loc)
            # add histories
            self.results.append(result)
            self.rewards.append(sum(rewards))
            self.turns.append(len(actions))
            if (e + 1) % train_after_every == 0:
                self.agent.train()
            if (e + 1) % report_after_every == 0:
                logger.info("#episode: {};\tsuccess_rate: {:.2%};\taverage reward: {:.2f};\taverage turn: {:.2f};".format(str(e+1), self.success_rates()[-1], self.average_rewards()[-1], self.average_turns()[-1]))
            self.agent.reset()
            self.env.reset()
    
    def run_episode(self):
        states = [self.env.getState()] # There's one more init 'state' so it's longer than other lists
        actions = []
        rewards = []
        infos_list = []
        while True:
            action = self.agent.act(states[-1])
            actions.append(action)
            state_, reward, isFinish, infos = self.env.update(action)
            rewards.append(reward)
            infos_list.append(infos)
            self.agent.record(states[-1], action, reward, state_, isFinish)
            states.append(state_)
            if isFinish: # environment will limit the max_turn
                return self.env.success, states, actions, rewards, infos_list
        
    def success_rates(self, average_use=50): # success for 50 turn
        s = 0
        ret = []
        for i in range(len(self.results)):
            if self.results[i] == True: # success
                s += 1
            if i >= average_use:
                s -= 1 if self.results[i-average_use] else 0
                success_rate = 1.0 * s / average_use
            else:
                success_rate = 1.0 * s / (i + 1.0)
            ret.append(success_rate)
        return ret

    def average_rewards(self, average_use=50):
        s = 0
        ret = []
        for i in range(len(self.rewards)):
            s += self.rewards[i]
            if i >= average_use:
                s -= self.rewards[i-average_use]
                average_rewards = 1.0 * s / average_use
            else:
                average_rewards = 1.0 * s / (i + 1.0)
            ret.append(average_rewards)
        return ret

    def average_turns(self, average_use=50):
        s = 0
        ret = []
        for i in range(len(self.turns)):
            s += self.turns[i]
            if i >= average_use:
                s -= self.turns[i-average_use]
                average_turns = 1.0 * s / average_use
            else:
                average_turns = 1.0 * s / (i + 1.0)
            ret.append(average_turns)
        return ret

    def report(self):
        '''
        logger.info("Success rates: \n{}".format(self.success_rates()))
        logger.info("Average rewards: \n{}".format(self.average_rewards()))
        logger.info("Average turns: \n{}".format(self.average_turns()))
        '''
        logger.record("Success: \n{}".format(self.success_rates(1)))
        logger.record("Rewards: \n{}".format(self.average_rewards(1)))
        logger.record("Turns: \n{}".format(self.average_turns(1)))