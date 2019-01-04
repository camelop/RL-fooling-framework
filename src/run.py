import sys, os, traceback

from util.config import Config
config = Config()
from util.logger import Logger
logger = Logger()
from manager.Manager import Manager


def run_experiment(agents, envs, episode=100, train_after_every=10, save_trajectory_every=None):
    try:
        for agent in agents:
            for env in envs:
                logger.info("")
                manager = Manager(agent, env)
                manager.run(episode, train_after_every, save_trajectory_every)
                manager.report()
    except:
        logger.error(traceback.format_exc())

#----------------------- experiment list -----------------------

def experiment_command_test(args):
    logger.debug("command_test start.")
    print("Args: " + str(args))
    logger.debug("command_test end.")

def experiment_0(args):
    '''2018-12-29: 
        To test whether we implement 'Manager' correctly.
    '''
    logger.debug("mnist_random_env_random_agent starts.")
    from agent.cifar10.RandomAgent import RandomAgent
    from env.MnistClassifierEnv import MnistClassifierEnv
    run_experiment([RandomAgent()], [MnistClassifierEnv()])
    logger.debug("mnist_random_env_random_agent ends.")

def experiment_1(args):
    '''2019-1-3: 
        To test whether we can save trajectorys.
    '''
    logger.debug("Trajectory test starts.")
    from agent.cifar10.RandomAgent import RandomAgent
    from env.MnistClassifierEnv import MnistClassifierEnv
    from model.mnist.models import LogisticRegression, LeNet
    run_experiment([RandomAgent(pixel_change_max=64)], [MnistClassifierEnv(LeNet())], episode=3, save_trajectory_every=1)
    logger.debug("mnist_random_env_random_agent ends.")

#----------------------- experiment list -----------------------

# start
logger.debug("New experiment started.")
logger.debug("Args: "+str(sys.argv))

# run experiment
argv = sys.argv # ['program_name', #experiment, ...(other args)]
try:
    f = globals()['experiment_'+argv[1]]
    f(argv[2:])
except:
    logger.error(traceback.format_exc())

logger.debug("Done.")
