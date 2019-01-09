import sys, os, traceback

from util.config import Config
config = Config()
from util.logger import Logger
logger = Logger()
from manager.Manager import Manager


def run_experiment(agents, envs, episode=100, train_after_every=10, save_trajectory_every=100, report_after_every=10):
    try:
        for agent in agents:
            for env in envs:
                logger.info("Begin {} vs {}".format(str(agent), str(env)))
                manager = Manager(agent, env)
                manager.run(episode, train_after_every, save_trajectory_every, report_after_every=report_after_every)
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
    from agent.mnist.RandomAgent import RandomAgent
    from env.MnistClassifierEnv import MnistClassifierEnv
    run_experiment([RandomAgent()], [MnistClassifierEnv()])
    logger.debug("mnist_random_env_random_agent ends.")

def experiment_1(args):
    '''2019-1-3: 
        To test whether we can save trajectorys.
    '''
    logger.debug("Trajectory test starts.")
    from agent.mnist.RandomAgent import RandomAgent
    from env.MnistClassifierEnv import MnistClassifierEnv
    from model.mnist.models import LogisticRegression, LeNet
    run_experiment([RandomAgent(pixel_change_max=64)], [MnistClassifierEnv()], episode=3, save_trajectory_every=1)
    logger.debug("mnist_random_env_random_agent ends.")

def experiment_2(args):
    '''2019-1-4:
        Test attacking LeNet with random agent
    '''
    logger.debug("LeNet<->RandomAgent test starts.")
    from agent.mnist.RandomAgent import RandomAgent
    from env.MnistClassifierEnv import MnistClassifierEnv
    from model.mnist.models import LeNet
    run_experiment([RandomAgent(pixel_change_max=256)], [MnistClassifierEnv(LeNet())], episode=1000, save_trajectory_every=1)
    logger.debug("LeNet<->RandomAgent ends.")

def experiment_3(args):
    '''2019-1-9:
        Test attacking LeNet with CNNDQN agent
        result: hard to train
    '''
    logger.info("LeNet<->CNNDQN test starts.")
    from agent.mnist.RandomAgent import RandomAgent
    from agent.mnist.DqnAgent import DqnAgent
    from agent.mnist.model.CNNDQN import CNNDQN
    from env.MnistClassifierEnv import MnistClassifierEnv
    from model.mnist.models import LeNet
    run_experiment([DqnAgent(CNNDQN(), action_step=128, memory_size=500, gamma=0.99)], [MnistClassifierEnv(LeNet(), max_turn=100)], 
                episode=100, 
                train_after_every=5,
                save_trajectory_every=10, 
                report_after_every=1)
    logger.info("LeNet<->CNNDQN ends.")

def experiment_4(args):
    '''2019-1-9:
        Test attacking LeNet with CNNDQN agent with specific settings
    '''
    logger.info("LeNet<->CNNDQN (specific settings) test starts.")
    from agent.mnist.RandomAgent import RandomAgent
    from agent.mnist.DqnAgent import DqnAgent
    from agent.mnist.model.CNNDQN import CNNDQN
    from env.MnistClassifierEnv import MnistClassifierEnv
    from model.mnist.models import LeNet
    max_turn = 1000
    episode = 500
    train_after_every = 20
    memory_size = 2 * train_after_every * max_turn
    run_experiment([DqnAgent(CNNDQN(learning_rate=1e-1), action_step=128, memory_size=memory_size, eps_greed=0.1, gamma=0.9)], [MnistClassifierEnv(LeNet(), max_turn=max_turn)], 
                episode=episode, 
                train_after_every=train_after_every,
                save_trajectory_every=20, 
                report_after_every=1)
    logger.info("LeNet<->CNNDQN (specific settings) ends.")

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
