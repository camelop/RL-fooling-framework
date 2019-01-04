from util.config import Config
config = Config()
from util.logger import Logger
logger = Logger()

import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers

class Trajectory(object):
    
    def __init__(self, agent_name=None, env_name=None, result=None, states=None, actions=None, rewards=None, infos_list=None, loc=None):
        if loc is None:
            self.data = dict(agent_name=agent_name, 
                            env_name=env_name,
                            result=result,
                            states=states,
                            actions=actions,
                            rewards=rewards,
                            infos_list=infos_list)
        else:
            self.load(loc)
        self.ani = None
    
    def dump(self, loc):
        with open(loc, 'wb') as f:
            pickle.dump(self.data, f)
    
    def load(self, loc):
        with open(loc, 'rb') as f:
            self.data = pickle.load(f)

    def createAnimation(self, fps=24.0):
        if self.data['env_name'].startswith('MnistClassifierEnv'):
            fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.6))
            confidence = [infos['confidence'] for infos in self.data['infos_list']]
            confidence = [confidence[0]] + confidence
            states = [s[1] for s in self.data['states']]
            diffs = [(256 + s - states[0])/2.0 - 1 for s in states]
            frames = list(zip(range(len(states)), diffs, states))
            frames = [frames[0]] + frames # for init_func default
            axs[0].imshow(diffs[0], cmap="PiYG", vmin=0, vmax=255)
            axs[0].axis('off')
            axs[0].set_title("Attacking region")
            axs[1].imshow(states[0], cmap="Greys", vmin=0, vmax=255)
            axs[1].axis('off')
            axs[1].set_title("Modified image")
            axs[2].plot(range(len(confidence)), confidence)
            axs[2].set(xlabel="Turn", ylabel="Confidence(%)")
            axs[2].set_title("Confidence")
            def update(frame):
                index, diff, state = frame
                title = "[ {} <-> {} ] Result: *{}*      Turn: {}".format(self.data['agent_name'],
                        self.data['env_name'],
                        "Success" if self.data['result'] else "Failed",
                        str(index))
                fig.suptitle(title)
                axs[0].clear()
                axs[0].imshow(diff, cmap="PiYG", vmin=0, vmax=255)
                axs[0].set_title("Attacking region")
                axs[1].clear()
                axs[1].imshow(state, cmap="Greys", vmin=0, vmax=255)
                axs[1].set_title("Modified image")
                axs[2].clear()
                axs[2].plot(range(len(confidence)), confidence)
                axs[2].set(xlabel="Turn", ylabel="Confidence(%)")
                axs[2].vlines([index], 0, 1, transform=axs[2].get_xaxis_transform(), colors='r')
                axs[2].set_title("Confidence {:.2%}".format(confidence[index]))
            self.ani = FuncAnimation(fig, update, frames=frames, interval=1000.0/fps, repeat=True)
        else:
            raise NotImplementedError

    def show(self):
        if self.ani is None:
            self.createAnimation()
        plt.show()

    def saveAsHtml5(self, loc):
        if self.ani is None:
            self.createAnimation()
        logger.info('Begin saving html5 to "{}"'.format(loc))
        with open(loc, 'w') as f:
            # f.write('<!DOCTYPE html> <html> <head> <meta charset="UTF-8"> <title>Test</title> </head> <body> ')
            f.write(self.ani.to_html5_video())
            # f.write('</body> </html>')
        logger.info('Saving html5 to "{}" finished.'.format(loc))
    
    def saveAsGif(self, loc):
        if self.ani is None:
            self.createAnimation()
        logger.info('Begin saving gif to "{}"'.format(loc))
        self.ani.save(loc, writer='imagemagick')
        logger.info('Saving gif to "{}" finished.'.format(loc))
    
    def saveAsMp4(self, loc):
        if self.ani is None:
            self.createAnimation()
        logger.info('Begin saving gif to "{}"'.format(loc))
        FFMpegWriter = writers['ffmpeg']
        writer = FFMpegWriter(metadata=dict(title=loc, artist='littleRound', comment="Testing {}'s performance in {}".format(self.data['agent_name'], self.data['env_name'])))
        self.ani.save(loc, writer=writer)
        logger.info('Saving gif to "{}" finished.'.format(loc))