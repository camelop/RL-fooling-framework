from manager.Trajectory import Trajectory
import os, sys

argv = sys.argv
assert len(argv) == 2 # python trajectory_player.py *.pickle

t = Trajectory(loc=argv[1])
s = argv[1].split(os.path.sep)
front = os.path.sep.join(s[:-1])
des_loc = os.path.sep.join((front, s[-1][:-7]+".html")) # replace .pickle with .html
# t.show()
t.saveAsHtml5(loc=des_loc)