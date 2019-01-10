from manager.Trajectory import Trajectory
import os, sys

argv = sys.argv
assert len(argv) >= 2 # python trajectory_player.py *.pickle
speed = int(argv[2]) if len(argv) > 2 else None
t = Trajectory(loc=argv[1])
s = argv[1].split(os.path.sep)
front = os.path.sep.join(s[:-1])
if len(front) != 0:
    des_loc_html = os.path.sep.join((front, s[-1][:-7]+".html")) # replace .pickle with .html
    des_loc_gif = os.path.sep.join((front, s[-1][:-7]+".gif")) # replace .pickle with .html
    des_loc_mp4 = os.path.sep.join((front, s[-1][:-7]+".mp4")) # replace .pickle with .html
else:
    des_loc_html = s[-1][:-7]+".html" # replace .pickle with .html
    des_loc_gif = s[-1][:-7]+".gif" # replace .pickle with .html
    des_loc_mp4 = s[-1][:-7]+".mp4" # replace .pickle with .html
t.show(speed=speed)
c = input("Save to HTML5? y/[n] ")
if c == "y":
    t.saveAsHtml5(loc=des_loc_html)
c = input("Save to GIF? y/[n] ")
if c == "y":
    t.saveAsGif(loc=des_loc_gif)
c = input("Save to MP4? y/[n] ")
if c == "y":
    t.saveAsMp4(des_loc_mp4)