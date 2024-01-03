
from UDN_secnario import scenario
import time
import os
from util import *
time_now = str(time.strftime("%m%d%H", time.localtime()))


APn = 100
MAPn = 5

ENV = scenario(area=200, step_size=1, num_block=20, AP_num=80, low_bound=100)
path = "save_env/apn80/"
if not os.path.exists(path):
    os.makedirs(path)
save_file(path, ENV)

ENV = scenario(area=200, step_size=1, num_block=20, AP_num=90, low_bound=100)
path = "save_env/apn90/"
if not os.path.exists(path):
    os.makedirs(path)
save_file(path, ENV)


ENV = scenario(area=200, step_size=1, num_block=20, AP_num=110, low_bound=100)
path = "save_env/apn110/"
if not os.path.exists(path):
    os.makedirs(path)
save_file(path, ENV)


ENV = scenario(area=200, step_size=1, num_block=20, AP_num=120, low_bound=100)
path = "save_env/apn120/"
if not os.path.exists(path):
    os.makedirs(path)
save_file(path, ENV)

