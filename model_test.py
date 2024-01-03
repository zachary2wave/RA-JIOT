from UDN_secnario import scenario
from torch.optim import lr_scheduler
from GraphSage_He import GraphSage_net
from util import *
import time
import os
import logger
import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
time_now = str(time.strftime("%m%d%H", time.localtime()))
path_env = "save_env/" + time_now
os.makedirs(path_env, exist_ok=True)
path = "test/" + time_now + "/"


APn = 100
MAPn = 5
loop_time = 100
load = 1
activate_UE_num = 120
Band_width = 20
batch_size = 64

if load == 0:
    if not os.path.exists(path):
        os.makedirs(path)
    ENV = scenario(area=200, step_size=1, num_block=20, AP_num=APn, low_bound=100)
    save_file(path_env, ENV)
else:
    ENV = load_file("save_env/063010")



model = torch.load("mimi_batch_st_update070919/model.pkl")

max_power = torch.cat((50 * torch.ones((MAPn)), 20 * torch.ones((APn))))
loss_re = []
steps = 0
high_rate = 100
weight = 50


for loop in range(loop_time):
    pl, adj_ue, adj_ap, activate = ENV.active_UE(activate_UE_num)
    channel = power_trans(-pl)
    SINR, rate_all = calculatingRateSINR(max_power.numpy(), channel, 0)
    pl_ = standardization(pl)
    requires = ENV.need_require(activate_UE_num)
    requires = standardization(requires)
    UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome = model.forward(pl_, requires, adj_ue, adj_ap)
    connect = []
    for i in range(activate_UE_num):
        connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
    rate_our, rate_ue_our = calcul_rate(activate_UE_num, connect,
                                        PCoutcome.cpu().detach().numpy(), pl)
    ENV.show_connect(connect, activate)
    pass