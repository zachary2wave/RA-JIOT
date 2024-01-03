from UDN_secnario import scenario
from torch.optim import lr_scheduler
from GraphSage_He import GraphSage_net
from util import *
import time
from copy import deepcopy
import logger
import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
time_now = str(time.strftime("%m%d%H", time.localtime()))
path_env = "save_env/" + time_now
os.makedirs(path_env, exist_ok=True)
path = "test/" + time_now + "/"
if not os.path.exists(path):
    os.makedirs(path)
APn = 100
MAPn = 5
loop_time = 50
load = 1
activate_UE_num = 120
Band_width = 20
batch_size = 64
# %%
if load == 0:

    ENV = scenario(area=200, step_size=1, num_block=20, AP_num=APn, low_bound=100)
    save_file(path_env, ENV)
else:
    ENV = load_file("save_env/063010")


# model = torch.load("080711UE120/model.pkl")
model = torch.load("./show/generate_OUR_DATE/sp081815UE120/model.pkl")

max_power = np.concatenate((16*np.ones((MAPn)), 0 * np.ones((APn))))
max_power_in_w = np.power(10, max_power/10)
loss_re = []
steps = 0
high_rate = 100
weight = 50

record_0, record_1, record_2, record_3, record_4, record_5, record_6 = [], [], [], [], [], [], []
record_ue_0, record_ue_1, record_ue_2, record_ue_3, record_ue_4, record_ue_5, record_ue_6 = [], [], [], [], [], [], []
re_entropy, re_max_UA, re_lsp = [], [], []
rtime_0, rtime_1, rtime_2, rtime_3, rtime_4, rtime_5, rtime_6 = [], [], [], [], [], [], []


for loop in range(loop_time):
    print(loop)
    pl, adj_ue, adj_ap, activate = ENV.active_UE(activate_UE_num)
    pl_ = standardization(pl)
    channel = power_trans(-pl)
    channel_ = standardization(channel)
    requires = ENV.need_require(activate_UE_num)
    requires = standardization(requires)
    SINR, rate_all = calculatingRateSINR(deepcopy(max_power_in_w), channel, 0)
    # 最大信干噪比链接  alg1
    start1 = datetime.datetime.now()
    UElabel1 = max_SINR_without_PC(pl, max_power_in_w)
    alg1_rate, alg1_ue_rate = calcul_rate(UElabel1, max_power_in_w, channel)
    end1 = datetime.datetime.now()
    time1 = (end1 - start1).seconds+(end1 - start1).microseconds*1e-6
    rtime_1.append(time1)
    record_1.append(alg1_rate)
    record_ue_1.append(alg1_ue_rate)

    ## 最大效用链接 alg2
    start2 = datetime.datetime.now()
    UElabel2 = max_utility_without_PC(rate_all)
    alg2_rate, alg2_ue_rate = calcul_rate(UElabel2, max_power_in_w,channel)
    end2 = datetime.datetime.now()
    record_2.append(alg2_rate)
    time2 = (end2 - start2).seconds + (end2 - start2).microseconds*1e-6
    rtime_2.append(time2)
    record_ue_2.append(alg2_ue_rate)

    # 最大效用链接 加功率控制 alg3
    start3 = datetime.datetime.now()
    UElabel3, Power3 = max_utility_with_PC(channel, deepcopy(max_power_in_w), 0)
    alg3_rate, alg3_ue_rate = calcul_rate(UElabel3, Power3, channel)
    end3 = datetime.datetime.now()
    time3 = (end3 - start3).seconds+(end3 - start3).microseconds*1e-6
    rtime_3.append(time3)
    record_3.append(alg3_rate)
    record_ue_3.append(alg3_ue_rate)

    # ## 最大能效链接 alg4
    start4 = datetime.datetime.now()
    UElabel4 = maxEffectiveRate_without_PC(rate_all)
    alg4_rate, alg4_ue_rate = calcul_rate(UElabel4, max_power_in_w, channel)
    end4 = datetime.datetime.now()
    time4 = (end4 - start4).seconds + (end4 - start4).microseconds * 1e-6
    rtime_4.append(time4)
    record_4.append(alg4_rate)
    record_ue_4.append(alg4_ue_rate)

    ## 最大能效链接 加功率控制 alg5
    start5 = datetime.datetime.now()
    UElabel5, currentPower5 = maxEffectiveRate_with_PC(channel, deepcopy(max_power_in_w), 0)
    alg5_rate, alg5_ue_rate = calcul_rate(UElabel5, currentPower5, channel)
    end5 = datetime.datetime.now()
    time5 = (end5 - start5).seconds + (end5 - start5).microseconds * 1e-6
    rtime_5.append(time5)
    record_5.append(alg5_rate)
    record_ue_5.append(alg5_ue_rate)

    ## our algorithm
    # start6 = datetime.datetime.now()
    # UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome, Var = model.forward(pl_, requires, adj_ue, adj_ap)
    # connect = []
    # for i in range(activate_UE_num):
    #     connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
    # PCoutcome = PCoutcome.squeeze()
    # PCoutcome = PCoutcome * max_power
    # rate_our, rate_ue_our = calcul_rate(connect, PCoutcome.cpu().detach().numpy(), pl)
    # alg6_rate = rate_calculation(UAoutcome_de, PCoutcome, activate_UE_num, pl, Band_width)
    # # print(rate_our,alg6_rate)
    # end6 = datetime.datetime.now()
    # time6 = (end6 - start6).seconds + (end6 - start6).microseconds * 1e-6
    # rtime_6.append(time6)
    # record_6.append(alg6_rate)

    print("*****************************************")
    print("loop: ",loop)
    # print( "1: ",time1, "2: ", time2)
    # print( "1: ", alg1_rate, "2: ", alg2_rate)
    # print("1: ",time1, "2: ", time2, "3: ",time3, "4: ",time4)
    # print("1: ", alg1_rate, "2: ", alg2_rate, "3: ", alg3_rate, "4: ", alg4_rate)
    print("1: ",time1, "2: ", time2, "3: ",time3, "4: ",time4, "5: ",time5)
    print("1: ", alg1_rate, "2: ", alg2_rate, "3: ", alg3_rate, "4: ", alg4_rate, "5: ", alg5_rate)
    pass


data = {"time_MSNRA":rtime_0,
        "time_MARA":rtime_1,
        "time_MSUA":rtime_2,
        "time_MSUAPC":rtime_3,
        "time_UAMWSER":rtime_4,
        "time_JUAPCMWSER":rtime_5,
        "time_OURS":rtime_6,
        "MSNRA":record_0,
        "MARA":record_1,
        "MSUA":record_2,
        "MSUAPC":record_3,
        "UAMWSER":record_4,
        "JUAPCMWSER":record_5,
        "OURS":record_6,
        "ue_MSNRA":record_ue_0,
        "ue_MARA":record_ue_1,
        "ue_MSUA":record_ue_2,
        "ue_MSUAPC":record_ue_3,
        "ue_UAMWSER":record_ue_4,
        "ue_JUAPCMWSER":record_ue_5,
        "ue_OURS":record_ue_6,
        }
from scipy.io import savemat
savemat(path+"data.mat",data)


alg = ["MSNRA"]*loop_time+["MARA"]*loop_time+["MSUA"]*loop_time\
      +["MSUAPC"]*loop_time+["UAMWSER"]*loop_time+["JUAPCMWSER"]*loop_time+["OURs"]*loop_time
time_alg = record_0 + record_1 + record_2 + record_3 + record_4+ record_5 + record_6
time = {"alg":alg, "time":time_alg}

rate_alg = rtime_0+rtime_1+rtime_2+rtime_3+rtime_4+rtime_5+rtime_6
rate = {"alg":alg, "rate":rate_alg}

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(1)
sns.set(style="whitegrid")
sns.boxplot(x="algorithm", y="time", data=time)

plt.figure(2)
sns.boxplot(x="algorithm", y="rate", data=rate)
plt.show()