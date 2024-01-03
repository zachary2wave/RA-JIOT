from UDN_secnario import scenario
from torch.optim import lr_scheduler
from GraphSage_He import GraphSage_net
from util import *
import time
import os
import logger
import os
import datetime
from collections import deque


def loss_calculation(UAoutcome_de, PCoutcome, activate_UE_num, pl, Band_width):
    UA_con = UAoutcome_de.cpu().detach().numpy()
    UAnum = torch.sum(UAoutcome_de, dim=0)
    sinr = torch.zeros([activate_UE_num])
    rate = torch.tensor(0)
    entropy = torch.tensor(0)
    max_UA = torch.tensor(0)

    pl_cal = torch.from_numpy(pl)
    rate_record = []
    for ue in range(activate_UE_num):
        signal = torch.pow(10, torch.sum(UAoutcome_de[ue, :] * (PCoutcome - pl_cal[:, ue])) / 10)
        interfence = torch.sum(torch.pow(10, (PCoutcome - pl_cal[:, ue]) / 10)) - signal
        sinr[ue] = signal / interfence
        connectAP_num = torch.max(UAoutcome_de[ue, :] * UAnum)
        ue_rate = Band_width / connectAP_num * torch.log2(1 + sinr[ue])
        rate_record.append(ue_rate.cpu().detach().numpy())
        rate = rate + ue_rate
        entropy = entropy - torch.sum(UAoutcome_st[ue, :] * torch.log(UAoutcome_st[ue, :] + 1e-10))
        max_UA = max_UA + torch.max(UAoutcome_st[ue, :])
    return rate, entropy, max_UA


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
time_now = str(time.strftime("%m%d%H", time.localtime()))
path_env = "save_env/" + time_now
os.makedirs(path_env, exist_ok=True)
path = "test/" + time_now + "/"

APn = 100
MAPn = 5
loop_time = 200
load = 1
activate_UE_num = 120
Band_width = 20
batch_size = 64
# %%
if load == 0:
    if not os.path.exists(path):
        os.makedirs(path)
    ENV = scenario(area=200, step_size=1, num_block=20, AP_num=APn, low_bound=100)
    save_file(path_env, ENV)
else:
    ENV = load_file("save_env/063010")


model = torch.load("./show/generate_OUR_DATE/sp081815UE120/model.pkl")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
Loss_surpvised = torch.nn.CrossEntropyLoss()

max_power = torch.cat((50 * torch.ones((MAPn)), 20 * torch.ones((APn))))
loss_re = []
steps = 0
high_rate = 100
weight = 1
window = deque( maxlen = 20)

record_0, record_1, record_2, record_3, record_4, record_5, record_6 = [], [], [], [], [], [], []
re_entropy, re_max_UA, re_lsp = [], [], []
rtime_0, rtime_1, rtime_2, rtime_3, rtime_4, rtime_5, rtime_6 = [], [], [], [], [], [], []


for loop in range(loop_time):
    print(loop)
    pl, adj_ue, adj_ap, activate = ENV.active_UE(activate_UE_num)
    channel = power_trans(-pl)
    SINR, rate_all = calculatingRateSINR(max_power.numpy(), channel, 0)
    pl_ = standardization(pl)
    requires = ENV.need_require(activate_UE_num)
    requires = standardization(requires)

    ## 最大信噪比链接  alg0
    start0 = datetime.datetime.now()
    UElabel0  = max_SNR_without_PC(pl, np.ones(APn + MAPn) * max_power.numpy())
    alg0_rate, _ = calcul_rate(UElabel0, max_power.numpy(), pl)

    end0 = datetime.datetime.now()
    time0 = (end0 - start0).seconds+(end0 - start0).microseconds*1e-6
    rtime_0 .append(time0)

    record_0.append(alg0_rate)
    ## 最大信干噪比链接  alg1
    start1 = datetime.datetime.now()
    UElabel1 = max_SINR_without_PC(pl, np.ones(APn + MAPn) * max_power.numpy())
    alg1_rate, _ = calcul_rate(UElabel1, max_power.numpy(),pl)
    end1 = datetime.datetime.now()
    time1 = (end1 - start1).seconds+(end1 - start1).microseconds*1e-6
    rtime_1.append(time1)
    record_1.append(alg1_rate)


    ## 最大效用链接 alg2
    start2 = datetime.datetime.now()
    UElabel = max_utility_without_PC(rate_all)
    alg2_rate, _ = calcul_rate(UElabel, max_power.numpy(),pl)
    end2 = datetime.datetime.now()
    record_2.append(alg2_rate)
    time2 = (end2 - start2).seconds + (end2 - start2).microseconds*1e-6
    rtime_2.append(time2)


    ## 最大效用链接 加功率控制 alg3
    start3 = datetime.datetime.now()
    UElabel, Power = max_utility_with_PC(channel, max_power.numpy(), 0)
    alg3_rate, _ = calcul_rate(UElabel, Power, pl)
    end3 = datetime.datetime.now()

    time3 = (end3 - start3).seconds+(end3 - start3).microseconds*1e-6
    rtime_3.append(time3)

    record_3.append(alg3_rate)
    ## 最大能效链接 alg4
    start4 = datetime.datetime.now()
    UElabel_label = maxEffectiveRate_without_PC(rate_all)
    alg4_rate, _ = calcul_rate(UElabel_label, max_power.numpy(), pl)
    end4 = datetime.datetime.now()

    time4 = (end4 - start4).seconds + (end4 - start4).microseconds * 1e-6
    rtime_4.append(time4)
    record_4.append(alg4_rate)
    ## 最大能效链接 加功率控制 alg5
    start5 = datetime.datetime.now()
    UElabel, currentPower = maxEffectiveRate_with_PC(channel, max_power.numpy(), 0)
    alg5_rate, _ = calcul_rate(UElabel, currentPower, pl)
    end5 = datetime.datetime.now()

    time5 = (end5 - start5).seconds + (end5 - start5).microseconds * 1e-6
    rtime_5.append(time5)
    record_5.append(alg5_rate)

    ## our algorithm
    start6 = datetime.datetime.now()
    while True:
        UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome = model.forward(pl_, requires, adj_ue, adj_ap)
        PCoutcome = PCoutcome.squeeze()
        PCoutcome = PCoutcome * max_power

        lsp = Loss_surpvised(UAoutcome_ori, torch.tensor(UElabel_label))
        rate, entropy, max_UA = loss_calculation(UAoutcome_de, PCoutcome, activate_UE_num, pl, Band_width)
        rate_now = rate.cpu().detach().numpy()
        loss_now = weight * lsp - rate - 0.005 * entropy - 0.005 * max_UA
        optimizer.zero_grad()
        loss_now.backward()
        optimizer.step()
        window.append(rate_now)
        if abs(rate_now-np.mean(window))<10:
            break
    connect = []
    for i in range(activate_UE_num):
        connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
    PCoutcome = PCoutcome.squeeze()
    PCoutcome = PCoutcome * max_power
    rate_our, rate_ue_our = calcul_rate(connect, PCoutcome.cpu().detach().numpy(), pl)
    # ENV.show_connect(connect, activate)
    alg6_rate = rate_calculation(UAoutcome_de, PCoutcome, activate_UE_num, pl, Band_width)
    # print(rate_our,alg6_rate)
    end6 = datetime.datetime.now()
    time6 = (end6 - start6).seconds + (end6 - start6).microseconds * 1e-6
    rtime_6.append(time6)
    record_6.append(alg6_rate)
    pass
    print("*****************************************")
    print("loop: ",loop)
    print("0: ",time0, "1: ",time1, "2: ", time2, "3: ",time3, "4: ",time4, "5: ",time5, "6: ",time6)
    print("0: ", alg0_rate, "1: ", alg1_rate, "2: ", alg2_rate, "3: ", alg3_rate, "4: ", alg4_rate, "5: ", alg5_rate, "6: ", alg6_rate)




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
        "OURS":record_6}
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