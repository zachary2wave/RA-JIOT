# %%
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
time_now = str(time.strftime("%m%d%H", time.localtime()))
path_env = "save_env/" + time_now
os.makedirs(path_env, exist_ok=True)
path = "./sp" + time_now + "UE120/"
configlist = ["stdout", "log", 'tensorboard']
logger.configure(path, configlist)


def loss_calculation_ua(UAoutcome_de, PCoutcome,  pl, Band_width):
    apn, uen = pl.shape
    PCoutcome = PCoutcome.detach()
    UAnum = torch.sum(UAoutcome_de, dim=0)
    activate_AP = np.array([i for i in range(apn) if UAnum[i] > 0])
    sinr = torch.zeros([uen])
    rate = torch.tensor(0)
    entropy = torch.tensor(0)
    max_UA = torch.tensor(0)
    pl_cal = torch.from_numpy(pl)
    rate_record = torch.zeros([uen])
    for ue in range(uen):
        signal = torch.sum(UAoutcome_de[ue, activate_AP] * (PCoutcome[activate_AP] * pl_cal[activate_AP, ue]))
        interfence = torch.sum(PCoutcome[activate_AP] * pl_cal[activate_AP, ue]) - signal
        sinr[ue] = signal / interfence
        connectAP_num = torch.max(UAoutcome_de[ue, :] * UAnum)
        ue_rate = Band_width / connectAP_num * torch.log2(1 + sinr[ue])
        rate_record[ue] = ue_rate
        rate = rate + ue_rate
        entropy = entropy - torch.sum(UAoutcome_st[ue, :] * torch.log(UAoutcome_st[ue, :] + 1e-10))
        # max_UA = max_UA + torch.max(UAoutcome_st[ue, :])
    return rate, entropy, torch.min(rate_record)

def loss_calculation_pc(UAoutcome_de, PCoutcome,  pl, Band_width):
    apn, uen = pl.shape
    UAoutcome_de = UAoutcome_de.detach()

    connect = []
    for i in range(activate_UE_num):
        connect.append(torch.argmax(UAoutcome_de[i, :]))
    UAnum = torch.zeros_like(PCoutcome)
    for i in connect:
        UAnum[i] += 1
    activate_AP = np.array([i for i in range(apn) if UAnum[i] > 0])
    sinr = torch.zeros([uen])
    rate = torch.tensor(0)
    entropy = torch.tensor(0)
    pl_cal = torch.from_numpy(pl)
    rate_record = torch.zeros([uen])
    for ue in range(uen):
        signal = torch.sum(UAoutcome_de[ue, activate_AP] * (PCoutcome[activate_AP] * pl_cal[activate_AP, ue]))
        interfence = torch.sum(PCoutcome[activate_AP] * pl_cal[activate_AP, ue]) - signal
        sinr[ue] = signal / interfence
        ue_rate = Band_width / UAnum[connect[ue]] * torch.log2(1 + sinr[ue])
        rate_record[ue] = ue_rate
        rate = rate + ue_rate
        entropy = entropy - torch.sum(UAoutcome_st[ue, :] * torch.log(UAoutcome_st[ue, :] + 1e-10))
        # max_UA = max_UA + torch.max(UAoutcome_st[ue, :])
    return rate, entropy, torch.min(rate_record)
APn = 100
MAPn = 5
loop_time = 10000
train_iter = 5000000
load = 1
activate_UE_num = 120
Band_width = 20
batch_size = 64
steps = 0
high_rate = 2000
weight = 100
window = deque(maxlen=30)


# %%d
if load == 0:
    if not os.path.exists(path):
        os.makedirs(path)
    ENV = scenario(area=200, step_size=1, num_block=20, AP_num=APn, low_bound=100)
    save_file(path_env, ENV)
else:
    ENV = load_file("save_env/063010")
# %% show_scenario
# ENV.show_scenario(0, 68)
# ENV.show_channel(68)


# %%

model = GraphSage_net(APn + MAPn, activate_UE_num,
                      Band_width=Band_width,
                      hidden_dim=128, num_sample=10,
                      cuda=False, gcn=False)
# model = torch.load("080815UE120/model.pkl")
# WWW = model.weight_ue
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure()
# sns.heatmap(WWW.detach().numpy())
# plt.show()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500, eta_min= 1e-5)
Loss_surpvised = torch.nn.CrossEntropyLoss()

max_power = torch.cat((10**(16/10) * torch.ones((MAPn)), 1 * torch.ones((APn))))
loss_re = []
# pretrain

# for loop in range(1000):
#     loss_batch = 0
#     for batch in range(batch_size):
#         pl, adj_ue, adj_ap, activate = ENV.active_UE(activate_UE_num)
#         requires = ENV.need_require(activate_UE_num)
#         channel = power_trans(-pl)
#         channel_ = standardization(channel)
#         requires = standardization(requires)
#         UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome, Var_ue = model.forward(channel_, requires, adj_ue, adj_ap)
#
#         UElabel00 = max_channels_connect(pl)
#         lsp = Loss_surpvised(UAoutcome_ori, torch.tensor(UElabel00))
#         loss_batch = loss_batch+lsp
#     optimizer.zero_grad()
#     loss_batch.backward()
#     optimizer.step()
#     print("loop",loop,"loss:",loss_batch)
# connect = []
# # for i in range(activate_UE_num):
# #     connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
# # rate_with_out_PC, rate_ue_with_out_PC = calcul_rate(connect, max_power.numpy(), channel)
# # ENV.show_connect(UElabel00, activate)
# # ENV.show_connect(connect, activate, rate_ue_with_out_PC)

for loop in range(loop_time):
    loss_batch = 0
    re_entropy, re_max_UA, re_lsp = [], [], []
    record_label,record_our = [],[]
    for batch in range(batch_size):
        pl, adj_ue, adj_ap, activate = ENV.active_UE(activate_UE_num)
        requires = ENV.need_require(activate_UE_num)
        channel = power_trans(-pl)
        channel_ = standardization(channel)
        requires = standardization(requires)
        # ## 最大信xiao噪比链接  alg0
        UElabel00 = max_channels_connect(pl)
        rate_stand, _ = calcul_rate(UElabel00, max_power.numpy(), channel)
        record_label.append(rate_stand)

        UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome, Var_ue = model.forward(channel_, requires, adj_ue, adj_ap)
        PCoutcome = PCoutcome.squeeze()
        PCoutcome = PCoutcome * max_power

        connect = []
        for i in range(activate_UE_num):
            connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
        rate_our, _ = calcul_rate(connect, PCoutcome.cpu().detach().numpy(), channel)
        record_our.append(rate_our)

        lsp = Loss_surpvised(UAoutcome_ori, torch.tensor(UElabel00))
        rate_pc, entropy, balance = loss_calculation_pc(UAoutcome_de, PCoutcome, channel, Band_width)
        rate_ua, entropy_ua, balance_ua = loss_calculation_ua(UAoutcome_de, PCoutcome, channel, Band_width)
        rate_now = rate_pc.cpu().detach().numpy()
        #
        loss_now =  weight * lsp - rate_pc * 10 - rate_ua
                    # - 0.01 * entropy - 1000 * balance + Var_ue - rate_ua / 10
        loss_batch = loss_batch + loss_now

        re_entropy.append(entropy.cpu().detach().numpy())
        re_max_UA.append(balance.cpu().detach().numpy())
        re_lsp.append(lsp.cpu().detach().numpy())


    optimizer.zero_grad()
    loss_batch.backward()
    optimizer.step()

    logger.record_tabular("steps", loop)
    logger.record_tabular("ourA/entropy", np.mean(re_entropy))
    logger.record_tabular("ourA/min_rate", np.mean(re_max_UA))
    logger.record_tabular("ourA/loss_surpvised", np.mean(re_lsp))
    logger.record_tabular("ourA/loss_total", loss_batch.cpu().detach().numpy())
    logger.record_tabular("rate_compare/label", np.mean(record_label))
    logger.record_tabular("rate_compare/alour", np.mean(record_our))


    connect = []
    for i in range(activate_UE_num):
        connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
    rate_with_out_PC, rate_ue_with_out_PC = calcul_rate(connect, max_power.numpy(), channel)
    rate_stand, rate_ue_stand = calcul_rate(UElabel00, max_power.numpy(), channel)
    rate_with_out_UA, rate_ue_with_out_UA = calcul_rate(UElabel00, PCoutcome.cpu().detach().numpy(), channel)
    rate_our, rate_ue_our = calcul_rate(connect, PCoutcome.cpu().detach().numpy(), channel)


    logger.record_tabular("rate_compare/our_signal", rate_ua.cpu().detach().numpy())
    logger.record_tabular("rate_compare/with_out_PC", rate_with_out_PC)
    logger.record_tabular("rate_compare/with_out_UA", rate_with_out_UA)
    logger.record_tabular("rate_compare/max", np.max(record_our))
    logger.record_tabular("lr", optimizer.state_dict()["param_groups"][0]["lr"])
    logger.dump_tabular()

    window.append(np.mean(record_our))

    if np.mean(record_our) > high_rate+10:
        high_rate = np.mean(record_our)
        torch.save(model, path + "model.pkl")
        print("the rate have reached", np.mean(record_our), "model has been saved")
        connect = []
        for i in range(activate_UE_num):
            connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
        ENV.show_connect(connect, activate, rate_ue_our)

    if np.mean(record_our) < np.mean(window) and np.mean(record_our)>np.mean(record_label):
        scheduler.step()




    # if rate_with_out_PC > 0.9yang5*np.mean(record_00):
    #     weight = 0.1



