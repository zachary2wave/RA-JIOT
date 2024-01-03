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
path = "./" + time_now + "UE120/"
configlist = ["stdout", "log", 'tensorboard']
logger.configure(path, configlist)


def loss_calculation(UAoutcome_de, UAoutcome_st, PCoutcome,  pl, Band_width):
    apn, uen = pl.shape
    P_bs = PCoutcome.detach()

    label = []
    for i in range(activate_UE_num):
        label.append(torch.argmax(UAoutcome_st[i, :]))
    UAnum = torch.zeros_like(PCoutcome)
    for i in connect:
        UAnum[i] += 1
    activate_AP = np.array([i for i in range(apn) if UAnum[i] > 0])
    pl_cal = torch.from_numpy(pl)
    #####  UA part

    rate_ua = torch.tensor(0)
    for ue in range(uen):
        index_vector = torch.zeros([apn])
        label_ue = int(label[ue])
        index_vector[int(label_ue)] = 1
        signal = torch.max(index_vector * P_bs * pl_cal[:, ue] * UAoutcome_st[ue, :])
        interfence = torch.sum(P_bs[activate_AP] * pl_cal[activate_AP, ue]) - signal
        sinr = signal / interfence
        connectAP_num = torch.sum(UAoutcome_de[:, int(label_ue)])
        ue_rate = Band_width / connectAP_num * torch.log2(1 + sinr)
        rate_ua = rate_ua + ue_rate
    #####  PC part
    rate_pc = torch.tensor(0)
    entropy = torch.tensor(0)
    max_UA = torch.tensor(0)
    rate_record = torch.zeros([uen])
    for ue in range(uen):
        label_ue = label[ue]
        signal = torch.sum(PCoutcome[label_ue] * pl_cal[label_ue, ue])
        interfence = torch.sum(PCoutcome[activate_AP] * pl_cal[activate_AP, ue]) - signal
        sinr = signal / interfence
        ue_rate = Band_width / UAnum[label_ue] * torch.log2(1 + sinr)
        rate_record[ue] = ue_rate
        rate_pc = rate_pc + ue_rate
        entropy = entropy - torch.sum(UAoutcome_st[ue, :] * torch.log(UAoutcome_st[ue, :] + 1e-10))
        max_UA = max_UA + torch.max(UAoutcome_st[ue, :])
    return rate_ua, rate_pc, entropy, torch.min(rate_record), max_UA

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
weight = 10
window = deque(maxlen=30)


# %%
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
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min= 1e-5)
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
        # ## 最大信噪比链接  alg0
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
        rate_ua, rate_pc, entropy, balance, max_UA = loss_calculation(UAoutcome_de,UAoutcome_st, PCoutcome, channel, Band_width)
        rate_now = rate_pc.cpu().detach().numpy()
        #
        loss_now = weight * lsp - rate_pc - rate_ua - 0.01 * entropy - 1e14 * balance + Var_ue


        optimizer.zero_grad()
        rate_pc.backward()
        for name, parms in model.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                  ' -->grad_value:', parms.grad)
        print("================================================================")
        optimizer.zero_grad()
        rate_ua.backward()
        for name, parms in model.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                  ' -->grad_value:', parms.grad)




        loss_batch = loss_batch + loss_now
        re_entropy.append(entropy.cpu().detach().numpy())
        re_max_UA.append(balance.cpu().detach().numpy())
        re_lsp.append(lsp.cpu().detach().numpy())
        if rate_now > high_rate:
            high_rate = rate_now
            torch.save(model, path + "model.pkl")
            print("the rate have reached", rate_now, "model has been saved")
            connect = []
            for i in range(activate_UE_num):
                connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
            # ENV.show_connect(connect, activate)


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
    # ENV.show_connect(UElabel00, activate, rate_ue_with_out_PC)
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

    if np.mean(record_our) < np.mean(window) and np.mean(record_our)>np.mean(record_label):
        scheduler.step()
    # if rate_with_out_PC > 0.9yang5*np.mean(record_00):
    #     weight = 0.1



