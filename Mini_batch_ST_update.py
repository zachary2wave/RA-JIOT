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
path = "./mimi_batch_st_update" + time_now + "/"
configlist = ["stdout", "log", 'tensorboard']
logger.configure(path, configlist)


def loss_calculation(UAoutcome_st,UAoutcome_de, PCoutcome, pl, Band_width):
    apn,uen = pl.shape
    UAnum = torch.sum(UAoutcome_de, dim=0)

    rate = torch.tensor(0)
    entropy = torch.tensor(0)
    max_UA = torch.tensor(0)

    pl_cal = torch.from_numpy(pl)
    rate_record = []
    for ue in range(uen):
        sinr_e = torch.tensor(0)
        interfence = torch.sum(torch.pow(10, (PCoutcome - pl_cal[:, ue]) / 10))
        for ap in range(apn):
            signal = torch.pow(10, (PCoutcome[ap] - pl_cal[ap, ue]) / 10)
            sinr_e = sinr_e + UAoutcome_st[ue, ap]*signal / (interfence- signal)
        connectAP_num = torch.max(UAoutcome_de[ue, :] * UAnum)
        ue_rate = Band_width / connectAP_num * torch.log2(1 + sinr_e)
        rate_record.append(ue_rate.cpu().detach().numpy())
        rate = rate + ue_rate
        entropy = entropy - torch.sum(UAoutcome_st[ue, :] * torch.log(UAoutcome_st[ue, :] + 1e-10))
        max_UA = max_UA + torch.max(UAoutcome_st[ue, :])
    return rate, entropy, max_UA


APn = 100
MAPn = 5
loop_time = 1000000
train_iter = 5000000
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
# %% show_scenario
# ENV.show_scenario(0, 68)
# ENV.show_channel(68)


# %%

model = GraphSage_net(APn + MAPn, activate_UE_num,
                      Band_width=Band_width,
                      hidden_dim=128, num_sample=10,
                      cuda=False, gcn=False)

# model = torch.load("060914/model.pkl")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
Loss_surpvised = torch.nn.CrossEntropyLoss()

max_power = torch.cat((50 * torch.ones((MAPn)), 20 * torch.ones((APn))))
loss_re = []
steps = 0
high_rate = 100
weight = 100
window = deque( maxlen = 20)
for loop in range(loop_time):
    record_0,record_1,record_2,record_3,record_4,record_5,record_6 =[],[],[],[],[],[],[]
    re_entropy, re_max_UA, re_lsp = [],[],[]
    rtime_0, rtime_1, rtime_2, rtime_3, rtime_4, rtime_5, rtime_6 = [], [], [], [], [], [], []
    loss_batch = 0
    for batch in range(batch_size):
        pl, adj_ue, adj_ap, activate = ENV.active_UE(activate_UE_num)
        requires = ENV.need_require(activate_UE_num)
        channel = power_trans(-pl)
        SINR, rate_all = calculatingRateSINR(max_power.numpy(), channel, 0)
        pl_ = standardization(pl)
        requires = standardization(requires)


        # ## 最大信噪比链接  alg0
        # start0 = datetime.datetime.now()
        # UElabel0  = max_SNR_without_PC(pl, np.ones(APn + MAPn) * max_power.numpy())
        # alg0_rate, _ = calcul_rate(activate_UE_num, UElabel0, max_power.numpy(),
        #                                         pl)
        # end0 = datetime.datetime.now()
        # rtime_0 .append((end0 - start0).seconds+(end0 - start0).microseconds*1e-6)
        #
        # ## 最大信干噪比链接  alg1
        # start1 = datetime.datetime.now()
        # UElabel1 = max_SINR_without_PC(pl, np.ones(APn + MAPn) * max_power.numpy())
        # alg1_rate, _ = calcul_rate(activate_UE_num, UElabel1, max_power.numpy(),
        #                                         pl)
        # end1 = datetime.datetime.now()
        # rtime_1.append((end1 - start1).seconds+(end1 - start1).microseconds*1e-6)
        #
        # ## 最大效用链接 alg2
        # start2 = datetime.datetime.now()
        # UElabel = max_utility_without_PC(rate_all)
        # alg2_rate, _ = calcul_rate(activate_UE_num, UElabel, max_power.numpy(),pl)
        # end2 = datetime.datetime.now()
        # rtime_2.append((end2 - start2).seconds + (end2 - start2).microseconds*1e-6)
        # ## 最大效用链接 加功率控制 alg3
        # start3 = datetime.datetime.now()
        # UElabel, Power = max_utility_with_PC(channel, max_power.numpy(), 0)
        # alg3_rate, _ = calcul_rate(activate_UE_num, UElabel, Power, pl)
        # end3 = datetime.datetime.now()
        # rtime_3.append((end3 - start3).seconds+(end3 - start3).microseconds*1e-6)

        # ## 最大能效链接 alg4
        start4 = datetime.datetime.now()
        UElabel = maxEffectiveRate_without_PC(rate_all)
        alg4_rate, _ = calcul_rate(activate_UE_num, UElabel, max_power.numpy(),pl)
        end4 = datetime.datetime.now()
        rtime_4.append((end4 - start4).seconds + (end4 - start4).microseconds * 1e-6)
        # ## 最大能效链接 加功率控制 alg5
        # start5 = datetime.datetime.now()
        # UElabel, currentPower = maxEffectiveRate_with_PC(channel, max_power.numpy(), 0)
        # alg5_rate, _ = calcul_rate(activate_UE_num, UElabel, currentPower, pl)
        # end5 = datetime.datetime.now()
        # rtime_5.append((end5 - start5).seconds + (end5 - start5).microseconds * 1e-6)

        UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome = model.forward(pl_, requires, adj_ue, adj_ap)
        PCoutcome = PCoutcome.squeeze()
        PCoutcome = PCoutcome * max_power

        # constraints_BS, constraints_UE = part_connect_with_SNR(pl, max_power.numpy(), activate_UE_num/2)
        # constraints_BS, constraints_UE = part_connect_with_policy(UElabel, max_power.numpy(), pl, activate_UE_num)
        # constrain_index = UAoutcome_ori[constraints_UE,:]
        # lsp = Loss_surpvised(constrain_index, torch.from_numpy(constraints_BS))
        lsp = Loss_surpvised(UAoutcome_ori, torch.tensor(UElabel))
        rate, entropy, max_UA = loss_calculation(UAoutcome_st,UAoutcome_de, PCoutcome, pl, Band_width)
        rate_now = rate.cpu().detach().numpy()
        # record_1.append(alg1_rate)
        # record_2.append(alg2_rate)
        # record_3.append(alg3_rate)
        record_4.append(alg4_rate)
        # record_5.append(alg5_rate)
        record_6.append(rate_now)
        loss_now = weight * lsp - rate + 0.01 * entropy
                   # - 0.005 * max_UA
        loss_batch = loss_batch + loss_now
        re_entropy.append(entropy.cpu().detach().numpy())
        re_max_UA.append(max_UA.cpu().detach().numpy())
        re_lsp.append(lsp.cpu().detach().numpy())

            # ENV.show_connect(connect, activate)


    optimizer.zero_grad()
    loss_batch.backward()
    optimizer.step()

    logger.record_tabular("steps", loop)
    logger.record_tabular("ourA/entropy", np.mean(re_entropy))
    logger.record_tabular("ourA/max_UA", np.mean(re_max_UA))
    logger.record_tabular("ourA/loss_surpvised", np.mean(re_lsp))
    logger.record_tabular("ourA/loss_total", loss_batch.cpu().detach().numpy())
    # logger.record_tabular("otherA/al1", np.mean(record_1))
    # logger.record_tabular("otherA/al2", np.mean(record_2))
    # logger.record_tabular("otherA/al3", np.mean(record_3))
    logger.record_tabular("otherA/al4", np.mean(record_4))
    # logger.record_tabular("otherA/al5", np.mean(record_5))
    logger.record_tabular("otherA/alour", np.mean(record_6))

    connect = []
    for i in range(activate_UE_num):
        connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
    # ENV.show_connect(connect, activate)

    rate_stand, rate_ue_stand = calcul_rate(activate_UE_num, UElabel, max_power.numpy(), pl)
    rate_with_out_PC, rate_ue_with_out_PC = calcul_rate(activate_UE_num, connect, max_power.numpy(), pl)
    rate_with_out_UA, rate_ue_with_out_UA = calcul_rate(activate_UE_num, UElabel, PCoutcome.cpu().detach().numpy(), pl)
    rate_our, rate_ue_our = calcul_rate(activate_UE_num, connect,
                                        PCoutcome.cpu().detach().numpy(), pl)
    logger.record_tabular("rate/al4", alg4_rate)
    logger.record_tabular("rate/our", rate_our)
    logger.record_tabular("rate/with_out_PC", rate_with_out_PC)
    logger.record_tabular("rate/with_out_UA", rate_with_out_UA)
    if rate_our > high_rate:
        high_rate = rate_now
        torch.save(model, path + "/model.pkl")
        print("the rate have reached", rate_now, "model has been saved")
        connect = []
        for i in range(activate_UE_num):
            connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
    logger.dump_tabular()

    window.append(np.mean(record_6))

    if np.mean(record_6) < np.mean(window) and np.mean(record_6)>1800:
        scheduler.step()



        # if rate_now > rate_reach:
        #
        #     print("model has been saved for the rate catching", rate_now)
        #     connect = []
        #     for i in range(activate_UE_num):
        #         connect.append(np.argmax(UA_con[i]))
        #     # ENV.show_connect(connect, activate)
        #     outcome_rate, outcome_rate_ue = calcul_rate(activate_UE_num, connect,
        #                                                 PCoutcome.cpu().detach().numpy(), pl)
        #     print("the stand rate is ", stand_rate, "now we achieve:", outcome_rate)
        #     print("the learning rete is ", optimizer.state_dict()['param_groups'][0]['lr'])
        #     rate_reach = rate_now
        # if rate_now > high_rate:
        #     high_rate = rate_now
        #     torch.save(model, "./" + time_now + "/model.pkl")
        #     print("model has been saved")
    # scheduler.step()



