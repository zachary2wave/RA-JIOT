#%%
from UDN_secnario import scenario
from torch.optim import lr_scheduler
from GraphSage_He import GraphSage_net
from util import *
import time
import os
import logger
import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
time_now = str(time.strftime("%m%d%H", time.localtime()))
path_env = "save_env/"+time_now
os.makedirs(path_env, exist_ok=True)
path = "./"+time_now+"/"
configlist = ["stdout", "log", 'tensorboard']
logger.configure(path, configlist)

APn = 100
MAPn = 5
loop_time = 1000000
train_iter = 1
load = 1
activate_UE_num = 120
Band_width = 20
#%%
if load == 0:
    if not os.path.exists(path):
        os.makedirs(path)
    ENV = scenario(area=200, step_size=1, num_block=20, AP_num=APn, low_bound=100)
    save_file(path_env, ENV)
else:
    ENV = load_file("save_env/060911")

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




#%%

model = GraphSage_net(APn+MAPn, activate_UE_num,
                 Band_width=Band_width,
                 hidden_dim=128, num_sample=10,
                 cuda=False, gcn=False)

# model = torch.load("060914/model.pkl")
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
Loss_surpvised = torch.nn.CrossEntropyLoss()

max_power = torch.cat((50 * torch.ones((MAPn)),20 * torch.ones((APn))))
loss_re = []
steps = 0
high_rate = 2000
for loop in range(loop_time):

    pl, adj_ue, adj_ap, activate = ENV.active_UE(activate_UE_num)
    requires = ENV.need_require(activate_UE_num)
    channel = power_trans(-pl)
    SINR, rate_all = calculatingRateSINR(max_power.numpy(), channel, 0)
    pl_ = standardization(pl)

    UElabel = maxEffectiveRate_without_PC(rate_all)
    alg4_rate, _ = calcul_rate(activate_UE_num, UElabel, max_power.numpy(), pl)

    for time in range(train_iter):
        steps += 1
        UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome = model.forward(pl_, requires, adj_ue, adj_ap)
        PCoutcome = PCoutcome.squeeze()
        PCoutcome = PCoutcome * max_power
        lsp_ori = Loss_surpvised(UAoutcome_ori, torch.tensor(UElabel))
        constraints_BS, constraints_UE = part_connect_with_policy(UElabel, max_power.numpy(), pl, activate_UE_num/3*2)
        constrain_index = UAoutcome_ori[constraints_UE,:]
        lsp = Loss_surpvised(constrain_index, torch.from_numpy(constraints_BS))
        rate, entropy, max_UA = loss_calculation(UAoutcome_de, PCoutcome, activate_UE_num, pl, Band_width)
        rate_now = rate.cpu().detach().numpy()
        loss = 10 * lsp - 0.01 * rate - 0.001*entropy - 0.001*max_UA

        # loss2 =
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if time % 50 == 0:
            logger.record_tabular("steps", steps)
            logger.record_tabular("iter", time)
            logger.record_tabular("loop", loop)
            logger.record_tabular("entropy", entropy.cpu().detach().numpy())
            logger.record_tabular("max_UA", max_UA.cpu().detach().numpy())
            logger.record_tabular("loss_surpvised", lsp.cpu().detach().numpy())
            logger.record_tabular("loss_total",loss.cpu().detach().numpy())

            connect = []
            for i in range(activate_UE_num):
                connect.append(np.argmax(UAoutcome_de[i,:].cpu().detach().numpy()))
            # ENV.show_connect(connect, activate)

            rate_stand, rate_ue_stand = calcul_rate(activate_UE_num, UElabel, max_power.numpy(), pl)
            rate_with_out_PC, rate_ue_with_out_PC = calcul_rate(activate_UE_num, connect, max_power.numpy(), pl)
            rate_with_out_UA, rate_ue_with_out_UA = calcul_rate(activate_UE_num, UElabel, PCoutcome.cpu().detach().numpy(), pl)
            rate_our, rate_ue_our = calcul_rate(activate_UE_num, connect,
                                                                PCoutcome.cpu().detach().numpy(), pl)
            logger.record_tabular("rate/al4", alg4_rate)
            logger.record_tabular("rate/our", rate.cpu().detach().numpy())
            logger.record_tabular("rate/with_out_PC", rate_with_out_PC)
            logger.record_tabular("rate/with_out_UA", rate_with_out_UA)
            logger.dump_tabular()
            pass

    #     if rate_now > rate_reach:
    #
    #         print("model has been saved for the rate catching", rate_now)
    #         connect = []
    #         for i in range(activate_UE_num):
    #             connect.append(np.argmax(UA_con[i]))
    #         # ENV.show_connect(connect, activate)
    #         outcome_rate, outcome_rate_ue = calcul_rate(activate_UE_num, connect,
    #                                                 PCoutcome.cpu().detach().numpy(), pl)
    #         print("the stand rate is ", stand_rate, "now we achieve:", outcome_rate)
    #         print("the learning rete is ", optimizer.state_dict()['param_groups'][0]['lr'])
    #         rate_reach = rate_now
    #     if rate_now > high_rate:
    #         high_rate = rate_now
    #         torch.save(model, "./" + time_now + "/model.pkl")
    #         print("model has been saved")
    # scheduler.step()

