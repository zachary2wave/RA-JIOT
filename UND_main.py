#%%
from UDN_secnario import scenario
from torch.optim import lr_scheduler
from GraphSage_He import GraphSage_net
from util import *
import time
import os
import logger

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
#%% show_scenario
# ENV.show_scenario(0, 68)
# ENV.show_channel(68)



#%%

model = GraphSage_net(APn+MAPn, activate_UE_num,
                 Band_width=Band_width,
                 hidden_dim=128, num_sample=10,
                 cuda=False, gcn=False)

# model = torch.load("060914/model.pkl")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9979)
Loss_surpvised = torch.nn.CrossEntropyLoss()

max_power = torch.cat((50 * torch.ones((MAPn)),20 * torch.ones((APn))))
loss_re = []
steps = 0
high_rate = 2000
weight = 1
for loop in range(loop_time):
    pl, adj_ue, adj_ap, activate = ENV.active_UE(activate_UE_num)
    requires = ENV.need_require(activate_UE_num)
    UElabel = max_SNR_without_PC(pl, np.ones(APn+MAPn)* max_power.numpy())

    pl_ = standardization(pl)
    requires = standardization(requires)
    stand_rate, stand_rate_ue = calcul_rate(activate_UE_num, UElabel, np.ones(APn+MAPn),
                                            pl)
    rate_reach = stand_rate

    for time in range(train_iter):
        steps += 1
        UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome = model.forward(pl_, requires, adj_ue, adj_ap)
        PCoutcome = PCoutcome.squeeze()
        PCoutcome = PCoutcome * max_power
        lsp = Loss_surpvised(UAoutcome_ori, torch.tensor(UElabel))

        UA_con = UAoutcome_de.cpu().detach().numpy()

        UAnum = torch.sum(UAoutcome_de, dim=0)
        sinr = torch.zeros([activate_UE_num])
        rate = torch.tensor(0)
        entropy = torch.tensor(0)
        max_UA = torch.tensor(0)

        pl_cal = torch.from_numpy(pl)
        rate_record = []
        for ue in range(activate_UE_num):
            signal = torch.pow(10, torch.sum(UAoutcome_de[ue,:] * (PCoutcome - pl_cal[:, ue]))/10)
            interfence = torch.sum(torch.pow(10, (PCoutcome - pl_cal[:, ue])/10)) - signal
            sinr[ue] = signal / interfence
            connectAP_num = torch.max(UAoutcome_de[ue,:] * UAnum)
            ue_rate = Band_width / connectAP_num * torch.log2(1 + sinr[ue])
            rate_record.append(ue_rate.cpu().detach().numpy())
            rate = rate + ue_rate
            entropy = entropy - torch.sum(UAoutcome_st[ue,:]*torch.log(UAoutcome_st[ue,:]+1e-10))
            max_UA = max_UA + torch.max(UAoutcome_st[ue,:])

        rate_now = rate.cpu().detach().numpy()
        loss = weight*lsp - rate - 0.005*entropy - 0.005*max_UA


        #loss2 =
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loop % 50 == 0:
            logger.record_tabular("steps", steps)
            logger.record_tabular("iter", time)
            logger.record_tabular("loop", loop)
            logger.record_tabular("rate", rate.cpu().detach().numpy())
            logger.record_tabular("entropy", entropy.cpu().detach().numpy())
            logger.record_tabular("max_UA", max_UA.cpu().detach().numpy())
            logger.record_tabular("loss_surpvised", lsp.cpu().detach().numpy())
            logger.record_tabular("loss_total",loss.cpu().detach().numpy())
            logger.record_tabular("learning rate", optimizer.state_dict()["param_groups"][0]["lr"])
            # logger.record_tabular("weight", weight)
            connect = []
            for i in range(activate_UE_num):
                connect.append(np.argmax(UA_con[i]))
            rate_propose, _ = calcul_rate(activate_UE_num, connect,
                                                    PCoutcome.cpu().detach().numpy(), pl)
            rate_without_PC, _ = calcul_rate(activate_UE_num, connect,
                                                        np.ones(APn+MAPn), pl)
            rate_without_UA, _ = calcul_rate(activate_UE_num, UElabel,
                                                        PCoutcome.cpu().detach().numpy(), pl)

            logger.record_tabular("stand rate" , stand_rate )
            logger.record_tabular("withour PC", rate_without_PC)
            logger.record_tabular("without UA", rate_without_UA)
            logger.record_tabular("now we achieve:", rate_propose)
            logger.dump_tabular()
            if rate_without_PC>rate_propose-50:
                weight = 0.01

        # if rate_now > rate_reach:
        #
        #     print("model has been saved for the rate catching", rate_now)
        #     connect = []
        #     for i in range(activate_UE_num):
        #         connect.append(np.argmax(UA_con[i]))
        #     # ENV.show_connect(connect, activate)
        #     rate_propose, _ = calcul_rate(activate_UE_num, connect,
        #                                             PCoutcome.cpu().detach().numpy(), pl)
        #     rate_without_PC, _ = calcul_rate(activate_UE_num, connect,
        #                                                 np.ones(APn+MAPn), pl)
        #     rate_without_UA, _ = calcul_rate(activate_UE_num, UElabel,
        #                                                 PCoutcome.cpu().detach().numpy(), pl)
        #     if stand_rate+10>=rate_without_PC>=stand_rate-10 and weight>5:
        #         weight -= 0.1
        #     print("the stand rate is ", stand_rate,
        #           "withour PC", rate_without_PC, "without UA", rate_without_UA,
        #           "now we achieve:", rate_propose)
        #     rate_reach = rate_now

        if rate_now > high_rate:
            high_rate = rate_now
            torch.save(model, "./" + time_now + "/model.pkl")
            print("model has been saved")

    # scheduler.step()
