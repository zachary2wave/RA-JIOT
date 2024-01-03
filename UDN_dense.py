#%%
from UDN_secnario import scenario
import torch
from Dense import DenseNet
from util import *
import time
import os
import logger
os.environ['KMP_DUPLICATE_LIB_OK'] =  'True'
time_now = str(time.strftime("%m%d%H", time.localtime()))
path = "save_env/"+time_now

path = "./"+time_now+"/"
configlist = ["stdout", "log", 'tensorboard']
logger.configure(path, configlist)

APn = 100
loop_time = 10000
train_iter = 5000
load = 1
activate_UE_num = 100
Band_width = 20
#%%
if load == 0:
    if not os.path.exists(path):
        os.makedirs(path)
    ENV = scenario(area=200, step_size=1, num_block=20, AP_num=APn, low_bound=100)
    save_file(path, ENV)
else:
    ENV = load_file("save_env/060212")
#%% show_scenario
# ENV.show_scenario(0, 68)
# ENV.show_channel(68)



#%%

model = DenseNet(APn, activate_UE_num)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=100, verbose=True
                                                       , threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08)
Loss_surpvised = torch.nn.CrossEntropyLoss()
rate_reach = 1000
loss_re = []
steps = 0
for loop in range(loop_time):
    pl, adj_ue, adj_ap, activate = ENV.active_UE(activate_UE_num)
    requires = ENV.need_require(activate_UE_num)
    UElabel = ENV.label_gene(activate)
    pl_ = standardization(pl)
    requires = standardization(requires)

    for time in range(train_iter):
        steps += 1
        f = torch.from_numpy(np.concatenate((pl, requires[np.newaxis,:]), axis=0).astype(np.float32))
        UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome = model.forward(f)
        PCoutcome = PCoutcome.squeeze()
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
            signal = torch.pow(10, torch.sum(UAoutcome_de[ue,:] * (PCoutcome.squeeze()*20 - pl_cal[:, ue]))/10)
            interfence = torch.sum(torch.pow(10, (PCoutcome.squeeze()*20 - pl_cal[:, ue])/10)) - signal
            sinr[ue] = signal / interfence
            connectAP_num = torch.max(UAoutcome_de[ue,:] * UAnum)
            ue_rate = Band_width / connectAP_num * torch.log2(1 + sinr[ue])
            rate_record.append(ue_rate.cpu().detach().numpy())
            rate = rate + ue_rate
            entropy = entropy - torch.sum(UAoutcome_st[ue,:]*torch.log(UAoutcome_st[ue,:]+1e-10))
            max_UA = max_UA + torch.max(UAoutcome_st[ue,:])

        rate_now = rate.cpu().detach().numpy()
        loss = 10 * lsp - 0.01 * rate - 0.0001*entropy - 0.0001*max_UA

        loss2 = - 0.001*entropy - 0.001*max_UA
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(rate)
        logger.record_tabular("steps", steps)
        logger.record_tabular("iter", time)
        logger.record_tabular("loop", loop)
        logger.record_tabular("rate", rate.cpu().detach().numpy())
        logger.record_tabular("entropy", entropy.cpu().detach().numpy())
        logger.record_tabular("max_UA", max_UA.cpu().detach().numpy())
        logger.record_tabular("loss_surpvised", lsp.cpu().detach().numpy())
        logger.record_tabular("loss_total",loss.cpu().detach().numpy() )
        logger.dump_tabular()

        if rate_now>rate_reach+100:
            torch.save(model,"./"+time_now+"/model.pkl")
            print("model has been saved for the rate catching", rate_now)
            stand_rate, stand_rate_ue = calcul_rate(activate_UE_num, UElabel, np.ones_like(PCoutcome.cpu().detach().numpy()), pl)
            print("and the given", rate.cpu().detach().numpy(), "the calcul_rate", rate_reach)
            # ENV.show_connect(UElabel, activate)
            connect = []
            for i in range(activate_UE_num):
                connect.append(np.argmax(UA_con[i]))
            ENV.show_connect(connect, activate)
            rate_reach = rate_now[:]            # test_outcome(PCoutcome, pl_cal, 0, UAoutcome_de)

