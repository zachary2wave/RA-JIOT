from UDN_secnario import scenario
from torch.optim import lr_scheduler
from GraphSage_He import GraphSage_net
from util import *
import time
import os
import logger
import os
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

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
loop_time = 1
load = 1
activate_UE_num = 100
Band_width = 20
batch_size = 64
# %%
if load == 0:
    ENV = scenario(area=200, step_size=1, num_block=20, AP_num=APn, low_bound=100)
    save_file(path_env, ENV)
else:
    ENV = load_file("save_env/063010")

model = torch.load("./show/generate_OUR_DATE/sp081815UE120/model.pkl")
# model.UEn = 100
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
Loss_surpvised = torch.nn.CrossEntropyLoss()
max_power = np.concatenate((10**(16/10) * np.ones((MAPn)), 1 * np.ones((APn))))

loss_re = []
steps = 0
high_rate = 100
weight = 200

for loop in range(loop_time):
    print(loop)
    pl, adj_ue, adj_ap, activate = ENV.active_UE(activate_UE_num)
    channel = power_trans(-pl)
    channel_ = standardization(channel)
    channel_ = np.concatenate((channel_, np.zeros((105, 20))),axis = 1)
    requires = ENV.need_require(activate_UE_num)
    requires = standardization(requires)
    #
    # our algorithm_1
    UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome, VAR = model.forward(channel_, requires, adj_ue, adj_ap)
    connect = []
    for i in range(activate_UE_num):
        connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
    PCoutcome = PCoutcome.squeeze()
    PCoutcome = PCoutcome * torch.from_numpy(max_power)
    rate_our, rate_ue_our = calcul_rate(connect, PCoutcome.cpu().detach().numpy(), channel)
    ENV.show_connect(connect, activate, rate_ue_our)
    rate_our, rate_ue_our = calcul_rate(connect, PCoutcome.cpu().detach().numpy(), channel)
    alg6_rate = rate_calculation(UAoutcome_de, PCoutcome, activate_UE_num, pl, Band_width)

    # # our algorithm_2
    # max_record_power = []
    # max_flag = 0
    # max_record_label = []
    # for train in range(2500):
    #     UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome = model.forward(channel_, requires, adj_ue, adj_ap)
    #     PCoutcome = PCoutcome.squeeze()
    #     PCoutcome = PCoutcome * torch.from_numpy(max_power)
    #
    #     UElabel = np.array(max_channels_connect(pl))
    #     constraints_BS, constraints_UE = part_connect_with_policy(UElabel, max_power, pl, activate_UE_num)
    #     constrain_index = UAoutcome_ori[constraints_UE,:]
    #     lsp = Loss_surpvised(UAoutcome_ori, torch.from_numpy(UElabel))
    #     rate, entropy, max_UA = loss_calculation(UAoutcome_de, PCoutcome, channel, Band_width)
    #     loss_now = weight * lsp - rate\
    #                # - 0.1 * entropy - 0.005 * max_UA
    #     optimizer.zero_grad()
    #     loss_now.backward()
    #     optimizer.step()
    #
    #     connect = []
    #     for i in range(activate_UE_num):
    #         connect.append(np.argmax(UAoutcome_de[i, :].cpu().detach().numpy()))
    #     rate_stand, rate_ue_stand = calcul_rate(connect, PCoutcome.cpu().detach().numpy(), channel)
    #
    #     print("step", train,"RATE",rate,"rate_stand", rate_stand)
    #     if rate_stand>max_flag:
    #         max_record_power = deepcopy(PCoutcome.cpu().detach().numpy())
    #         max_record_label = connect[:]


    # ENV.show_connect(max_record_label, activate, rate_ue_stand)
    # rate_stand, rate_ue_stand = calcul_rate(max_record_label, max_record_power, channel)
    # plt.figure()
    # ax = plt.gca()
    #
    # sns.distplot([r/20 for r in rate_ue_alg1],label="MARA",
    #              hist_kws=dict(cumulative=True, histtype="step"),
    #              kde_kws=dict(cumulative=True,bw=.15))
    #
    # sns.distplot([r/20 for r in rate_ue_alg2],label="MSUA",
    #              hist_kws=dict(cumulative=True, histtype="step"),
    #              kde_kws=dict(cumulative=True,bw=.15))
    #
    # sns.distplot([r/20 for r in rate_ue_alg3],label="UAMWSER",
    #              hist_kws=dict(cumulative=True, histtype="step"),
    #              kde_kws=dict(cumulative=True,bw=.15))
    #
    # sns.distplot([r/20 for r in rate_ue_alg4],label="MSUAPC",
    #              hist_kws=dict(cumulative=True, histtype="step"),
    #              kde_kws=dict(cumulative=True,bw=.15))
    #
    # sns.distplot([r/20 for r in rate_ue_alg5],label="JUAPCMWSER",
    #              hist_kws=dict(cumulative=True, histtype="step"),
    #              kde_kws=dict(cumulative=True,bw=.15))
    #
    # sns.distplot([r/20 for r in rate_ue_our],label="ours_without_retrain",
    #              hist_kws=dict(cumulative=True, histtype="step"),
    #              kde_kws=dict(cumulative=True,bw=.15))
    #
    # sns.distplot([r/20 for r in rate_ue_stand],label="ours",
    #              hist_kws=dict(cumulative=True, histtype="step"),
    #              kde_kws=dict(cumulative=True,bw=.15))
    # plt.legend()
    # plt.savefig('./test2.pdf', dpi=300, format='pdf')
    # plt.show()
    pass


