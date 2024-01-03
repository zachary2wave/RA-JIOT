import pickle
import numpy as np
import torch
from copy import deepcopy
def save_file(path, ENV):
    with open(path + "/env.pkl", 'wb') as output_hal:
        str = pickle.dumps(ENV)
        output_hal.write(str)
        output_hal.close()

def load_file(path):
    with open(path + "/env.pkl", 'rb') as file:
        ENV = pickle.loads(file.read())
    return ENV

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def calcul_rate(label, P, pl):
    apn, uen = pl.shape
    connectUE_num = np.zeros(apn)
    for i in label:
        connectUE_num[i] += 1
    activate_AP = [i for i in range(apn) if connectUE_num[i]>0]
    rate = 0
    rate_record = []
    for ue in range(uen):
        label_ = label[ue]
        signal = P[label_] * pl[label_, ue]
        interfence = np.sum(P[activate_AP] * pl[activate_AP, ue]) - signal
        sinr = signal / (interfence+4e-14)
        ue_rate = 20 / connectUE_num[label_] * np.log2(1 + sinr)
        rate = rate + ue_rate
        rate_record.append(ue_rate)
    return rate, rate_record

def test_outcome(PCoutcome, pl_cal, ue, UAoutcome_de):
    UAoutcome_de[ue, :] * (PCoutcome.squeeze() * 20 - pl_cal[:, ue])
    APselected = torch.argmax(UAoutcome_de[ue, :])
    UAnum = torch.sum(UAoutcome_de, dim=0)
    print("所链接BS的概率是", torch.max(UAoutcome_de[ue, :]))
    print("所连接的BS是", APselected)
    print("所连接的BS有多少个UE", UAnum[APselected])
    print("实际计算的 BS所链接UE数目", torch.max(UAoutcome_de[ue, :] * UAnum))
    receiver = (PCoutcome.squeeze() * 20 - pl_cal[:, ue])

    print("接受功率", receiver[APselected])
    print("实际计算的功率", (UAoutcome_de[ue, :] * (PCoutcome.squeeze() * 20 - pl_cal[:, ue]))[APselected])

    signal = torch.pow(10, torch.sum(UAoutcome_de[ue, :] * (PCoutcome.squeeze() * 20 - pl_cal[:, ue])) / 10)
    interfence = torch.sum(torch.pow(10, (PCoutcome.squeeze() * 20 - pl_cal[:, ue]) / 10)) - signal
    print("干扰", interfence)
    sinr = signal / interfence
    print("信噪比",sinr)

    connectAP_num = torch.max(UAoutcome_de[ue, :] * UAnum)
    ue_rate = 20 / connectAP_num * torch.log2(1 + sinr)
    print("速率：", ue_rate)

def DB_trans(inputs):
    outputs = 10*np.log10(inputs)
    return outputs

def power_trans(inputs):
    outputs = np.power(10,inputs/10)
    return outputs



def calculatingRateSINR(transmitPower,channel,noise):
    noise = 1e-17
    BSNum, userNum = channel.shape
    rate = np.zeros((BSNum, userNum))
    SINR = np.zeros((BSNum, userNum))
    interference = np.zeros((userNum, 1))
    for k in range(userNum):
        for n in range(BSNum):
            interference[k] = interference[k] + transmitPower[n] * channel[n, k]

    for n in range(BSNum):
        for k in range(userNum):
            receivedPower = transmitPower[n] * channel[n, k]
            SINR[n, k] = receivedPower / (interference[k] - receivedPower + noise)+1e-16
            rate[n, k] = np.log2(1 + SINR[n, k])+1e-16
    return SINR, rate



def calculatingSimpleSINR(y,transmitPower,channel,noise):
    noise = 1e-16
    BSNum, userNum = channel.shape
    SINR = np.zeros((BSNum, userNum))
    interference = np.zeros((userNum, 1))
    for k in range(userNum):
        for n in range(BSNum):
            interference[k] = interference[k] + transmitPower[n] * channel[n, k]

    for n in range(BSNum):
        for k in range(userNum):
            receivedPower = transmitPower[n] * channel[n, k]
            SINR[n, k] = y[n,k] / (interference[k] - receivedPower + noise)
    return SINR

def calculatingSimpleSINR_2(x,transmitPower,channel,noise):
    noise = 1e-16
    BSNum, userNum = channel.shape
    SINR = np.zeros((BSNum, userNum))
    interference = np.zeros((userNum, 1))
    for k in range(userNum):
        for n in range(BSNum):
            interference[k] = interference[k] +  transmitPower[n] * channel[n, k]

    for n in range(BSNum):
        for k in range(userNum):
            receivedPower = transmitPower[n] * channel[n, k]
            SINR[n, k] = (x[n,k]*np.log(2)) / (interference[k] - receivedPower + noise)
    return SINR

def part_connect_with_SNR(pl, P, num):
    ap_num,ue_num = pl.shape
    uelabel = []
    sinr_label = []
    for ue in range(ue_num):
        snr = []
        for ap in range(ap_num):
            signal = np.power(10, (P[ap] - pl[ap, ue]) / 10)
            snr.append(signal)
        sinr_label.append(np.max(snr))
        uelabel.append(np.argmax(snr))
    index = np.argsort(-np.array(sinr_label))
    constraints_UE = index[:int(num)]
    constraints_BS = np.array(uelabel)[list(constraints_UE)]
    return constraints_BS, constraints_UE

def part_connect_with_policy(UElabel, P, pl, num):
    ap_num, ue_num = pl.shape
    rate, rate_record = calcul_rate( UElabel, P, pl)
    index = np.argsort(-np.array(rate_record))
    constraints_UE = index[:int(num)]
    constraints_BS = np.array(UElabel)[list(constraints_UE)]

    return constraints_BS, constraints_UE

def loss_calculation(UAoutcome_de, UAoutcome_st, PCoutcome, activate_UE_num, pl, Band_width):
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

def rate_calculation(UAoutcome_de, PCoutcome, activate_UE_num, pl, Band_width):

    UAnum = torch.sum(UAoutcome_de, dim=0)
    sinr = torch.zeros([activate_UE_num])
    rate = torch.tensor(0)

    pl_cal = torch.from_numpy(pl)
    for ue in range(activate_UE_num):
        connect = torch.argmax(UAoutcome_de[ue, :])
        signal = torch.pow(10,  (PCoutcome[connect] - pl_cal[connect, ue]) / 10)
        interfence = torch.sum(torch.pow(10, (PCoutcome - pl_cal[:, ue]) / 10)) - signal
        sinr[ue] = signal / interfence
        connectAP_num = torch.max(UAoutcome_de[ue, :] * UAnum)
        ue_rate = Band_width / connectAP_num * torch.log2(1 + sinr[ue])
        rate = rate + ue_rate
    return rate.cpu().detach().numpy()

def max_channels_connect(pl):
    ap_num, ue_num = pl.shape
    UElabel = []
    for ue in range(ue_num):
        UElabel.append(np.argmin(pl[:,ue]))
    return UElabel

def max_SNR_without_PC(pl, P):
    ap_num,ue_num = pl.shape
    uelabel = []
    for ue in range(ue_num):
        snr = []
        for ap in range(ap_num):
            signal = np.power(10, (P[ap] - pl[ap, ue]) / 10)
            snr.append(signal)
        uelabel.append(np.argmax(snr))
    return uelabel


def max_SINR_without_PC(pl, P):
    ap_num,ue_num = pl.shape
    uelabel = []
    for ue in range(ue_num):
        interfence_total = np.sum(np.power(10, (P - pl[:, ue]) / 10))
        sinr = []
        for ap in range(ap_num):
            signal = np.power(10, (P[ap] - pl[ap, ue]) / 10)
            interfence = interfence_total - signal
            sinr.append(signal / interfence)
        uelabel.append(np.argmax(sinr))
    return uelabel


def max_utility_without_PC(rate):
    BSNum, userNum = rate.shape
    y = np.zeros((BSNum, 1))
    mu = np.log(userNum) * np.ones((BSNum, 1))
    iterationNum = 150
    maxUtility = np.zeros((iterationNum, 1))
    c = np.log(rate)
    step = 1e-2
    iter = 0
    while iter < iterationNum:
        x = np.zeros((BSNum, userNum))
        for j in range(userNum):
            maxU = -np.inf
            for i in range(BSNum):
                utility = c[i, j] - mu[i]
                if maxU < utility:
                    maxU = utility
                    ii = i
            x[ii, j] = 1
        for i in range(BSNum):
            y[i] = np.exp(mu[i] - 1)
        for i in range(BSNum):
            mu[i] = mu[i] - step * (y[i] - sum(x[i, :]))
        maxUtility[iter] = np.sum(np.sum(np.sum(x*c))) - np.sum(np.sum(y * np.log(y)))
        iter += 1
    uelabel = []
    for ue in range(userNum):
        uelabel.append(np.argmax(x[:,ue]))
    return uelabel


def max_utility_with_PC(channel, maxPower, noise):
    BSNum, userNum = channel.shape
    currentPower = deepcopy(maxPower)
    y = np.zeros((BSNum, 1))
    iterationNum1 = 20
    iterationNum2 = 150
    iterationNum3 = 15
    object1 = np.zeros((iterationNum1, 1))
    object2 = np.zeros((iterationNum2, 1))
    object3 = np.zeros((iterationNum3, 1))
    rate, SINR = calculatingRateSINR(maxPower, channel, noise);
    c = np.log(rate)
    step = 2e-2
    iter1 = 0
    while iter1 < iterationNum1:
        iter2 = 0
        mu = np.log(userNum) * np.ones((BSNum, 1))
        while iter2 < iterationNum2:
            x = np.zeros((BSNum, userNum))
            for k in range( userNum):
                maxU = -np.inf
                for n in range( BSNum):
                    utility = c[n, k] - mu[n]
                    if maxU < utility:
                        maxU = utility
                        ii = n
                x[ii, k] = 1
            for n in range( BSNum):
                y[n] = np.exp(mu[n] - 1)
            for n in range( BSNum):
                mu[n] = mu[n] - step * (y[n] - np.sum(x[n,:]));

            for n in range( BSNum):
                for k in range( userNum):
                    if x[n, k] != 0:
                        object2[iter2] = object2[iter2] + x[n, k] * c[n, k]
                if y[n] != 0:
                    object2[iter2] = object2[iter2] - y[n] * np.log(y[n])
            iter2 = iter2 + 1

        iter3 = 0
        currentPower = deepcopy(maxPower)
        simpleSINR = calculatingSimpleSINR_2(x, currentPower, channel, noise)
        while iter3 < iterationNum3:
            for n in range( BSNum):
                S1 = 0
                for m in range( BSNum):
                    if m != n:
                        for k in range( userNum):
                            S1 = S1 + channel[n, k] * simpleSINR[m, k]
                S2 = 0
                for k in range( userNum):
                    S2 = S2 + x[n, k]
                currentPower[n] = min(S2 / S1, maxPower[n])
            [rate, SINR] = calculatingRateSINR(currentPower, channel, noise)
            for n in range(BSNum):
                for k in range( userNum):
                    if x[n, k] != 0:
                        object3[iter3] = object3[iter3] + x[n, k] * np.log(SINR[n, k]);

            simpleSINR = calculatingSimpleSINR_2(x, currentPower, channel, noise)
            c = np.log(rate + 1e-30)
            iter3 = iter3 + 1

        for n in range( BSNum):
            for k in range( userNum):
                if x[n, k] !=0:
                    object1[iter1] = object1[iter1] + x[n, k] * c[n, k];
            if y[n] != 0:
                object1[iter1] = object1[iter1] - y[n] * np.log(y[n]);
        iter1 = iter1 + 1
    uelabel = []
    for ue in range(userNum):
        uelabel.append(np.argmax(x[:,ue]))
    return uelabel, currentPower

def maxEffectiveRate_without_PC(rate):
    BSNum, userNum = rate.shape
    x = np.zeros((BSNum, userNum))
    y = np.zeros((BSNum, 1))
    mu = np.zeros((BSNum, userNum))
    belta = np.zeros((BSNum, userNum))
    iterationNum2 = 150
    object = np.zeros((iterationNum2, 1))
    step = 0.5
    epsilon = 1e-3

    for k in range(userNum):
        maxRate = -np.inf
        for n in range(BSNum):
            if maxRate < rate[n, k]:
                maxRate = rate[n, k]
            id = n
        x[id, k] = 1
    for n in range(BSNum):
        y[n] = np.sum(x[n,:])

    for n in range(BSNum):
        for k in range(userNum):
            mu[n, k] = x[n, k] / (1 + y[n])
            belta[n, k] = rate[n, k] / (1 + y[n])
    iter2 = 0
    while iter2 < iterationNum2:
        x = np.zeros((BSNum, userNum))
        for k in range(userNum):
            maxUtility = -np.inf
            for n in range(BSNum):
                utility = belta[n, k] - np.sum(mu[n, :] * belta[n, :])
                if maxUtility < utility:
                    maxUtility = utility
                    id = n
            x[id, k] = 1
        for n in range(BSNum):
            y[n] = np.sum(x[n, :])
        sx1 = np.zeros((BSNum, userNum))
        sx2 = np.zeros((BSNum, userNum))

        for n in range(BSNum):
            for k in range(userNum):
                sx1[n, k] = x[n, k] - mu[n, k] * (1 + y[n])
                sx2[n, k] = rate[n, k] - belta[n, k] * (1 + y[n])
        if sx1.all() == 0:
            if sx2.all() == 0:
                break

        mu1 = mu
        belta1 = belta
        phi1 = np.zeros((BSNum, userNum))
        varphi1 = np.zeros((BSNum, userNum))
        phi2 = np.zeros((BSNum, userNum))
        varphi2 = np.zeros((BSNum, userNum))
        for n in range(BSNum):
            for k in range(userNum):
                phi1[n, k] = mu1[n, k] * (1 + y[n]) - x[n, k]
                varphi1[n, k] = belta[n, k] * (1 + y[n]) - rate[n, k]
        S = np.zeros((50, 1))
        for t in range(50):
            for n in range(BSNum):
                for k in range(userNum):
                    mu1[n, k] = mu1[n, k] - step ** t * phi1[n, k] / (1 + y[n])
                    belta1[n, k] = belta1[n, k] - step ** t * varphi1[n, k] / (1 + y[n])
                    phi2[n, k] = mu1[n, k] * (1 + y[n]) - x[n, k]
                    varphi2[n, k] = belta1[n, k] * (1 + y[n]) - rate[n, k]
            if np.sum(np.sum(phi2 ** 2)) + np.sum(np.sum(varphi2 ** 2)) <= (1 - epsilon * step ** t) ** 2 * (
                    np.sum(np.sum(phi1 ** 2)) + np.sum(np.sum(varphi1 ** 2))):
                break
            S[t] = np.sum(np.sum(phi2 ** 2)) + np.sum(np.sum(varphi2 ** 2));

        for n in range( BSNum):
            for k in range( userNum):
                mu[n, k] = mu[n, k] - step ** t * phi1[n, k] / (1 + y[n])
                belta[n, k] = belta[n, k] - step ** t * varphi1[n, k] / (1 + y[n])
        mu = mu / np.sum(np.sum(mu))
        object[iter2] = np.sum(np.sum(x * belta))
        iter2 = iter2 + 1
    uelabel = []
    for ue in range(userNum):
        uelabel.append(np.argmax(x[:, ue]))
    return uelabel

def maxEffectiveRate_with_PC(channel,maxPower,noise):
    BSNum, userNum = channel.shape
    previousPower = deepcopy(maxPower)
    currentPower = deepcopy(maxPower)
    x = np.zeros((BSNum, userNum))
    y = np.zeros((BSNum, 1))
    mu = np.zeros((BSNum, userNum))
    belta = np.zeros((BSNum, userNum))
    iterationNum1 = 30
    iterationNum2 = 50
    iterationNum3 = 50
    object1 = np.zeros((iterationNum1, 1))
    object2 = np.zeros((iterationNum2, 1))
    object3 = np.zeros((iterationNum3, 1))
    rate, SINR = calculatingRateSINR(previousPower, channel, noise)
    step = 0.5
    epsilon = 1e-3
    iter1 = 0


    while iter1 < iterationNum1:
        for k in range( userNum):
            maxRate = -np.inf
            for n in range(BSNum):
                if maxRate < rate[n, k]:
                    maxRate = rate[n, k]
                id = n
            x[id, k] = 1
        for n in range(BSNum):
            y[n] = np.sum(x[n,:])

        for n in range(BSNum):
            for k in range( userNum):
                mu[n, k] = x[n, k] / (1 + y[n])
                belta[n, k] = rate[n, k] / (1 + y[n])

        iter2 = 0
        while iter2 < iterationNum2:
            x = np.zeros((BSNum, userNum))
            for k in range( userNum):
                maxUtility = -np.inf
                for n in range( BSNum):
                    utility = belta[n, k] - np.sum(mu[n,:]*belta[n,:])
                    if maxUtility < utility:
                        maxUtility = utility
                        id = n
                x[id, k] = 1
            for n in range(BSNum):
                y[n] = np.sum(x[n,:])
            sx1 = np.zeros((BSNum, userNum))
            sx2 = np.zeros((BSNum, userNum))

            for n in range( BSNum):
                for k in range( userNum):
                    sx1[n, k] = x[n, k] - mu[n, k] * (1 + y[n])
                    sx2[n, k] = rate[n, k] - belta[n, k] * (1 + y[n])
            if sx1.all() == 0:
                if sx2.all() == 0:
                    break

            mu1 = mu
            belta1 = belta
            phi1 = np.zeros((BSNum, userNum))
            varphi1 = np.zeros((BSNum, userNum))
            phi2 = np.zeros((BSNum, userNum))
            varphi2 = np.zeros((BSNum, userNum))
            for n in range(BSNum):
                for k in range(userNum):
                    phi1[n, k] = mu1[n, k] * (1 + y[n]) - x[n, k]
                    varphi1[n, k] = belta[n, k] * (1 + y[n]) - rate[n, k]

            for t in range(50):
                for n in range( BSNum):
                    for k in range( userNum):
                        mu1[n, k] = mu1[n, k] - step ** t * phi1[n, k] / (1 + y[n])
                        belta1[n, k] = belta1[n, k] - step ** t * varphi1[n, k] / (1 + y[n])
                        phi2[n, k] = mu1[n, k] * (1 + y[n]) - x[n, k]
                        varphi2[n, k] = belta1[n, k] * (1 + y[n]) - rate[n, k]
                if np.sum(np.sum(phi2 ** 2)) + np.sum(np.sum(varphi2** 2)) <= (1 - epsilon * step ** t) ** 2 * (np.sum(np.sum(phi1** 2)) + np.sum(np.sum(varphi1** 2))):
                    break
            for n in range( BSNum):
                for k in range( userNum):
                    mu[n, k] = mu[n, k] - step ** t * phi1[n, k] / (1 + y[n])
                    belta[n, k] = belta[n, k] - step ** t * varphi1[n, k] / (1 + y[n])
            mu = mu / np.sum(np.sum(mu))
            object2[iter2] = np.sum(np.sum(x* belta))
            iter2 = iter2 + 1
        z = np.zeros((BSNum, userNum))
        for n in range(BSNum):
            if y[n]!=0:
                for k in range( userNum):
                    z[n, k] = x[n, k] / y[n]

        previousPower = maxPower
        simpleSINR = calculatingSimpleSINR(z, previousPower, channel, noise)
        iter3 = 0
        while iter3 < iterationNum3:
            for m in range( BSNum):
                S1 = 0
                for n in range( BSNum):
                    if n!=m:
                        for k in range(userNum):
                            S1 = S1 + simpleSINR[n, k] * channel[m, k]
                currentPower[m] = min(np.sum(z[m,:]) / S1, maxPower[m])

            previousPower = currentPower
            simpleSINR = calculatingSimpleSINR(z, previousPower, channel, noise)
            rate, SINR = calculatingRateSINR(previousPower, channel, noise)
            for n in range( BSNum):
                for k in range( userNum):
                    object3[iter3] = object3[iter3] + z[n, k] * rate[n, k]

            iter3 = iter3 + 1
        for n in range(BSNum):
            for k in range(userNum):
                object1[iter1] = object1[iter1] + z[n, k] * rate[n, k]
        iter1 = iter1 + 1
    uelabel = []
    for ue in range(userNum):
        uelabel.append(np.argmax(x[:, ue]))
    return uelabel, currentPower



