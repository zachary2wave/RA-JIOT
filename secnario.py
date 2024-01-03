import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from random import sample
from collections import defaultdict
from collections import namedtuple

class scenario():

    def __init__(self, area, step_size, num_block, AP_num, low_bound):
        self.area = area
        self.step_size = step_size
        self.UE_step = int(self.area / self.step_size)
        self.num_block = num_block
        self.AP_num = AP_num + 5
        self.lowband = low_bound
        self.block = self.block_generate()
        self.AP_loc, self.UE_loc = self.loc_generate()
        # channel parameters
        self.channel_parameter = {"theta_los": 2.09, "theta_Nlos": 3.75,
                                  "DL":10.38, "NDL": 14.54}
        self.UE_num = len(self.UE_loc["x"])
        self.pl = self.channel()
        self.pl_ap = self.channel_ap()
        self.adj_ap2ue, self.adj_ue2ap = self.adj_total()

        unconver = [ue for ue in range(self.UE_num) if len(self.adj_ue2ap[ue]) == 0]
        oneconnect = [ue for ue in range(self.UE_num) if len(self.adj_ue2ap[ue]) == 1]

        self.UE_index = set(range(self.UE_num)) - set(unconver)
        self.UE_train = self.UE_index - set(oneconnect)

    def active_UE(self, UE_num):
        # activate for train
        rand_indices = np.random.permutation(list(self.UE_train))
        activate = list(rand_indices[:UE_num])
        pl = self.pl[:, activate]
        adj_ue, connect_ap = {}, []
        for flag, ue in enumerate(activate):
            adj_ue[flag] = self.adj_ue2ap[ue]
            connect_ap.append(self.adj_ue2ap[ue])
        adj_ap = {}
        for flag1 in range(self.AP_num):
            adj_ap[flag1] = set()
            for flag2, ue in enumerate(activate):
                if self.pl[flag1, ue] < self.lowband:
                    adj_ap[flag1].add(flag2)
        return pl, adj_ue, adj_ap, activate

    def active_part_UE(self, UE_num, activatenum):
        # activate for train
        rand_indices = np.random.permutation(list(self.UE_train))
        activate = list(rand_indices[:activatenum])
        pl = self.pl[:, activate]
        adj_ue, connect_ap = {}, []
        for flag, ue in enumerate(activate):
            adj_ue[flag] = self.adj_ue2ap[ue]
            connect_ap.append(self.adj_ue2ap[ue])
        adj_ap = {}
        for flag1 in range(self.AP_num):
            adj_ap[flag1] = set()
            for flag2, ue in enumerate(activate):
                if self.pl[flag1, ue] < self.lowband:
                    adj_ap[flag1].add(flag2)
        pl = np.concatenate((pl, np.zeros(shape=(self.AP_num,UE_num-activatenum))))
        return pl, adj_ue, adj_ap, activate




    def adj_total(self):
        adj_ap2ue = {}
        for ap in range(self.AP_num):
            adj_ap2ue[ap] = set()
            for ue in range(self.UE_num):
                if self.pl[ap, ue] < self.lowband:
                    adj_ap2ue[ap].add(ue)
        adj_ue2ap = {}
        for ue in range(self.UE_num):
            adj_ue2ap[ue] = set()
            for ap in range(self.AP_num):
                if self.pl[ap, ue] < self.lowband:
                    adj_ue2ap[ue].add(ap)
        return adj_ap2ue, adj_ue2ap

    def adj_gene(self, UE):
        # UE is the first 100 nodes index is 0~99
        adj = {}
        for flag, ue in enumerate(UE):
            adj[flag] = set()
            for ap in range(self.AP_num):
                if self.pl[ap, ue] < self.lowband:
                    adj[flag].add(ap)
        # AP is the first 100 nodes index is 0~99
        for ap in range(self.AP_num):
            adj[ap] = set()
            for flag, ue in enumerate(UE):
                if self.pl[ap, ue] < self.lowband:
                    adj[ap].add(flag)
        return adj

    def label_gene(self, activate):
        UElabel = []
        for ue in activate:
            UElabel.append(int(np.argmin(self.pl[:, ue])))
        return UElabel


    def need_require(self,UE):
        return np.random.normal(500,100,UE)

    def block_generate(self):
        L = [x for x in range(20)]
        X = np.array(sample(L, self.num_block))
        Y = np.array(sample(L, self.num_block))
        blockx = (self.area/20) * X
        blocky = (self.area/20) * Y
        blockw = (self.area/10) * np.ones(self.num_block)
        blockl = (self.area/10) * np.ones(self.num_block)
        blockh = 30 * np.ones(self.num_block)
        block = {"x":blockx, "y":blocky,"w":blockw, "l":blockl, "h": blockh }
        return block

    def loc_generate(self):
        self.MAPx1, self.MAPy1 = self.area / 2, self.area / 2
        self.MAPx2, self.MAPy2 = self.area / 4, self.area / 4
        self.MAPx3, self.MAPy3 = self.area / 4 * 3, self.area / 4
        self.MAPx4, self.MAPy4 = self.area / 4, self.area / 4 * 3
        self.MAPx5, self.MAPy5 = self.area / 4 * 3, self.area / 4 * 3

        APx_label = np.array([self.MAPx1,self.MAPx2,self.MAPx3,self.MAPx4,self.MAPx5])
        APy_label = np.array([self.MAPy1,self.MAPy2,self.MAPy3,self.MAPy4,self.MAPy5])
        APz_label = np.array([0])
        for q in range(0, self.AP_num):
            suitable = False
            while not suitable:
                tempx = np.random.uniform(0, self.area)
                tempy = np.random.uniform(0, self.area)
                dis = (tempx - APx_label) ** 2 + (tempy - APy_label) ** 2
                if np.min(dis) > 98:
                    block_num = self.inblock(tempx, tempy)
                    if block_num > 0:
                        APz_label = np.append(APz_label, self.block["h"][block_num-1])
                    else:
                        APz_label = np.append(APz_label, 8)
                    APx_label = np.append(APx_label, tempx)
                    APy_label = np.append(APy_label, tempy)
                    suitable = True
        UEx_label = []
        UEy_label = []
        for x in np.arange(0, self.area, self.step_size):
            for y in np.arange(0, self.area, self.step_size):
                if not bool(self.inblock(x, y)):
                    UEx_label.append(x)
                    UEy_label.append(y)
        AP_loc = {"x": APx_label, "y": APy_label, "z": APz_label[1:]}
        UE_loc = {"x": UEx_label, "y": UEy_label}
        return AP_loc, UE_loc

    def channel(self):
        channel = 10000 * np.ones([self.AP_num, self.UE_num])
        for ap in range(self.AP_num):
            for ue in range(self.UE_num):
                point1 = {"x": self.AP_loc["x"][ap],
                          "y": self.AP_loc["y"][ap],
                          "z": self.AP_loc["z"][ap]}
                point2 = {"x": self.UE_loc["x"][ue],
                          "y": self.UE_loc["y"][ue],
                          "z": 0}
                apx, apy, aph = self.AP_loc["x"][ap],self.AP_loc["y"][ap], self.AP_loc["z"][ap]
                uex, uey = self.UE_loc["x"][ue], self.UE_loc["y"][ue]
                apblock, ueblock = self.inblock(apx, apy), self.inblock(uex, uey)
                inline = self.inline(point1, point2)
                los = True
                if len(inline) == 1:
                    if apblock != inline[0] and ueblock!= inline[0] :
                        los = False
                elif len(inline) > 1:
                    los = False
                channel[ap, ue] = self.PL(point1, point2, los)
        return channel

    def channel_ap(self):
        channel = 10000 * np.ones([self.AP_num, self.AP_num])
        for ap1 in range(self.AP_num):
            for ap2 in range(self.AP_num):
                point1 = {"x": self.AP_loc["x"][ap1],
                          "y": self.AP_loc["y"][ap1],
                          "z": self.AP_loc["z"][ap1]}
                point2 = {"x": self.UE_loc["x"][ap2],
                          "y": self.UE_loc["y"][ap2],
                          "z": self.UE_loc["y"][ap2]}
                inline = self.inline(point1, point2)
                channel[ap1, ap2] = self.PL(point1, point2, bool(inline))
        return channel

    def PL(self, point1, point2, los):
        deltax = point2["x"] - point1["x"]
        deltay = point2["y"] - point1["y"]
        deltaz = point2["z"] - point1["z"]
        d = np.sqrt(deltax ** 2 + deltay ** 2 + deltaz ** 2)
        if los:
            return self.channel_parameter["DL"] + self.channel_parameter["theta_los"] * 20 * np.log10(d)
        else:
            return self.channel_parameter["NDL"] + self.channel_parameter["theta_Nlos"] * 20 * np.log10(d)

    def inblock(self, x, y):
        flag = 0
        for time in range(self.num_block):
            if self.block["x"][time] <= x <= self.block["x"][time]+self.block["w"][time] \
                    and self.block["y"][time] <= y <= self.block["y"][time]+self.block["l"][time]:
                flag = time+1
                break
        return flag


    def inline(self, point1, point2):
        # 空间直线穿过立方体
        TDPoint = namedtuple("TDPoint", ["x", "y", "z"])
        point = namedtuple("point", ["max", "min"])

        ori = TDPoint(x=point1["x"], y=point1["y"], z=point1["z"])
        deltax = point2["x"] - point1["x"]
        deltay = point2["y"] - point1["y"]
        deltaz = point2["z"] - point1["z"]
        norm = np.sqrt(deltax ** 2 + deltay ** 2 + deltaz ** 2)
        dirct = TDPoint(x=deltax / norm, y=deltay / norm, z=deltaz / norm)
        flag = []
        for time in range(self.num_block):
            point1 = TDPoint(x=self.block["x"][time], y=self.block["y"][time], z=0)
            point2 = TDPoint(x=self.block["x"][time] + self.block["w"][time],
                             y=self.block["y"][time] + self.block["l"][time],
                             z=self.block["h"][time])
            AABB = point(max=point2, min=point1)
            if self.intersectWithAABB(AABB, [ori, dirct]):
                flag.append( time + 1)
                break
        return flag

    def intersectWithAABB(self, AABB, Ray):
        tmin = 0
        tmax = 10000
        # <editor-fold desc="平行于x轴">
        if (math.fabs(Ray[1].x) < 0.000001):
            if (Ray[0].x < AABB.min.x) or (Ray[0].x > AABB.max.x):
                return False
        else:
            ood = 1.0 / Ray[1].x
            t1 = (AABB.min.x - Ray[0].x) * ood
            t2 = (AABB.max.x - Ray[0].x) * ood
            # t1做候选平面，t2做远平面
            if (t1 > t2):
                temp = t1
                t1 = t2
                t2 = temp
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2
            if tmin > tmax:
                return False
        # </editor-fold>

        # <editor-fold desc="平行于y轴">
        if (math.fabs(Ray[1].y) < 0.000001):
            if (Ray[0].y < AABB.min.y) or (Ray[0].y > AABB.max.y):
                return False
        else:
            ood = 1.0 / Ray[1].y
            t1 = (AABB.min.y - Ray[0].y) * ood
            t2 = (AABB.max.y - Ray[0].y) * ood
            # t1做候选平面，t2做远平面
            if (t1 > t2):
                temp = t1
                t1 = t2
                t2 = temp
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2
            if tmin > tmax:
                return False
        # </editor-fold>

        # <editor-fold desc="平行于z轴">
        if (math.fabs(Ray[1].z) < 0.000001):
            if (Ray[0].z < AABB.min.z) or (Ray[0].z > AABB.max.z):
                return False
        else:
            ood = 1.0 / Ray[1].z
            t1 = (AABB.min.z - Ray[0].z) * ood
            t2 = (AABB.max.z - Ray[0].z) * ood
            # t1做候选平面，t2做远平面
            if (t1 > t2):
                temp = t1
                t1 = t2
                t2 = temp
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2
            if tmin > tmax:
                return False
        # </editor-fold>
        return True

    def show_loss(self):
        point1 = {"x": 0,
                  "y": 0,
                  "z": 0}
        channel_los, channel_Nlos = np.zeros([200]), np.zeros([200])
        r_los, r_Nlos = np.zeros([200]), np.zeros([200])
        for i in range(200):
            point2 = {"x": i+1,
                      "y": 0,
                      "z": 0}
            channel_los[i] = self.PL(point1, point2, True)
            channel_Nlos[i] = self.PL(point1, point2, False)
            r_los[i] = 20 * np.log2(1 + 10**(13-channel_los[i]/10))
            r_Nlos[i] = 20 * np.log2(1 + 10**(13-channel_Nlos[i]/10))
        plt.figure(1)
        plt.plot(channel_los)
        plt.plot(channel_Nlos)
        plt.figure(2)
        plt.plot(r_los)
        plt.plot(r_Nlos)
        plt.show()


    def show_scenario(self, UE, AP):
        def plot_linear_squard(x, y, dx, dy, color='red'):
            xx = [x, x, x + dx, x + dx, x]
            yy = [y, y + dy, y + dy, y, y]
            kwargs = {'alpha': 1, 'color': color}
            block = ax.plot(xx, yy, **kwargs)
            return block

        fig = plt.figure()
        ax = fig.gca()
        ax.cla()

        plt.xlim(0, self.area)
        plt.ylim(0, self.area)

        for i in range(self.num_block):
            plot_linear_squard(self.block["x"][i], self.block["y"][i],
                               self.block["w"][i], self.block["l"][i], color='b')
        plt.scatter(self.AP_loc["x"], self.AP_loc["y"], color="red", marker='1')
        plt.scatter(self.CenterAPx, self.CenterAPy, s=60, marker="*", edgecolors='blue')
        # plt.scatter(self.UE_loc["x"][UE], self.UE_loc["y"][UE], s=60, marker="*", edgecolors='green')

        for i in range(self.num_block):
            plot_linear_squard(self.block["x"][i], self.block["y"][i],
                               self.block["w"][i], self.block["l"][i], color='b')
        adj = self.adj_ap2ue[AP]
        plt.scatter(self.UE_loc["x"][AP], self.UE_loc["y"][AP], s=80, marker="d", edgecolors='black')
        for a in adj:
            plt.plot([self.AP_loc["x"][AP], self.UE_loc["x"][a]],[self.AP_loc["y"][AP], self.UE_loc["y"][a]], linewidth=0.1, linestyle=":", color='yellow')
        plt.show()
        return fig

    def show_channel(self, num):
        import seaborn as sb
        def plot_linear_squard(ax, x, y, dx, dy, color='red'):
            xx = [x, x, x + dx, x + dx, x]
            yy = [y, y + dy, y + dy, y, y]
            kwargs = {'alpha': 1, 'color': color}
            block = ax.plot(yy, xx, **kwargs)
            return block

        fig = plt.figure()
        ax = fig.gca()
        ma = np.zeros((self.UE_step, self.UE_step))
        for i in range(self.UE_num):
            ma[int(self.UE_loc["x"][i]/self.step_size), int(self.UE_loc["y"][i]/self.step_size)] = self.pl[num, i]

        ax = sb.heatmap(ma, ax = ax,cmap="YlGnBu")

        for i in range(self.num_block):
            plot_linear_squard(ax, self.block["x"][i], self.block["y"][i],
                               self.block["w"][i], self.block["l"][i], color='black')
        ax.scatter(self.AP_loc["y"][num], self.AP_loc["x"][num], marker='x', color='red')

        plt.show()


    def show_connect(self, adj, activate, rate):
        def plot_linear_squard(x, y, dx, dy, color='red'):
            xx = [x, x, x + dx, x + dx, x]
            yy = [y, y + dy, y + dy, y, y]
            kwargs = {'alpha': 1, 'color': color}
            block = ax.plot(xx, yy, **kwargs)
            return block

        fig = plt.figure()
        ax = fig.gca()
        ax.cla()

        plt.xlim(0, self.area)
        plt.ylim(0, self.area)

        for i in range(self.num_block):
            plot_linear_squard(self.block["x"][i], self.block["y"][i],
                               self.block["w"][i], self.block["l"][i], color='b')
        plt.scatter(self.AP_loc["x"], self.AP_loc["y"], color="red", marker='1')
        for i in range(self.AP_num):
             plt.text(self.AP_loc["x"][i], self.AP_loc["y"][i], str(i), fontsize=6, color ="red")
        plt.scatter(self.MAPx1, self.MAPy1, s=60, marker="*", edgecolors='blue')
        plt.scatter(self.MAPx2, self.MAPy2, s=60, marker="*", edgecolors='blue')
        plt.scatter(self.MAPx3, self.MAPy3, s=60, marker="*", edgecolors='blue')
        plt.scatter(self.MAPx4, self.MAPy4, s=60, marker="*", edgecolors='blue')
        plt.scatter(self.MAPx5, self.MAPy5, s=60, marker="*", edgecolors='blue')
        # plt.scatter(self.UE_loc["x"][UE], self.UE_loc["y"][UE], s=60, marker="*", edgecolors='green')
        for i in range(self.num_block):
            plot_linear_squard(self.block["x"][i], self.block["y"][i],
                               self.block["w"][i], self.block["l"][i], color='b')
        for flag, ue in enumerate(activate):
            plt.plot([self.AP_loc["x"][adj[flag]], self.UE_loc["x"][ue]],[self.AP_loc["y"][adj[flag]], self.UE_loc["y"][ue]], linewidth=1, linestyle=":")
            plt.text(self.UE_loc["x"][ue],  self.UE_loc["y"][ue], str(round(rate[flag],1)), fontsize=6)

        plt.show()
        return fig



if __name__ == '__main__':
    from util import load_file
    ENV = load_file("save_env/063010")
    ENV.show_loss()