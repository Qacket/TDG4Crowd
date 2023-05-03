# encoding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['Arial']

sns.set_theme(style="whitegrid")
plt.grid(linewidth=0.2, alpha=0.5)
ax = plt.gca() # 获取当前的axes
ax.spines['right'].set_color('black')
ax.spines['right'].set_linewidth(0.5)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(0.5)
ax.spines['top'].set_color('black')
ax.spines['top'].set_linewidth(0.5)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(0.5)
names = ['1000', '2000', '3000', '4000', '5000', '6000', '7000']
x = range(len(names))

for i in range(8):
    mean = pd.read_csv('../../datasets/labelme_1/KS/KS_stat_mean_' + str(i) + '.csv')
    std = pd.read_csv('../../datasets/labelme_1/KS/KS_stat_std_' + str(i) + '.csv')


    path = '../../result/labelme_exp1/labelme_KS_exp1/exp1_label_' + str(i) + '.pdf'

    plt.title('exp1_label_' + str(i))
    y0 = list(mean['0'])
    y1 = list(mean['1'])
    y2 = list(mean['2'])
    y3 = list(mean['3'])
    y4 = list(mean['4'])
    y5 = list(mean['5'])
    ye0 = list(std['0'])
    ye1 = list(std['1'])
    ye2 = list(std['2'])
    ye3 = list(std['3'])
    ye4 = list(std['4'])
    ye5 = list(std['5'])


    # plt.plot(x, y, color=线条颜色, linestyle=线条类型, linewidth=线条宽度,
    # marker=标记类型 , markeredgecolor=标记边框颜色, markeredgwidth=标记边框宽度 , markerfacecolor=标记填充颜色, markersize=标记大小,
    # label=线条标签)
    plt.errorbar(x, y0, yerr=ye0, color='c', capsize=3, linewidth=0.3, marker='+', mec='c', mfc='c', ms=7, label=u'G_DS')
    plt.errorbar(x, y1, yerr=ye1, color='r', capsize=3, linewidth=0.3, marker='^', mec='r', mfc='r', ms=7, label=u'G_HDS')
    plt.errorbar(x, y2, yerr=ye2, color='g', capsize=3, linewidth=0.3, marker='*', mec='g', mfc='g', ms=7, label=u'G_MV')
    plt.errorbar(x, y3, yerr=ye3, color='b', capsize=3, linewidth=0.3, marker='o', mec='b', mfc='b', ms=7, label=u'G_GLAD')
    plt.errorbar(x, y4, yerr=ye4, color='y', capsize=3,  linewidth=0.3, marker='x', mec='y', mfc='y', ms=7, label=u'G_RDG')
    plt.errorbar(x, y5, yerr=ye5, color='black', capsize=3,  linewidth=0.3, marker='v', mec='black', mfc='black', ms=7, label=u'G_Mymodel')
    leg = plt.legend(prop={'size': 8})

    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.5)
    plt.xticks(x, names)
    plt.yscale('log')
    # plt.ylim(top=1e-1)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"task numbers") #X轴标签
    plt.ylabel("KL divergence") #Y轴标签
    plt.gcf().savefig(path, dpi=600, format='pdf')
    plt.show()






