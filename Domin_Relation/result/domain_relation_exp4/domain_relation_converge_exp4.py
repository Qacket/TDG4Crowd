# encoding=utf-8
import pandas as pd
from pylab import *
import seaborn as sns

data_scale_range = range(300, 1350, 150)


for number in data_scale_range:
    mpl.rcParams['font.sans-serif'] = ['Arial']
    sns.set_theme(style="whitegrid")
    plt.grid(linewidth=0.8, alpha=1)
    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)

    x = np.array([0, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3])
    names = [r"$\bf{real\ dataset}$", 'G_DS', 'G_HDS', 'G_MV', 'G_GLAD', 'G_IRT', r"$\bf{DGCrowd}$"]



    mean = pd.read_csv('../../datasets/domain_relation_4/' + str(number) + '/mean.csv')
    std = pd.read_csv('../../datasets/domain_relation_4/' + str(number) + '/std.csv')
    path = '../../result/domain_relation_exp4/domain_relation_converge_exp4/exp4_converge_' + str(number) + '.pdf'

    dict = {'0': ['DS', 'b', '////'],
            '1': ['HDS', 'g', '\\\\\\'],
            '2': ['MV', 'lightcoral', 'xxxx'],
            '3': ['GLAD', 'y', '----']}
    sorted_indices = list(np.argsort(mean[0:1].values[0]))

    print(sorted_indices)
    a = str(sorted_indices[0])
    b = str(sorted_indices[1])
    c = str(sorted_indices[2])
    d = str(sorted_indices[3])

    y0 = list(mean[a])
    y1 = list(mean[b])
    y2 = list(mean[c])
    y3 = list(mean[d])

    ye0 = list(std[a])
    ye1 = list(std[b])
    ye2 = list(std[c])
    ye3 = list(std[d])
    bar_width = 0.2
    # plt.plot(x, y, color=线条颜色, linestyle=线条类型, linewidth=线条宽度,
    # marker=标记类型 , markeredgecolor=标记边框颜色, markeredgwidth=标记边框宽度 , markerfacecolor=标记填充颜色, markersize=标记大小,
    # label=线条标签)
    plt.bar(x, y0, bar_width, color='w', yerr=ye0, capsize=3, edgecolor=dict[a][1], hatch=dict[a][2], label=dict[a][0])
    plt.bar(x+bar_width, y1, bar_width, color='w', capsize=3, yerr=ye1, edgecolor=dict[b][1], hatch=dict[b][2], label=dict[b][0])
    plt.bar(x+bar_width*2, y2, bar_width, color='w', capsize=3, yerr=ye2, edgecolor=dict[c][1], hatch=dict[c][2], label=dict[c][0])
    plt.bar(x+bar_width*3, y3, bar_width, color='w', capsize=3, yerr=ye3, edgecolor=dict[d][1], hatch=dict[d][2], label=dict[d][0])

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[sorted_indices.index(0)], handles[sorted_indices.index(1)], handles[sorted_indices.index(2)], handles[sorted_indices.index(3)]]
    labels = [labels[sorted_indices.index(0)], labels[sorted_indices.index(1)], labels[sorted_indices.index(2)], labels[sorted_indices.index(3)]]
    leg = ax.legend(handles, labels, loc=2)

    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.5)

    plt.xticks(x+bar_width, names)
    plt.xticks(fontsize=7)
    plt.ylim(0.5, 1)
    plt.grid(True, axis='y', ls=':', color='black', alpha=0.3)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(str(number) + " instances")  # Y轴标签
    plt.ylabel("ACC") #Y轴标签

    plt.gcf().savefig(path, dpi=600)
    plt.show()