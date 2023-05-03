# encoding=utf-8
import pandas as pd
from pylab import *
import seaborn as sns
data_scale_range = [100, 200, 300, 450, 600, 750, 900, 1050, 1200]

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

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
names = ['100', '200', "300", "450", "600", "750", "900", "1050", "1200"]

path = '../../result/domain_relation_exp3/domain_relation_converge_exp3/exp3_converge_ori.pdf'

dict = {'0': 'DS',
        '1': 'HDS',
        '2': 'MV',
        '3': 'GLAD'}
y0 = []
y1 = []
y2 = []
y3 = []

ye0 = []
ye1 = []
ye2 = []
ye3 = []

for number in data_scale_range:
    print(number)
    mean = pd.read_csv('../../datasets/domain_relation_3/' + str(number) + '/mean.csv')
    std = pd.read_csv('../../datasets/domain_relation_3/' + str(number) + '/std.csv')
    sorted_indices = list(np.argsort(mean[0:1].values[0]))
    print(sorted_indices)

    # a = str(sorted_indices[0])
    # b = str(sorted_indices[1])
    # c = str(sorted_indices[2])
    # d = str(sorted_indices[3])

    y0.append(list(mean['0'])[0])
    y1.append(list(mean['1'])[0])
    y2.append(list(mean['2'])[0])
    y3.append(list(mean['3'])[0])

    ye0.append(list(std['0'])[0])
    ye1.append(list(std['1'])[0])
    ye2.append(list(std['2'])[0])
    ye3.append(list(std['3'])[0])

print(y0)
print(y1)
print(y2)
print(y3)

bar_width = 0.16
# plt.plot(x, y, color=线条颜色, linestyle=线条类型, linewidth=线条宽度,
# marker=标记类型 , markeredgecolor=标记边框颜色, markeredgwidth=标记边框宽度 , markerfacecolor=标记填充颜色, markersize=标记大小,
# label=线条标签)
plt.bar(x, y0, bar_width, color='w', yerr=ye0, capsize=3, edgecolor='b', hatch='////', label=dict['0'])
plt.bar(x+bar_width, y1, bar_width, color='w', capsize=3, yerr=ye1, edgecolor='g', hatch='\\\\\\', label=dict['1'])
plt.bar(x+bar_width*2, y2, bar_width, color='w', capsize=3, yerr=ye2, edgecolor='lightcoral', hatch='xxxx', label=dict['2'])
plt.bar(x+bar_width*3, y3, bar_width, color='w', capsize=3, yerr=ye3, edgecolor='y', hatch='----', label=dict['3'])
leg = plt.legend()
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(0.5)

plt.xticks(x+bar_width, names)
plt.xticks(fontsize=7)
plt.ylim(0.5, 0.8)
plt.grid(True, axis='y', ls=':', color='black', alpha=0.3)
plt.subplots_adjust(bottom=0.15)

plt.xlabel("instances")  # Y轴标签
plt.ylabel("ACC") #Y轴标签

plt.gcf().savefig(path, dpi=600)
plt.show()