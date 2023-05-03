import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import seaborn as sns


mpl.rcParams['font.sans-serif'] = ['Arial']

sns.set_theme(style="whitegrid")
plt.grid(linewidth=0.2, alpha=0.5)
ax = plt.gca() # 获取当前的axes
ax.spines['right'].set_color('black')
ax.spines['right'].set_linewidth(1)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(1)
ax.spines['top'].set_color('black')
ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(1)
names = ["30", "45", "60", "75", "90", "105", "120"]
x = range(len(names))


crowd_file = '../../datasets/test_data/domain_relation_crowd_test.txt'
truth_file = '../../datasets/test_data/domain_relation_truth_test.txt'

gt2t = {}
f_truth_open = open(truth_file, 'r')
reader = f_truth_open.readlines()
reader = [line.strip("\n") for line in reader]
for line in reader:
    task, gt = line.split('\t')
    gt = int(gt)
    if gt not in gt2t:
        gt2t[gt] = []
    gt2t[gt].append(task)

f_truth_open.close()

gt_sum = len(gt2t.keys())

gt2t = sorted(gt2t.items(), key=lambda item: item[0], reverse=False)

classification_task = []

for i in range(len(gt2t)):
    classification_task.append(gt2t[i][1])

print(classification_task)
task_sum = []
for k in classification_task:
    task_sum.append(len(k))
print(task_sum)
task_p = []
for k in task_sum:
    task_p.append(k/sum(task_sum))
print(task_p)

mean = pd.DataFrame()
std = pd.DataFrame()
for i in range(13):

    mean_sub = pd.read_csv('../../datasets/domain_relation_2/KL/KL_mean_' + str(i) + '.csv')
    std_sub = pd.read_csv('../../datasets/domain_relation_2/KL/KL_std_' + str(i) + '.csv')
    if i == 0:
        mean = task_p[i]*mean_sub
        std = task_p[i]*std_sub
    else:
        mean += task_p[i]*mean_sub
        std += task_p[i]*std_sub


path = '../../result/domain_relation_exp2/domain_relation_KL_exp2/exp2_KL.pdf'

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
plt.errorbar(x, y0, yerr=ye0, color='c', capsize=3, linewidth=1, marker='+', mec='c', mfc='c', ms=7, label=u'G_DS')
plt.errorbar(x, y1, yerr=ye1, color='r', capsize=3, linewidth=1, marker='^', mec='r', mfc='r', ms=7, label=u'G_HDS')
plt.errorbar(x, y2, yerr=ye2, color='g', capsize=3, linewidth=1, marker='*', mec='g', mfc='g', ms=7, label=u'G_MV')
plt.errorbar(x, y3, yerr=ye3, color='b', capsize=3, linewidth=1, marker='o', mec='b', mfc='b', ms=7, label=u'G_GLAD')
plt.errorbar(x, y4, yerr=ye4, color='y', capsize=3, linewidth=1, marker='x', mec='y', mfc='y', ms=7, label=u'G_IRT')
plt.errorbar(x, y5, yerr=ye5, color='black', capsize=3, linewidth=1, marker='v', mec='black', mfc='black', ms=7,
             label=u'DGCrowd')
leg = plt.legend(prop={'size': 10}, frameon=False)


plt.xticks(x, names)
plt.yscale('log')
# plt.ylim(top=1e-1)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Annotator numbers")  # X轴标签
plt.ylabel("KL divergence")  # Y轴标签

plt.gcf().savefig(path, dpi=600, format='pdf')
plt.show()
