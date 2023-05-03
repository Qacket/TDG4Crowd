
import pandas as pd
from pylab import *
import seaborn as sns

data_scale_range = range(300, 1350, 150)

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
generate_methods = ['DS_generate', 'HDS_generate', 'MV_generate', 'GLAD_generate', 'RDG_generate', 'Mymodel_generate']
names = ["300", "450", "600", "750", "900", "1050", "1200"]
x = range(len(names))
def MSE(y, t):
    return 0.5 * np.sum((y - t)**2)

mse = []
for number in data_scale_range:

    real_sort = np.argsort(pd.read_csv('../../datasets/domain_relation_4/' + str(number) + '/mean.csv').values[0])

    for generate_method_name in generate_methods:

        acc = pd.read_csv('../../datasets/domain_relation_4/' + str(number) + '/' + generate_method_name + '/accuracy.csv').values
        acc_sort = np.argsort(acc)
        mse_item = []
        for i in range(len(acc_sort)):
            mse_item.append(MSE(real_sort, acc_sort[i]))
        mse.append(mean(mse_item))

mse = np.array(mse).reshape(-1, 6)
print(mse)

# mse = []
# for number in data_scale_range:
#     mean = pd.read_csv('../../datasets/domain_relation_4/' + str(number) + '/mean.csv').values
#     mean = np.argsort(mean)
#
#     for i in range(1, len(mean)):
#         mse.append(MSE(mean[0], mean[i]))
#
# mse = np.array(mse).reshape(-1, 6)
# print(mse)
    # std = pd.read_csv('../../datasets/domain_relation_1/' + str(number) + '/std.csv')

    # print(std)

y0 = mse[:, 0]
y1 = mse[:, 1]
y2 = mse[:, 2]
y3 = mse[:, 3]
y4 = mse[:, 4]
y5 = mse[:, 5]


plt.errorbar(x, y0,  color='c', capsize=3, linewidth=1, marker='+', mec='c', mfc='c', ms=7, label=u'G_DS')
plt.errorbar(x, y1,  color='r', capsize=3, linewidth=1, marker='^', mec='r', mfc='r', ms=7, label=u'G_HDS')
plt.errorbar(x, y2,  color='g', capsize=3, linewidth=1, marker='*', mec='g', mfc='g', ms=7, label=u'G_MV')
plt.errorbar(x, y3,  color='b', capsize=3, linewidth=1, marker='o', mec='b', mfc='b', ms=7, label=u'G_GLAD')
plt.errorbar(x, y4,  color='y', capsize=3, linewidth=1, marker='x', mec='y', mfc='y', ms=7, label=u'G_IRT')
plt.errorbar(x, y5,  color='black', capsize=3, linewidth=1, marker='v', mec='black', mfc='black', ms=7, label=u'DGCrowd')


leg = plt.legend(prop={'size': 10}, frameon=False)


plt.xticks(x, names)
# plt.yscale('log')
# plt.ylim(top=1e-1)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Task numbers")  # X轴标签
plt.ylabel("MSE")  # Y轴标签


plt.show()
