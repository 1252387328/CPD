import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import xlrd

matplotlib.use('TkAgg')

sns.set(style='darkgrid', color_codes=True,rc={'axes.grid': False,'axes.edgecolor': 'black'})
tag=list(range(1,102))
for i in range(101):
    tag[i]=str(tag[i])
# 这里是创建一个数据
vegetables = tag

farmers = tag

readbook = xlrd.open_workbook("./data/task4.xlsx")
sheet=readbook.sheet_by_index(0)
nrowsmax = sheet.nrows
print(nrowsmax)
DSM=np.zeros((99,99))
for i in range(1, nrowsmax):
    DSM[int(sheet.cell(i, 0).value)-1][int(sheet.cell(i, 1).value)-1]=(sheet.cell(i, 2).value)
print(DSM)

def add_node(x):
    return "node " + str(x)
vfunc = np.vectorize(add_node)

harvest = DSM
# 这里是创建一个画布
fig, ax = plt.subplots()
im = ax.imshow(harvest,cmap='YlGnBu')

# 这里是修改标签
# We want to show all ticks...
# ax.set_xticks(np.arange(len(farmers)))
# ax.set_yticks(np.arange(len(vegetables)))
# x=np.arange(1,len(vegetables),10)

xa=np.arange(1,len(vegetables)+1,5)
ya=np.arange(1,len(farmers)+1,5)

xa1 = vfunc(xa)
ya1 = vfunc(ya)

ax.set_xticks(xa)
ax.set_yticks(ya)
# ... and label them with the respective list entries
ax.set_xticklabels(xa1)
ax.set_yticklabels(ya1)

# 因为x轴的标签太长了，需要旋转一下，更加好看
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.xticks(fontname='Times New Roman',fontsize=8)
plt.yticks(fontname='Times New Roman',fontsize=8)


# # 添加每个热力块的具体数值
# # Loop over data dimensions and create text annotations.
# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         text = ax.text(j, i, harvest[i, j],
#                        ha="center", va="center", color="w")
# # ax.set_title("Harvest of local farmers (in tons/year)")

# fig.tight_layout()
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
cb=plt.colorbar(im,cax=cax)
cb.ax.tick_params(labelsize=8)
plt.savefig("dsm.pdf",bbox_inches='tight')
plt.savefig("dsm.eps",bbox_inches='tight')
plt.savefig("dsm.png", bbox_inches='tight', dpi=600)
plt.show()