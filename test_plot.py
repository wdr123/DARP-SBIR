import numpy as np
import matplotlib.pyplot as plt

time = np.arange(10)
temp = np.random.random(10)*30
swdown = np.random.random(10)*100-10
rn = np.random.random(10)*100-10

# fig = plt.figure()
# ax = fig.add_subplot(111)
# lns1 = ax.plot(time,swdown,'-',label='swdown', linewidth=3)
# lns2 = ax.plot(time,rn,'-',label='rn', linewidth=3)
# ax.legend(loc='best')
# ax.grid(axis='both') # 此参数为默认参数
# plt.show()

plt.figure(figsize=(9,7))
# ax = plt.subplot(121)
# print(ax)
plt.plot(time, swdown,'-',label='a',linewidth=3)
plt.ylabel('training loss',size=20)
plt.xlabel('epochs',size=20)
plt.grid() # 此参数为默认参数
plt.xticks(np.arange(0, 300, step=50), fontsize=20)
plt.yticks(np.arange(0, 300, step=50), fontsize=20)
# plt.legend(loc='best')
# plt.title('Training Loss Plot')
plt.savefig('train_loss.png')
plt.close()
# plt.show()