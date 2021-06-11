import pickle
import matplotlib.pyplot as plt
import numpy as np


list_a = list(range(1000))
list_b = [i*2 for i in range(317)]
list_c = [i*3 for i in range(256)]
list_d = [i*4 for i in range(1530)]

with open("Paint_a.pickle", "wb") as f:
    pickle.dump(list_a, f)

with open("Paint_b.pickle", "wb") as f:
    pickle.dump(list_b, f)

with open("Paint_c.pickle", "wb") as f:
    pickle.dump(list_c, f)

with open("Paint_d.pickle", "wb") as f:
    pickle.dump(list_d, f)

with open("Paint_a.pickle", "rb") as f:
    a = pickle.load(f)

with open("Paint_b.pickle", "rb") as f:
    b = pickle.load(f)

with open("Paint_c.pickle", "rb") as f:
    c = pickle.load(f)

with open("Paint_d.pickle", "rb") as f:
    d = pickle.load(f)

len_a = len(a)
len_b = len(b)
len_c = len(c)
len_d = len(d)

sampleNo_a = 2000 - len_a
sampleNo_b = 2000 - len_b
sampleNo_c = 2000 - len_c
sampleNo_d = 2000 - len_d
mu = 0
sigma = 5
np.random.seed(0)
s_a = np.random.normal(mu, sigma, sampleNo_a )
s_b = np.random.normal(mu, sigma, sampleNo_b )
s_c = np.random.normal(mu, sigma, sampleNo_c )
s_d = np.random.normal(mu, sigma, sampleNo_d )
# print(s_a)
print(type(s_a))

for idx in range(2000-len_a):
    a.append(a[len_a-1]+s_a[idx])
for idx in range(2000-len_b):
    b.append(b[len_b-1]+s_b[idx])
for idx in range(2000-len_c):
    c.append(c[len_c-1]+s_c[idx])
for idx in range(2000-len_d):
    d.append(d[len_d-1]+s_d[idx])

# plt.figure(figsize=(9, 7))
# ,marker = 'o',markerfacecolor='r',markersize = 10
plt.plot(range(len(a)), a, color='blue', linestyle='-', label='a', linewidth=3)
plt.plot(range(len(b)), b, color='red', linestyle='--', label='b', linewidth=3)
plt.plot(range(len(c)), c, color='green', marker = 'o', markersize = 3, linestyle='-.', label='c', linewidth=3)
plt.plot(range(len(d)), d, color='magenta', marker = 'v', markersize = 3, linestyle=':',label='d', linewidth=3)

plt.ylabel('valid reward',size=20)
plt.xlabel('epochs',size=20)
plt.grid()  # 此参数为默认参数
# plt.xticks(np.arange(0, 251, step=50), fontsize=20)
# plt.yticks(np.arange(0, 0.361, step=0.09), fontsize=20)
plt.legend(loc='best')
plt.title('Valid Reward Plot')
plt.savefig('paint.png')
plt.close()