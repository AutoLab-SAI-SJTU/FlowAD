import os
import numpy as np
import matplotlib.pyplot as plt

total_epoch = 24
metrics_names = ['mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE', 'NDS']

mAP_1 = []
mATE_1 = []
mASE_1 = []
mAOE_1 = []
mAVE_1 = []
mAAE_1 = []
NDS_1 = []

mAP_2 = []
mATE_2 = []
mASE_2 = []
mAOE_2 = []
mAVE_2 = []
mAAE_2 = []
NDS_2 = []

metrics_1 = [mAP_1, mATE_1, mASE_1, mAOE_1, mAVE_1, mAAE_1, NDS_1]
metrics_2 = [mAP_2, mATE_2, mASE_2, mAOE_2, mAVE_2, mAAE_2, NDS_2]

len_metrics = len(metrics_1)

path_1 = './'
# path_2 = './'

path_2 = './'

end_point = 68

start_point_1 = 0

with open(path_1, 'r', encoding='utf-8') as f:
    contents = f.readlines()

flag = 0
for line in contents:
    if flag == 1:
        for name_idx, name in enumerate(metrics_names):
            if name in line:
                metric_tmp = float(line.split(name+':')[-1])
                if len(metrics_1[name_idx]) == 0 or (len(metrics_1[name_idx]) != 0 and metrics_1[name_idx][-1] != metric_tmp):
                    metrics_1[name_idx].append(metric_tmp)

                if name == 'NDS':
                    flag = 0
                continue
    if 'mAP:' in line:
        flag = 1
        metric_tmp = float(line.split('mAP:')[-1])
        if len(metrics_1[0]) == 0 or (len(metrics_1[0]) != 0 and metrics_1[0][-1] != metric_tmp):
            metrics_1[0].append(metric_tmp)

with open(path_2, 'r', encoding='utf-8') as f:
    contents = f.readlines()

flag = 0
for line in contents:
    if flag == 1:
        for name_idx, name in enumerate(metrics_names):
            if name in line:
                metric_tmp = float(line.split(name+':')[-1])
                if len(metrics_2[name_idx]) == 0 or (len(metrics_2[name_idx]) != 0 and metrics_2[name_idx][-1] != metric_tmp):
                    metrics_2[name_idx].append(metric_tmp)
                if name == 'NDS':
                    flag = 0
                continue
    if 'mAP:' in line:
        flag = 1
        metric_tmp = float(line.split('mAP:')[-1])
        if len(metrics_2[0]) == 0 or (len(metrics_2[0]) != 0 and metrics_2[0][-1] != metric_tmp):
            metrics_2[0].append(metric_tmp)

print([len(one) for one in metrics_1])
print([len(one) for one in metrics_2])
print(metrics_1[0])
print(metrics_2[0])

plt.figure(figsize=(20, 10), dpi=300)
plt.gca()
# plt.axis('off')

legend_size = 28
tick_size = 24
label_size = 28
title_size = 32

for metric_id, metric_name in enumerate(metrics_names):
    plt.subplot(2,4,metric_id+1)
    plt.grid(which='major', axis='both', linestyle='-.', zorder=0) # color='r', linestyle='-', linewidth=2

    metrics_1[metric_id] = metrics_1[metric_id][-24:]
    metrics_2[metric_id] = metrics_2[metric_id][-24:]

    # epoches = list(range(1,1+len(metrics_1[metric_id])))
    # plt.plot(epoches[:end_point], metrics_1[metric_id][:end_point], marker='o', linestyle='-', color='b', label='Official')
    # epoches = list(range(1,1+len(metrics_2[metric_id])))
    # plt.plot(epoches[:end_point], metrics_2[metric_id][:end_point], marker='o', linestyle='-', color='r', label='clip')

    epoches = np.array(list(range(1,1+len(metrics_1[metric_id]))))
    plt.plot(epoches[:end_point][start_point_1:], metrics_1[metric_id][:end_point][start_point_1:], marker='o', linestyle='-', color='b', label='Official')
    epoches = np.array(list(range(1,1+len(metrics_2[metric_id]))))
    plt.plot(epoches[:(-start_point_1+end_point)]+start_point_1, metrics_2[metric_id][:(-start_point_1+end_point)], marker='o', linestyle='-', color='r', label='clip')

    plt.legend()
    plt.title(metric_name)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    
    # plt.legend(fontsize=legend_size)
    # plt.title(metric_name, fontsize=title_size)
    # plt.xlabel('Epoch', fontsize=label_size)
    # plt.ylabel('Value', fontsize=label_size)
    # plt.xticks(fontsize=tick_size)
    # plt.yticks(fontsize=tick_size)


plt.tight_layout()

# plt.savefig('./')
plt.savefig('./')