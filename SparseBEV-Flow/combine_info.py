import os
import mmcv
import json

path = 'data/'

files = sorted(os.listdir(path), key=lambda e: int(e.split('.')[0].split('_')[-1]))

print(files)

overall = []
tokens = set()
for file in files:
    info = mmcv.load(os.path.join(path, file)) # , file_format='pkl')
    overall.extend(info)
    print(len(info))
    
    for one in info:
        tokens.add(one['token'])
    print(len(tokens))

with open('data/', "w") as f:
    json.dump(overall, f)
