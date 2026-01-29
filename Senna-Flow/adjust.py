import copy
import json
import os

from tqdm import tqdm
from llava.model.builder import load_pretrained_model, load_senna_pretrained_model
from llava.mm_utils import get_model_name_from_path

# from data_tools.senna_qa_utils import eval_multi_img_model_wo_init

def generate():
    eval_data_path = './Senna/infos/senna_nusc_train.json'

    with open(eval_data_path, 'r') as file:
        eval_data = json.load(file)

    tot_num, correct_num = 0, 0

    SPEED_PLAN = ['KEEP', 'ACCELERATE', 'DECELERATE', 'STOP']
    PATH_PLAN = ['RIGHT_TURN', 'RIGHT_CHANGE', 'LEFT_TURN', 'LEFT_CHANGE', 'STRAIGHT']

    final = []
    cache = []
    last_token = None
    selected = False

    cnt_keep_straight = 0
    cnt_stop_straight = 0

    select_num = 0
    all_num = 0

    for idx, sample in tqdm(enumerate(eval_data)):
        # print(sample['token'] == last_token, selected)
        # if idx > 500:
        #     raise EOFError
        if sample['token'] == last_token:
            cache.append(sample)
            if selected:
                continue
        else:
            if selected:
                # print(len(cache))
                final.extend(copy.deepcopy(cache))
            cache = []
            selected = False
        last_token = sample['token']


        img_path = sample['image']
        question = sample['conversations'][0]['value']

        if 'SPEED plan' in question:
            all_num += 1
            # args = type('Args', (), {
            #     "model_path": model_path,
            #     "model_base": None,
            #     "query": question,
            #     "conv_mode": 'llava_v1',
            #     "image_file": sample['images'],
            #     "sep": ",",
            #     "temperature": 0,
            #     "top_p": None,
            #     "num_beams": 1,
            #     "max_new_tokens": 512
            # })()

            tot_num = tot_num + 1
            gt_answer = sample['conversations'][1]['value']
            # answer = eval_multi_img_model_wo_init(args, tokenizer, model, image_processor)

            speed_plan, path_plan = gt_answer.split(', ')
            path_plan = path_plan.split('\n')[0]

            if 'KEEP' in speed_plan and 'STRAIGHT' in path_plan and cnt_keep_straight < 200:
                cnt_keep_straight += 1
                selected = True
            elif 'STOP' in speed_plan and 'STRAIGHT' in path_plan and cnt_stop_straight < 180:
                cnt_stop_straight += 1
                selected = True
            elif speed_plan in ['ACCELERATE', 'DECELERATE'] or path_plan in ['RIGHT_TURN', 'RIGHT_CHANGE', 'LEFT_TURN', 'LEFT_CHANGE']:
                selected = True
            if selected:
                select_num += 1
            # print(selected, speed_plan, path_plan)

    print(len(final), cnt_keep_straight, cnt_stop_straight, select_num, all_num)

    with open('./Senna/infos/senna_nusc_select_2.json', "w") as f:
        json.dump(final, f)

def combine():
    path1 = './Senna/infos/senna_nusc_select.json'
    with open(path1, 'r') as file:
        d1 = json.load(file)
    
    path2 = './Senna/infos/senna_nusc_select_2.json'
    with open(path2, 'r') as file:
        d2 = json.load(file)
    
    d1.extend(d2)
    with open('./Senna/infos/senna_nusc_select_all.json', "w") as f:
        json.dump(d1, f)

# generate()
combine()
