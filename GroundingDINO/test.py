from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import random
import torch
import time
import math
from torchvision.ops import box_convert
import json
import os
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

model = load_model("/home/fahim/Projects/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/fahim/Projects/GroundingDINO/asset/groundingdino_swint_ogc.pth")

BOX_TRESHOLD = 0.40
TEXT_TRESHOLD = 0.25
 
def point_in_rectangle(rect_tl, rect_br, point):
    x1, y1 = rect_tl
    x2, y2 = rect_br
    px, py = point

    inside = x1 <= px <= x2 and y1 <= py <= y2
    if inside:
        return True, 0

    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    distance = math.dist(center, point)
    return False, distance

def get_predict(model, image, prompt):
    with torch.no_grad():
        t1 = time.time()
        out = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        t2 = time.time()
    return out, t2-t1

def read_file_to_dicts(file_path):
     
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
def point_in_rectangle(rect_tl, rect_br, point):
    x1, y1 = rect_tl
    x2, y2 = rect_br
    px, py = point

    inside = x1 <= px <= x2 and y1 <= py <= y2
    if inside:
        return 1.0, 0

    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    distance = math.dist(center, point)
    return 0.0, distance

path = "/home/fahim/Projects/jsac_rlc/RL-Chemist/GroundingDINO/inference_results/images_3"
info_path = "/home/fahim/Projects/jsac_rlc/RL-Chemist/GroundingDINO/inference_test_data/info_3.txt"

def inference(object_name, seed):
    random.seed(seed)
    out_path = path + "/" + object_name
    os.makedirs(out_path, exist_ok=False)
    log_path = out_path + "/log.txt"
    log = open(log_path, "w")
    
    all_data = read_file_to_dicts(info_path)
    all_object_data = [line for line in all_data if line['prompt'] == object_name]
    
    total = 0
    image_index = 0
    ret = []
    
    for i in range(10):
        x_min = i * 80
        x_max = (i+1) * 80
        range_data = [line for line in all_object_data if line['x'] >= x_min and line['x'] < x_max]
        range_data.sort(key=lambda x: x['x'])
        
        indices = random.sample(range(len(range_data)), 5)
        for index in indices:
            img_path = range_data[index]['img_path']
            x = range_data[index]['x']
            y = range_data[index]['y'] 
            image_source, image = load_image(img_path)
            out, time = get_predict(model, image, object_name)
            boxes, logits, phrases = out    
            
            if logits.shape[0] > 0:
                mx = torch.argmax(logits).item() 
                logits = logits[mx].unsqueeze(0)
                boxes = boxes[mx, :].unsqueeze(0)
            
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            itr_str = str(image_index).zfill(3)
            image_index += 1
            image_path = out_path + "/" + itr_str + ".png"
            cv2.imwrite(image_path, annotated_frame)
            
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            
            if xyxy.size == 0: 
                ret.append(f'Image: {itr_str} needs manual checking\n')
                flag = 0.0
                dist = -1.0
            else:
                rect_tl = xyxy[0][0], xyxy[0][1]
                rect_br = xyxy[0][2], xyxy[0][3]
                point = (800-x, y) 
                
                flag, dist = point_in_rectangle(rect_tl, rect_br, point)
                
                if flag > 0.5:
                    total += 1
                else:
                    ret.append(f'Image: {itr_str} needs manual checking. Dist: {dist}\n')
            
            log_txt = '{' + f'"Index": "{itr_str}", "flag": {flag}, "dist": {dist}, "time": {time}' + '}\n'
            log.write(log_txt)
            log.flush()
            
    ret.append(f'{object_name}  --  total: {total}. success rate: {total/50.0}\n')
    
    log.close() 
    
    return ret
    
    
target_names = ['banana', 'alarm clock', 'cup', 'beaker', 'apple', 'block', 'donut', 'rubber duck']
seed = 42
i = 1
result_fl = open(path + "/result.txt", "w")

for name in target_names:
    print(name)
    result_fl.write(name + "\n")
    rets = inference(name, seed * i)
    for ret in rets:
        result_fl.write(ret)
    result_fl.write("\n\n")
    i += 1
    
result_fl.close()

# image_source, image = load_image(IMAGE_PATH)
# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("/home/fahim/Projects/jsac_rlc/RL-Chemist/GroundingDINO/annotated_image.jpg", annotated_frame)
