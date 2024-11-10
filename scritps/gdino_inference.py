from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import time
import math
from torchvision.ops import box_convert
import json
import os
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
# "../GroundingDINO/asset/groundingdino_swint_ogc.pth")

model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
"../GroundingDINO/asset/groundingdino_swinb_cogcoor.pth")

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

    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    distance = math.dist(center, point)
    inside = x1 <= px <= x2 and y1 <= py <= y2

    if inside:
        return 1.0, distance

    return 0.0, distance


def inference(object_name, out_path, inference_image_size, info_path, distance_threshold):
    all_data = read_file_to_dicts(info_path)
    all_object_data = [line for line in all_data if line['prompt'] == object_name]
    
    image_index = len(os.listdir(out_path))
    
    total = len(all_object_data)
    correct = 0
    inference_time = 0
    errors = []
    infos = []
    
    for index in range(len(all_object_data)):        
        img_path = all_object_data[index]['img_path']
        x = all_object_data[index]['x']
        y = all_object_data[index]['y'] 
        arm_distance = all_object_data[index]['distance'] 
        
        image_source, image = load_image(img_path, inference_image_size)
        if object_name == 'rubber duck':
            o_name = 'yellow toy duck'
        elif object_name == 'beaker':
            o_name = 'round glass beaker'
        elif object_name == 'alarm clock':
            o_name = 'purple clock'
        else:
            o_name = object_name
        out, tm = get_predict(model, image, o_name)
        inference_time += tm
        boxes, logits, phrases = out    
        
        if logits.shape[0] > 0:
            mx = torch.argmax(logits).item() 
            logits = logits[mx].unsqueeze(0)
            boxes = boxes[mx, :].unsqueeze(0)
        
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        itr_str = str(image_index).zfill(3)
        image_index += 1
        image_path = out_path + "/" + itr_str + ".png"
        cv2.imwrite(image_path, cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1))
        
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        if xyxy.size == 0: 
            errors.append((image_path, "No box")) 
            flag = 0.0
            dist = -1.0
        else:
            rect_tl = xyxy[0][0], xyxy[0][1]
            rect_br = xyxy[0][2], xyxy[0][3]
            point = (x, y) 
            
            flag, dist = point_in_rectangle(rect_tl, rect_br, point) 
            
            if dist < distance_threshold:
                correct += 1
            else: 
                errors.append((image_path, "Incorrect prediction", dist)) 
                
        infos.append(f'"Image path": "{image_path}", "flag": {flag}, "dist": {dist}, "time": {tm}, "arm_distance": {arm_distance}')
        
    inference_time = inference_time / total
    
    return correct, total, inference_time, errors, infos
    
    

if __name__ == '__main__':
    
    info_paths = ["inference_test_data_1", 
                  "inference_test_data_2"]
    
    inference_image_sizes = [[848, 480], [636, 360], [424, 240]]
    distance_thresholds = [100.0, 75.0, 50.0]
    base_path = "inference_results/"
    if not os.path.exists(base_path): 
        os.makedirs(base_path)
    target_names = ['apple',    'green block', 'donut',    'beaker',   'rubber duck', 'banana',   'alarm clock', 'cup']
    
    results_fl = open(f'inference_results/results.txt', 'w')
    errors_fl = open(f'inference_results/errors.txt', 'w')
    logs_fl = open(f'inference_results/logs.txt', 'w')
    
    for i in range(1, 4, 2):
        total_imgs = [[0] * 3 for _ in range(8)]
        correct_imgs = [[0] * 3 for _ in range(8)]
        errors = {}
        infos = []

        for j in range(3):
            image_w, image_h = inference_image_sizes[j]
            folder_path = f'{base_path}/inference_result_{image_w}_{image_h}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            sub_folder_path = f'{folder_path}/images_{i}'
            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path)
            
            
            for ti, target in enumerate(target_names):
                sub_sub_folder_path = f'{sub_folder_path}/{target}'
                if not os.path.exists(sub_sub_folder_path):
                    os.makedirs(sub_sub_folder_path)
            
                for k in range(2): 
                    info_path = info_paths[k] + f'/info_{i}.txt'
                    
                    correct, total, inference_time, errors_ret, infos_ret = inference(target, sub_sub_folder_path, inference_image_sizes[j], info_path, distance_thresholds[j])
                    
                    total_imgs[ti][j] += total
                    correct_imgs[ti][j] += correct
                    
                    if len(errors_ret) > 0: 
                        idx = f'{ti}_{j}'
                        if idx not in errors:
                            errors[idx] = []
                        for error in errors_ret:
                            errors[idx].append(error)
                    for info in infos_ret:
                        infos.append('{' + info + f', "num_objects": {i}, "resolution": "{image_w}_{image_h}"' + '}\n')
                                            
                    print(target, k, inference_time) 
                    
        #  f"{float_number:.3f}"
        results_fl.write(f'Num objects: {i}\n') 
        for ti in range(9):
            st = ''
            for j in range(4):
                if ti == 0:
                    if j == 0:
                        st += " ".ljust(15) + '\t'
                    else:
                        image_w, image_h = inference_image_sizes[j-1] 
                        st += f'\t{image_w}_{image_h}'
                else:
                    if j == 0:
                        st += target_names[ti-1].ljust(15) + '\t'
                    else:
                        v1 = correct_imgs[ti-1][j-1]
                        v2 = total_imgs[ti-1][j-1]
                        st += f'\t{v1}/{v2}'
            results_fl.write(st + '\n') 
            
        
        results_fl.write(f'\n\n') 
        for ti in range(9):
            st = ''
            for j in range(4):
                if ti == 0:
                    if j == 0:
                        st += " ".ljust(15) + '\t'
                    else:
                        image_w, image_h = inference_image_sizes[j-1] 
                        st += f'\t{image_w}_{image_h}'
                else:
                    if j == 0:
                        st += target_names[ti-1].ljust(15) + '\t'
                    else:
                        v1 = correct_imgs[ti-1][j-1]
                        v2 = total_imgs[ti-1][j-1]
                        if v2 > 0:
                            perc = float(v1)/float(v2)
                        else:
                            perc = 0
                        st += f"\t{perc:.3f}"
            results_fl.write(st + '\n') 
        results_fl.write(f'\n\n') 
        
        errors_fl.write(f'Num objects: {i}\n') 
        for ti in range(8): 
            for j in range(3): 
                idx = f'{ti}_{j}'
                if idx in errors:
                    image_w, image_h = inference_image_sizes[j] 
                    errors_fl.write('\n' + target_names[ti] + f'  {image_w}_{image_h} - {j}' + '\n') 
                    for error in errors[idx]:
                        if len(error) == 3:
                            dist = error[2]
                            st = error[0] + ",   " + error[1] + ",   " + f"\tDist: {dist:.3f}"
                        else:
                            st = error[0] + ",   " + error[1]
                        
                        errors_fl.write(st + '\n')
                    
        errors_fl.write(f'\n\n')
                    
        for info in infos:
            logs_fl.write(info)
            
    results_fl.close()
    errors_fl.close()
    logs_fl.close()
        


# image_source, image = load_image(IMAGE_PATH)
# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("/home/fahim/Projects/jsac_rlc/RL-Chemist/GroundingDINO/annotated_image.jpg", annotated_frame)
