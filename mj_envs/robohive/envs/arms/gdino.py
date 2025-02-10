import torch
from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T 
from groundingdino.util.inference import load_model, predict
import cv2 as cv
import time
import numpy as np
import multiprocessing as mp
from PIL import Image

BOX_THRESHOLD = 0.40
TEXT_THRESHOLD = 0.25

def load_image(image_source, image_size):
    transform = T.Compose(
        [
            T.ResizeDebug(image_size),
            # T.RandomResize(image_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image_source, None)
    return image_transformed

def create_mask(image_source, boxes) -> np.ndarray:
    h, w = image_source.shape
    boxes = torch.tensor(boxes, dtype=torch.float32) * torch.Tensor([w, h, w, h])

    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (-2000, -2000)
        
    if xyxy.size != 0:
        top_left = (int(xyxy[0]), int(xyxy[1]))
        bottom_right = (int(xyxy[2]), int(xyxy[3]))
        center = (int((top_left[0] + bottom_right[0]) / 2), int((top_left[1] + bottom_right[1]) / 2))
        cv.rectangle(mask, top_left, bottom_right, (255), thickness=-1)  # Fill the rectangle
        white_pixels = np.argwhere(mask == 255)
    
        centroid = np.mean(white_pixels, axis=0).astype(int)  # Returns (y, x)
        centroid = (centroid[1], centroid[0])

    return mask, center

def g_dino_inference(image, model, caption, height, width):
    pil_image = Image.fromarray(image)
    t1 = time.time()
    boxes, logits, phrases = predict(
        model=model,
        image=load_image(pil_image, [800, 600]),
        # image=load_image(pil_image, [width, height]),
        caption=caption,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    t2 = time.time() 
    if logits.nelement() > 0:
        _, indices = torch.max(logits, dim = 0)
        boxes = boxes.numpy()
        boxes = boxes[indices] 
    return boxes, t2-t1
    
def async_g_dino_inference(img_shape, mem_name, image_queue, mask_queue):
    model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
                       "../GroundingDINO/asset/groundingdino_swinb_cogcoor.pth")
    count = 0
    
    img_shm = mp.shared_memory.SharedMemory(name=mem_name) 
    img = np.ndarray(img_shape, dtype=np.uint8, buffer=img_shm.buf)
    h, w, c = img_shape 
    
    while True:
        data = image_queue.get() 
        if data == 'close':
            img_shm.close()
            return 
        target_name = data  
        count += 1 
        
        boxes, inference_time = g_dino_inference(img.copy(), model, target_name, h, w)
        mask_queue.put((boxes, inference_time, count))