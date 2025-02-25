import os
OWLV2DIR = os.path.dirname(os.path.abspath(__file__))
OWLV2DIR = os.path.join(OWLV2DIR, '../../../../owlv2')
import sys
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import time
import numpy as np
from PIL import Image
import torch

class OwlV2:
    model_path=os.path.join(OWLV2DIR, 'models/owlv2-base-patch16-ensemble')
    def __init__(self, vocabs):
        print(f"Load OwlV2 classifier from {self.model_path}")
        self.vocabs = vocabs
        # self.predictor = pipeline(model=self.model_path, task="zero-shot-object-detection")
        self.predictor = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self._cls_caption_ = {i:v for i,v in enumerate(vocabs)}
        self.get_cls_caption = lambda c: self._cls_caption_[c]

    def predict(self, image, caption):
        # return self.predictor(image, candidate_labels=self.vocabs)
        inputs = self.processor(text=[caption], images=image, return_tensors="pt")
        outputs = self.predictor(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
        return results
def owlv2_inference(_image, model, caption):
    # https://huggingface.co/docs/transformers/en/tasks/zero_shot_object_detection
    image = Image.fromarray(_image) if isinstance(_image, np.ndarray) else _image
    t1 = time.time()
    with torch.no_grad():
        results = model.predict(image, caption)
    scores = results["scores"].tolist()
    labels = results["labels"].tolist()
    boxes = results["boxes"].tolist()
    
    if len(scores) > 0:
        mx_idx = results["scores"].argmax().item()
        xmin, ymin, xmax, ymax = boxes[mx_idx]
        xyxy = [xmin, ymin, xmax, ymax]
    else:
        # print("Did not find anything")
        xyxy = [0,0,0,0]
    t2 = time.time()
    return np.array(xyxy), t2-t1

# def owlv2_inference(_image, model, caption):
#     # https://huggingface.co/docs/transformers/en/tasks/zero_shot_object_detection
#     # TODO: refactor and make it better
#     image = Image.fromarray(_image) if isinstance(_image, np.ndarray) else _image
#     t1 = time.time()
#     predictions = model.predict(image, caption) # list of dictionaries
#     idx = None
#     mx_score = -1
#     for i, p in enumerate(predictions):
#         if p['label'] == caption and p['score'] > mx_score:
#             idx = i
#             mx_score = p['score']
            
#     if idx is not None:
#         # print("Found one")
#         box = predictions[idx]['box']
#         xyxy = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
#     else:
#         # print("Did not find anything")
#         xyxy = [0,0,0,0]
#     t2 = time.time()
#     return np.array(xyxy), t2-t1