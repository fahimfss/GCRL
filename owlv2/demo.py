import os
OWLV2DIR = os.path.dirname(os.path.abspath(__file__))
print(OWLV2DIR)
from transformers import pipeline
import numpy as np
from PIL import Image
import time
from PIL import ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


vocabs = ['red apple', 'green block', 'chocolate donut', 'round bottomed flask', 'yellow toy duck', 'banana', 'purple alarm clock', 'pink cup', 'water bottle', 'light bulb', 'wine glass', 'copper bowl', 'silver headphone', 'hammer', 'digital camera', 'blue stapler', 'white egg', 'toy train', 'teapot', 'eyeglasses']


# detector = pipeline(model='/home/hany606/repos/rlc/RLC/owlv2/models/owlv2-base-patch16-ensemble', task="zero-shot-object-detection")


image = Image.open(os.path.join(OWLV2DIR, 'test_obj.png')).convert("RGB")

# predictions = detector(
#     image,
#     candidate_labels=vocabs,
# )

# print(predictions)



# draw = ImageDraw.Draw(image)

# for prediction in predictions:
#     box = prediction["box"]
#     label = prediction["label"]
#     score = prediction["score"]

#     xmin, ymin, xmax, ymax = box.values()
#     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
#     draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

# image.save("output.png")

# class OwlV2:
#     model_path=os.path.join(OWLV2DIR, 'models/owlv2-base-patch16-ensemble')
#     def __init__(self, vocabs):
#         self.vocabs = vocabs
#         self.predictor = pipeline(model=self.model_path, task="zero-shot-object-detection")

#         self._cls_caption_ = {i:v for i,v in enumerate(vocabs)}
#         self.get_cls_caption = lambda c: self._cls_caption_[c]

#     def predict(self, image):
#         return self.predictor(image, self.vocabs)
    
    
    
# def owlv2_inference(image, model, caption):
#     # https://huggingface.co/docs/transformers/en/tasks/zero_shot_object_detection
#     # TODO: refactor and make it better
#     t1 = time.time()
#     predictions = model.predict(image) # list of dictionaries
#     print(predictions)
#     idx = None
#     mx_score = -1
#     for i, p in enumerate(predictions):
#         if p['label'] == caption:
#             print(p)
#         if p['label'] == caption and p['score'] > mx_score:
#             idx = i
#             mx_score = p['score']
            
#     #   'box': {'xmin': 277, 'ymin': 338, 'xmax': 327, 'ymax': 380}},
#     if idx is not None:
#         print("Found one")
#         print(predictions[idx])
#         box = predictions[idx]['box']
#         xyxy = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
#     else:
#         print("Did not find anything")
#         xyxy = [0,0,0,0]
#     t2 = time.time()
#     return np.array(xyxy), t2-t1


# m = OwlV2(vocabs)

# owlv2_inference(image, m, 'hammer')


# No need to specify the vocab
class OwlV2:
    model_path=os.path.join(OWLV2DIR, 'models/owlv2-base-patch16-ensemble')
    def __init__(self, vocabs):
        print(f"Load OwlV2 classifier from {self.model_path}")
        # self.vocabs = vocabs
        # self.predictor = pipeline(model=self.model_path, task="zero-shot-object-detection")
        self.predictor = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        # self._cls_caption_ = {i:v for i,v in enumerate(vocabs)}
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
    print(scores, labels)
    if len(scores) > 0:
        print("Found one")

        mx_idx = results["scores"].argmax().item()
        print(mx_idx)
        xmin, ymin, xmax, ymax = boxes[mx_idx]
        xyxy = [xmin, ymin, xmax, ymax]
    else:
        print("Did not find anything")
        xyxy = [0,0,0,0]
    t2 = time.time()
    return np.array(xyxy), t2-t1



m = OwlV2(vocabs)

owlv2_inference(image, m, 'hammer')
