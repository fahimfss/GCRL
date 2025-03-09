DETICDIR='/home/hany606/repos/rlc/RLC/Detic/'
import os
import sys
import time
import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import _create_text_labels, GenericMask

from detectron2.data.detection_utils import read_image

from detectron2.config import get_cfg

sys.path.insert(0, os.path.join(DETICDIR, 'third_party/CenterNet2/'))
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import get_clip_embeddings
from detic.modeling.utils import reset_cls_test

def debug(msg="debug", interrupt=True, postmortem=False, pdb=False):
    if interrupt:
        # import pdb
        import pudb

        spaces = min(len(msg) + 4, 30)
        print(f'{"#"*spaces}')
        # print(f"#{' '*(spaces//2)}debug{' '*(spaces//2)}#")
        print(f"# {msg} #")
        print(f'{"#"*spaces}')
        # print(f'{"#"*(spaces+5)}')
        # not working for now
        # if postmortem:
        #     import traceback
        #     pudb.post_mortem()
        # else:
        if pdb:
            pdb.set_trace()
        else:
            breakpoint()
            


class Env:
    def __init__(self):
        self.objects = {
            'object_1': 'red apple',
            'object_2': 'green block',
            'object_3': 'chocolate donut',
            'object_4': 'round bottomed flask',
            'object_5': 'yellow toy duck',
            'object_6': 'banana',
            'object_7': 'purple alarm clock',
            'object_8': 'pink cup',
            'object_9': 'water bottle',
            'object_10': 'light bulb',
            'object_11': 'wine glass',
            'object_12': 'copper bowl',
            'object_13': 'silver headphone',
            'object_14': 'hammer',
            'object_15': 'digital camera',
            'object_16': 'blue stapler',
            'object_17': 'white egg',
            'object_18': 'toy train',
            'object_19': 'teapot',
            'object_20': 'eyeglasses',
        }

        vocabs = [v for v in self.objects.values()]
        self.classifier_model = Detic(vocabs)
        
        
    def step(self):
        # load the image from path
        from PIL import Image
        image = np.asarray(Image.open('/home/hany606/repos/rlc/RLC/Detic/test_obj.png'))[:,:,:3]
        image = image[:,:,::-1]
        for i in range(1, 21):
            text = self.objects[f'object_{i}']
            print(f"classify for {text}")
            xyxy, t = detic_inference(image, self.classifier_model, text)
            print(f"time taken for prediction {t}")
            # draw on image
            mask = np.zeros_like(image)
            x1,y1,x2,y2 = xyxy
            mask[y1:y2, x1:x2] = np.ones_like(mask[y1:y2, x1:x2])*255
            image_mask = np.concatenate([image[:,:,::-1], mask], axis=1)
            im = Image.fromarray(image_mask)
            im.save(f"demos/image_mask_{i}_{text}.jpeg")

class Detic:
    clip_path=os.path.join(DETICDIR, 'models/ViT-B-32.pt')
    config_file=os.path.join(DETICDIR, 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml')
    opts=['MODEL.WEIGHTS', os.path.join(DETICDIR, 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth')]
    confidence_threshold=0.5
    def __init__(self, vocabs):
        self.metadata = MetadataCatalog.get("__unused")
        self.metadata.thing_classes = vocabs
        classifier = get_clip_embeddings(self.metadata.thing_classes, clip_path=self.clip_path)

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")

        self.predictor = self.init()
        reset_cls_test(self.predictor.model, classifier, num_classes)
        
        self._caption_cls = {v:i for i,v in enumerate(vocabs)}
        self.get_caption_cls = lambda c: self._caption_cls[c]


        self._cls_caption_ = {i:v for i,v in enumerate(vocabs)}
        self.get_cls_caption = lambda c: self._cls_caption_[c]

    def init(self):
        cfg = get_cfg()
        # cfg.MODEL.DEVICE="cpu"
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(self.config_file)
        cfg.merge_from_list(self.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.confidence_threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.freeze()
        # if self.parallel:
        #     num_gpu = torch.cuda.device_count()
        #     predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        # else:
        #     predictor = DefaultPredictor(cfg)
        predictor = DefaultPredictor(cfg)
        return predictor

    def predict(self, image):
        return self.predictor(image)
    
    
    
def detic_inference(image, model, caption):
    t1 = time.time()
    width, height = image.shape[1], image.shape[0]
    _predictions = model.predict(image)
    predictions = _predictions['instances'].to(model.cpu_device)
    idx = None
    caption_cls = model.get_caption_cls(caption)
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    # Get the one with the maximum score
    if classes is not None:
        # cls = classes.tolist()
        # print([model.get_cls_caption(c) for c in cls])
        indices = torch.where(classes==caption_cls)[0]
        if indices.shape[0] > 0:
            idx = indices[scores[indices].argmax().item()].item()
    # debug()
    # keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
    # if predictions.has("pred_masks"):
    #     masks = np.asarray(predictions.pred_masks)
    #     # masks = [GenericMask(x, height, width) for x in masks]
    #     bbox = [GenericMask(x, height, width).bbox() for x in masks]
    
    if idx is not None:
        print("Found one")
        xyxy = boxes[idx].tensor.int().tolist()[0]
    else:
        print("Did not find anything")
        xyxy = [0,0,0,0]
    t2 = time.time()
    print("---------------")
    return xyxy, t2-t1

if __name__ == '__main__':
    env = Env()
    
    env.step()