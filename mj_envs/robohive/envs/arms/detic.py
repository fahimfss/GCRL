# DETICDIR='/home/hany606/repos/rlc/RLC/Detic/'
import os
DETICDIR = os.path.dirname(os.path.abspath(__file__))
DETICDIR = os.path.join(DETICDIR, '../../../../Detic')
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
        # print("Found one")
        xyxy = boxes[idx].tensor.int().tolist()[0]
    else:
        # print("Did not find anything")
        xyxy = [0,0,0,0]
    t2 = time.time()
    return np.array(xyxy), t2-t1