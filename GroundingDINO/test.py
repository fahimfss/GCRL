from groundingdino.util.inference import load_model, load_image, predict, annotate
import time

model = load_model("/home/fahim/Projects/RLC/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "/home/fahim/Projects/RLC/GroundingDINO/asset/groundingdino_swinb_cogcoor.pth")
IMAGE_PATH = "asset/cat_dog.jpeg"
TEXT_PROMPT = "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH, [640, 480])
print(image.shape)
for i in range(30):
    t1 = time.time()
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    t2 = time.time()
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    t3 = time.time()
    
    print(f'{i}: {t2 - t1} \t {t3 - t2}')