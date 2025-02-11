import torch, clip, numpy as np

objects = {    
    'object_1': 'red apple',
    'object_2': 'green block',
    'object_3': 'chocolate donut',
    'object_4': 'round bottomed flask',
    'object_5': 'yellow toy duck',
    'object_6': 'banana',
    'object_7': 'purple alarm clock',
    'object_8': 'cup',
    'object_9': 'blue water bottle',
    'object_10': 'light bulb',
    'object_11': 'wine glass',
    'object_12': 'copper bowl',
    'object_13': 'silver headphone',
    'object_14': 'hammer',
    'object_15': 'digital camera',
    'object_16': 'blue stapler',
    'object_17': 'white egg',
    'object_18': 'green toy train',
    'object_19': 'teapot',
    'object_20': 'red eyeglasses'
}

names = list(objects.values())
print(names)


device="cuda" if torch.cuda.is_available() else "cpu"
model,_=clip.load("ViT-B/32",device)
texts=names
tokens=clip.tokenize(texts).to(device)
with torch.no_grad():
 emb=model.encode_text(tokens).cpu().numpy()
np.save("embeddings.npy",emb)

import numpy as np
emb=np.load("embeddings.npy")
print(emb.shape)
for i in range(20):
    print(emb[i, :5])
