''' We can call this function to prepare the photo data for testing our models, then save the resulting dictionary to a file named ‘features.pkl‘. '''
import os
import numpy as np
from pickle import dump
from keras.applications.nasnet import NASNetLarge
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.nasnet import preprocess_input
from keras.models import Model
from pycocotools.coco import COCO

COCO = COCO('coco/annotations/instances_train2014')
IMAGE_PATH = 'coco/images_train_2014/train2014'


# extract features from each photo in the directory
def extract_features():
    # carrega o modelo NASNetLarge do Keras pré-treinado com os pesos da base ImageNet
    model = NASNetLarge(input_shape=(331, 331, 3), include_top=True,
                        weights='imagenet', input_tensor=None, pooling=None)
    # retira a última camada para obter os descritores da penúltima camada
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # características do modelo
    # model.summary()

    img_ids = COCO.getImgIds()
    imgs = COCO.loadImgs(img_ids)
    total = len(imgs)
    print(total)
    results = []
    i = 0
    batch_size = 16
    N = np.ceil(total / 2)
    while i < N:
        if i % 1024 == 0:
            print('{} from {} images.'.format(i, N))
        if N - i < batch_size:
            batch_size = N - i
        batch = imgs[i:i + batch_size]
        i += batch_size
        images = [
            load_img(
                os.path.join(IMAGE_PATH, img['file_name']),
                target_size=(331, 331)
            )
            for img in batch
        ]
        images = [preprocess_input(img_to_array(img)) for img in images]
        images = np.stack(images)
        r = model.predict(images)
        for ind in range(batch_size):
            results.append(r[ind])
    return results

# extrai o vetor de características das imagens
features = extract_features()
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('featuresCocoTrain2014.pkl', 'wb'))
