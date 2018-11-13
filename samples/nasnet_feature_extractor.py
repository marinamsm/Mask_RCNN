''' We can call this function to prepare the photo data for testing our models, then save the resulting dictionary to a file named ‘features.pkl‘. '''
from os import listdir
from pickle import dump
from keras.applications.nasnet import NASNetLarge
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.nasnet import preprocess_input
from keras.models import Model

# extract features from each photo in the directory
def extract_features(directory):
	# carrega o modelo NASNetLarge do Keras pré-treinado com os pesos da base ImageNet
	#model = NASNetLarge(input_shape=(224, 224, 3), include_top=True, weights='imagenet', input_tensor=None, pooling=None)
	# retira a última camada para obter os descritores da penúltima camada
	#model.layers.pop()
	#model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# características do modelo
	#print(model.summary())
	features = dict()
	i = 0
	file = open("imageNames2.txt","w") 
	for name in listdir(directory):
		if i % 100 == 0:
			print('{} from {} images.'.format(i, len(listdir(directory))))
		if i > len(listdir(directory))/2: break
		#filename = directory + '/' + name
		#redimensiona a imagem para que seja 224x224 por causa da NASNet
		#image = load_img(filename, target_size=(224, 224))
		#image = img_to_array(image)
		#image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepara a imagem para o modelo NASNet
		#image = preprocess_input(image)
		# extrai as características
		#feature = model.predict(image, verbose=0)
		#image_id = name.split('.')[0]
		print(i)
		file.write(name.split('.')[0])
		# armazena as características da imagem
		#features[image_id] = feature
		i += 1
		#print('>%s' % name)
	print(i)
	file.close()
	return features

# extrai o vetor de características das imagens
directory = 'coco/images_train_2014/train2014' #pasta com as imagens da  base
features = extract_features(directory)
#print('Extracted Features: %d' % len(features))
# save to file
#dump(features, open('featuresCocoTrain2014.pkl', 'wb'))
