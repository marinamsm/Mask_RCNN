import os
import sys
import json
import numpy as np
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from pycocotools.coco import COCO

coco = COCO('coco/annotations/instances_train2014.json')
IMAGE_PATH = "coco/images_train_2014/train2014"
LAST_IMG_NUM = 41392

def load_COCO_imgs():
	cats = coco.loadCats(coco.getCatIds())
	nms = [cat['name'] for cat in cats]
	class_names = nms
	catIds = coco.getCatIds(catNms=class_names)
	img_ids = coco.getImgIds()
	return coco.loadImgs(img_ids)

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	imgs = load_COCO_imgs()
	total = len(imgs)
	dataset = []
	imgsIds = []
	i = 0
	batch_size = 8
	N = total/2
	while i < N:
		#if i % 1024 == 0:
		#	print('{} from {} images.'.format(i, N))
		batch = imgs[i:i + batch_size]
		i += batch_size
		# get the images identifiers
		imagesNames = [img['file_name'].split('.')[0] for img in batch]
		imagesIds = [img['id'] for img in batch]
		for ind in range(len(imagesNames)):
			dataset.append(imagesNames[ind])
			imgsIds.append(imagesIds[ind])
	return set(dataset), set(imgsIds)

def concat_features_to_bboxes(features, bboxes):	
	results = []
	for ind in range(len(bboxes)):
		results.append(np.append(features[ind], bboxes[ind].flatten()))
	return results
	
# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if int(image_id) in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
	#The parameter 'photo' contains the characteristics vector and photo's detections
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for ind in range(len(imgsIds)):
			# retrieve the photo feature
			photo = photos[ind]
			id = next(iter(imgsIds))
			desc_list = descriptions[str(id)]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]

# define the captioning model
def define_model(vocab_size, max_length):
	# 4632 features + detections
	inputs1 = Input(shape=(4632,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	#model.summary()
	# plot_model(model, to_file='model.png', show_shapes=True)
	return model

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None	
	
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = np(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text
	
# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	for ind in range(len(imgsIds)):
			# generate description
			yhat = generate_desc(model, tokenizer, photos[ind], max_length)
			id = next(iter(imgsIds))
			desc_list = descriptions[str(id)]
			# store actual and predicted
			references = [d.split() for d in desc_list]
			actual.append(references)
			predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# load training dataset (+/- 41K)
imgsNames, imgsIds = load_set(IMAGE_PATH)
print('Dataset: %d' % len(imgsIds))
# descriptions
descriptions = load_clean_descriptions('COCO_descriptions_train2014.txt', imgsIds)
print('Descriptions: %d' % len(descriptions))
# photo features (extracted by NasNet)
features = load(open('saidaDescritoresNasnet/featuresCocoTrain2014.pkl', 'rb'))
print('Features: %d' % len(features))
# photo bboxes (generated by Mask R-CNN)
bboxes = load(open('saidaMaskRCNN/coco_evaluation_mrcnn_train2014.pkl', 'rb'))
print('BBoxes: %d' % len(bboxes))
#concat features and bboxes into 1D array
rnn_input = concat_features_to_bboxes(features, bboxes)
print('Input RNN: %d' % len(rnn_input))
# prepare tokenizer
tokenizer = create_tokenizer(descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(descriptions)
print('Description Length: %d' % max_length)
# load the model to evaluate it
filename = 'model_11.h5'
model = load_model(filename)
# evaluate model
evaluate_model(model, descriptions, rnn_input, tokenizer, max_length)
