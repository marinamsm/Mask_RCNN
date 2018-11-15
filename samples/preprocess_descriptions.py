import string
import json

# extract descriptions for images
def load_descriptions(filename):
	with open(filename, encoding='utf-8-sig') as json_file:
		json_data = json.load(json_file)
	descriptions = dict()
	anns = json_data.get("annotations", "none")
	for obj in anns:
		# split caption by white space
		tokens = obj.get("caption").split()
		if len(obj.get("caption")) < 2:
			continue
		# split id from description
		image_id = obj.get("image_id", "none")
		image_desc = tokens
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create list
		if image_id not in descriptions:
			descriptions[image_id] = list()
		# store description
		descriptions[image_id].append(image_desc)
	return descriptions

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(str(key) + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

filename = 'coco/annotations/captions_val2014.json'
# parse descriptions
descriptions = load_descriptions(filename)
print('Loaded descriptions: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(descriptions, 'COCO_descriptions_val2014.txt')