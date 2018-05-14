import os
import re
import codecs
import random

import numpy as np

from utils.util import create_dico, create_mapping, create_mask


def tag_mapping(tags):
	"""
	Create a dictionary a mapping of tags, sorted by frequency.
	Note that this tag mapping is for the sequence tags.
	So, if you use this as a single classification. 
	Input tag list should be wraped by additional list
	Ex) tags = [[0,1,2],[1,2],[22,3]] => tag_mapping(tags)
		tags = [0,1,2,3,4,5 ... ] => tag_mapping([tags])
	"""
	dico = create_dico(tags)
	tag_to_id, id_to_tag = create_mapping(dico)
	
	print("Found %i unique relation tags."%len(dico))

	return dico, tag_to_id, id_to_tag



def prepare_sentences(dataset,word_to_id, char_to_id, pos_to_id, ner_to_id, tag_to_id):
	"""	
	**kwargs should be mapper dictionary Ex) word_to_id, char_to_id ...
	"""
	words = []
	chars = []
	poss = []
	ners = []
	tags = []

	for d in dataset:
		words.append([word_to_id[token] if token in word_to_id else word_to_id['<UNK>'] for token in d['tokens']])
		poss.append([pos_to_id[pos] if pos in pos_to_id else pos_to_id['<UNK>'] for pos in d['pos']])
		ners.append([ner_to_id[ner] if ner in ner_to_id else ner_to_id['<UNK>'] for ner in d['ner']])
		
		sent_char = []
		for token in d['tokens']:
			word_char = []
			for c in token:
				if c in char_to_id:
					word_char.append(char_to_id[c])	
			sent_char.append(word_char)
		chars.append(sent_char)
		#chars.append([[char_to_id[c] if c in char_to_id for c in token] for token in d['tokens']])
		tags.append(tag_to_id[d['relation']])
	
	sentences = {'words': words, 'chars': chars, 'poss': poss, 'ners': ners, 'tags': tags}
	return sentences
	
def prepare_batch_datas(sentences, batch_size = 16, shuffle = True):
	"""
	Prepare input for one epoch.
	dataset is a ~ and it is made up of
	1) words,
	2) chars,
	2) ners,
	3) poss,
	4) depdency heads
	"""
	words = sentences['words']
	chars = sentences['chars']
	poss = sentences['poss']
	ners = sentences['ners']
	tags = sentences['tags']
	#dephs = dataset['dephs']
	#pos1 = dataset['pos1']
	#pos2 = dataset['pos2']
	
	# Make batch index
	num_of_sents = len(words)
	indices = random.shuffle(range(num_of_sents)) if shuffle else range(num_of_sents)
	batch_indices = [indices[i*batch_size:(i+1)*batch_size] for i in range(int(num_of_sents/batch_size + 1))]
	print(batch_indices)	
	# Padding and Masking for the batch level operation
	batchfied_words = []
	batchfied_chars = []
	batchfied_ners = []
	batchfied_poss = []
	batchfied_tags = []
	
	for batch_index in batch_indices:
		batch_words = [words[bi] for bi in batch_index]
		batch_chars = [chars[bi] for bi in batch_index]
		batch_poss = [poss[bi] for bi in batch_index]
		batch_ners = [ners[bi] for bi in batch_index]
		batch_tags = [tags[bi] for bi in batch_index]
		# Find the size of the maximum length sent and word
		max_words = 0
		max_chars = 0
		for sent_chars in batch_chars: 
			if len(sent_chars) > max_words:
				max_words = len(sent_chars)
			for word_char in sent_chars:
				if len(word_char) > max_chars:
					max_chars = len(word_char)
		
		batchfied_words += [np.array([sent_words + [0]*(max_words - len(sent_words)) for sent_words in batch_words])]
		batchfied_ners += [np.array([sent_ners + [0]*(max_words - len(sent_ners)) for sent_ners in batch_ners])]
		batchfied_poss += [np.array([sent_poss + [0]*(max_words - len(sent_poss)) for sent_poss in batch_poss])]
		
		# Make a padded input for the characters
		b_c = []
		for	sent_chars in batch_chars:
			padded_sent_chars = []
			for word_chars in sent_chars:
				padded_sent_chars.append(word_chars + [0] * (max_chars - len(word_chars)))
			padded_sent_chars += [[0]*max_chars for tt in range(max_words - len(sent_chars))]
			b_c.append(padded_sent_chars)
		batchfied_chars += [np.array(b_c)]

		batchfied_tags += [np.array(batch_tags)]
		

	batchfied_datas = {'words': batchfied_words,
					   'chars': batchfied_chars,
					   'ners': batchfied_ners,
					   'poss': batchfied_poss,
					   'tags': batchfied_tags}

	return batchfied_datas

