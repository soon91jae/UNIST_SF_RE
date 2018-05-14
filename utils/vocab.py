import os
import re
import codecs
import operator
import random

import numpy as np

import torch

from utils import util
from utils.util import create_mapping, create_dico
#from util import create_mapping, create_dico

random.seed(1234)
np.random.seed(1234)

class vocab(object):
	def calculate_oov(self, words, token_name = ""):
		word_set = create_dico(words).keys()
		word_vocab = self.token_vocab
		matched_words = set(word_set) & set(word_vocab)

		mw = len(matched_words)
		ws = len(word_set)
		oov = len(word_set) - len(matched_words)
		oov_rate = float(oov)/ws*100
		print("We matched %d %s out of %d %s. OOV rate is %.2f(%d/%d)"%(mw, token_name, ws, token_name, oov_rate, oov, ws))
		
		
		
	def create_emb(self, tokens, token_embed_size, token_freq_thres = 0, token_name = "word", ext_emb_path = ""):
		# Creating dictionary that counting num of tokens
		token_dico = create_dico(tokens)
		num_of_unfiltered_tokens = len(token_dico)
		
		# Adding PAD tokens and UNK tokens
		max_token_value = max(token_dico.items(), key=operator.itemgetter(1))[1]
		token_dico['<PAD>'] = max(max_token_value + 2, 1000000)
		token_dico['<UNK>'] = max(max_token_value + 1, 999999)
		
		# Remove the rare words by frequency
		token_dico = {k: v for k, v in token_dico.items() if v >= token_freq_thres}
		token_to_id, id_to_token = create_mapping(token_dico)
		
		
		# Initialize embedding vector (Random vector initialize / Pretrained vector initialize)
		emb_vec = np.random.uniform(low=-1.0, high=1.0, size=(len(token_dico), token_embed_size))
		n_tokens = len(token_dico)
		if not ext_emb_path:
			print('Generate random initialized embeddings for %s' % token_name)
		else:
			print('Loading pretrained %s embeddings from %s...' % (token_name, ext_emb_path))
			assert os.path.isfile(ext_emb_path)
			
			# Load pretrained embeddings from file
			pretrained = {}
			emb_invalid = 0
			for i, line in enumerate(codecs.open(ext_emb_path, 'r', 'utf-8')):
				line = line.strip().split()
				if len(line) == token_embed_size + 1:
					pretrained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
				else:
					emb_invalid += 1
			if emb_invalid > 0:
				print ('WARNING: %i invalid lines' % emb_invalid)
	
			c_found = 0
			c_lower = 0
			c_zeros = 0
			for i in range(len(emb_vec)):
				token = id_to_token[i]
				if token in pretrained:
					emb_vec[i] = pretrained[token]
					c_found += 1
				elif token.lower() in pretrained:
					emb_vec[i] = pretrained[token.lower()]
					c_lower += 1
				elif re.sub('\d', '0', token.lower()) in pretrained:
					emb_vec[i] = pretrained[re.sub('\d', '0', token.lower())]
					c_zeros += 1

			print ('Loaded %i pretrained embeddings.' % len(pretrained))
			print ('%i / %i (%.4f%%) words have been initialized with pretrained embeddings.'
					%(c_found + c_lower + c_zeros, n_tokens, 100. * (c_found + c_lower + c_zeros) / n_tokens))
			print ('%i found directly, %i after lowercasing, %i after lowercasing + zero.' 
					% (c_found, c_lower, c_zeros))
			
		# Initialize <PAD> token with 0 vector
		emb_vec[0] = np.zeros(token_embed_size)
		
		# Initialize <UNK> token with average vector of vocabulary
		emb_vec[1] = emb_vec[2: ].mean(axis=0)
		
		return emb_vec, token_to_id, id_to_token 
			
			
	def __init__(self, sentences, token_embed_size, token_freq_thres = 0, token_name = "", ext_emb_path = ""):
		"""
		sentences: list of the words
		emb_path: path for the word embedding
		"""
		self.ext_emb_path = ext_emb_path
		
		self.tokens = [[w for w in s] for s in sentences]
		#self.chars = ["".join(s) for s in sentences]
		
		
		
		
		self.token_emb, self.token_to_id, self.id_to_token = self.create_emb(self.tokens, 
																		token_embed_size,
																		token_freq_thres = token_freq_thres, 
																		token_name = token_name, 
																		ext_emb_path = ext_emb_path)
		self.token_vocab = self.token_to_id.keys()
		self.token_mapping = (self.token_to_id, self.id_to_token)
		self.calculate_oov(sentences, token_name)	
			
