import os
import re
import codecs
import torch

import numpy as np

def get_name(parameters):
	"""
	generate a model name from its parameters
	"""
	l = []
	for k, v in parameters.items():
		if type(v) is str and "/" in v:
			l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
		else:
			l.append((k, v))
	name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
	mode_name = "".join(i for i in name if i not in "\/:*?<>|")
	print(model_name)
	
	return model_name

def token_mapping(self, tokens):
	"""
	Create a dictionary and a mapping of tokens
	"""
	token_dico = create_dico(tokens)

	token_dico = {k: v for k, v in token_dico.items()}	
	token_to_id, id_to_token = create_mapping(token_dico)
	
	return 	token_dico, token_to_id, id_to_token
	

	
def create_mapping(dico):
	"""
	Create a mapping (item to ID / ID to item) from a dictionary.
	Items are ordered by decreasing frequency.
	"""
	sorted_items = sorted(dico.items(), key = lambda x: (-x[1], x[0]))
	id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
	item_to_id = {v: k for k, v in id_to_item.items()}
	return item_to_id, id_to_item

def create_dico(item_list):
	"""
	Create a dictionary of items from a list of list of items.
	"""
	assert type(item_list) is list
	dico = {}
	for items in item_list:
		for item in items:
			if item not in dico:
				dico[item] = 1
			else:
				dico[item] += 1
	return dico

def create_mask(data):
	"""
	Make mask for the data
	Data should be numpy matrix
	"""
	return datas.where(datas>0, 1, 0)
