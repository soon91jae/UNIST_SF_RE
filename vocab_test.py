from utils import util
from utils import vocab
from loader import tag_mapping
from loader import prepare_sentences, prepare_batch_datas
import pickle
data = pickle.load(open('dataset/TACRED/TACRED.pkl','rb'))
train = data['train']
dev = data['dev']
test = data['test']

print(train[0])
train_tokens = [t['tokens'] for t in train]
train_poss = [t['pos'] for t in train]
train_ners = [t['ner'] for t in train]
train_tags = [t['relation'] for t in train]
train_chars = ["".join(s) for s in train_tokens]

word_vocab = vocab.vocab(train_tokens, 300, 1, "word")
word_emb = word_vocab.token_emb; 
word_to_id, id_to_word = word_vocab.token_mapping

char_vocab = vocab.vocab(train_chars, 100, 0, "char")
char_emb = char_vocab.token_emb; 
char_to_id, id_to_char = char_vocab.token_mapping

pos_vocab = vocab.vocab(train_poss, 10, 0, "pos")
pos_emb = pos_vocab.token_emb; 
pos_to_id, id_to_pos = pos_vocab.token_mapping

ner_vocab = vocab.vocab(train_ners, 10, 0, "ner")
ner_emb = ner_vocab.token_emb; 
ner_to_id, id_to_ner = ner_vocab.token_mapping

tag_dico, tag_to_id, id_to_tag = tag_mapping([train_tags])

train_dataset = prepare_sentences(train, word_to_id, char_to_id, pos_to_id, ner_to_id, tag_to_id)
#dev_dataset = prepare_sentences(dev, word_to_id, char_to_id, pos_to_id, ner_to_id, tag_to_id)
#test_dataset = prepare_sentences(test, word_to_id, char_to_id, pos_to_id, ner_to_id, tag_to_id)

batchfied_train_dataset = prepare_batch_datas(train_dataset,shuffle = False)
print(char_to_id)
print(batchfied_train_dataset['chars'][0].shape)
print(batchfied_train_dataset['words'])
