import json
import pickle

class TACRED(object):
    
    def data_transform(self, data):
        dataset = []
        for sent in data:
            data = {}
            
            id = sent['id']
            tokens = sent['tokens']
            sentence = " ".join(sent['tokens'])
            ner = sent['stanford_ner']
            pos = sent['stanford_pos']
            head = sent['stanford_head']
            deprel = sent['stanford_deprel']

            key_start = sent['obj_start']
            key_end = sent['obj_end']
            key_type = sent['obj_type']

            ans_start = sent['subj_start']
            ans_end = sent['subj_end']
            ans_type = sent['subj_type']

            relation = sent['relation']

            
            data['id'] = id
            data['tokens'] = tokens
            data['sentence'] = sentence
            data['ner'] = ner
            data['pos'] = pos
            data['head'] = head
            data['deprel'] = deprel
            data['key_start'] = key_start
            data['key_end'] = key_end
            data['key_type'] = key_type
            data['ans_start'] = ans_start
            data['ans_end'] = ans_end
            data['ans_type'] = ans_type
            data['relation'] = 'NA' if relation == 'no_relation' else relation
        
            
            
            dataset.append(data)
        return dataset
    
    def __init__(self):
        train = json.load(open('train.json'))
        test = json.load(open('test.json'))
        dev = json.load(open('dev.json'))
        
        train_dataset = self.data_transform(train)
        dev_dataset = self.data_transform(dev)
        test_dataset = self.data_transform(test)
        
        self.dataset = {'train': train_dataset, 'test': test_dataset, 'dev': dev_dataset}
        
    def get_dataset(self):
        return self.dataset
    
    def get_train_dataset(self):
        return self.dataset['train']
    
    def get_test_dataset(self):
        return self.dataset['test']
    
    def get_dev_dataset(self):
        return self.dataset['dev']

tacred = TACRED()
pickle.dump(tacred.get_dataset(), open('TACRED.pkl', 'wb'))
