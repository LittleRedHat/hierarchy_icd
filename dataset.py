import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import ast
from conf import BasicConf as conf
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import scipy.sparse as sp
from models.utils import normalize, sparse_mx_to_torch_sparse_tensor


class MIMICDataset(Dataset):
    def __init__(self, notes_file, dicts, mode = 'train', desc_embed=False, max_length = 2500, args={}):
        super(MIMICDataset, self).__init__()
        self.notes = load_notes(notes_file, args = args)
        self.dicts = dicts
        self.max_length = max_length
        print('max length for document is {}'.format(self.max_length))


    def __getitem__(self, index):
        note = self.notes.iloc[index]
        text = note['TEXT']
        labels = note['LABELS']
        hadm = note['HADM_ID']

        labels = set(str(labels).split(';'))

        if 'nan' in labels:
            labels =  labels.remove('nan')
        
        if labels is None:
            labels = set()

        c2ind = self.dicts['c2ind']  
        label_idxs = np.zeros(len(c2ind))
        code_set = set()
       
        for label in labels:
            if label in c2ind:
                label_idxs[c2ind[label]] = 1.0
                code_set.add(c2ind[label])
            
        ## encode doc
        """
           split document to max length 
        """
        w2ind = self.dicts['w2ind']
       
        doc = []
        section_length = []
        current_length = 0
        MAX_LENGTH = self.max_length
        for section in text:
            remained_length = MAX_LENGTH - current_length
            if remained_length <= 0:
                break
            
            content = section['content'].split()

            actual_added_length = min(remained_length, len(content))
            section_length.append(actual_added_length)
            doc = doc + content[:actual_added_length]
            current_length += actual_added_length
            
        section_length = np.array(section_length)
        doc_idx = []
        for word in doc:
            if word in w2ind:
                doc_idx.append(w2ind[word])
            else:
                doc_idx.append(len(w2ind) + 1)

        
        """
            load description vector according to label index
        """
        dv_dict = self.dicts['dv']
        ind2c = self.dicts['ind2c']
        ind2w = self.dicts['ind2w']
        desc_vectors = []
        current_idxs = []
        for label in labels:
            if label in c2ind:
                current_idxs.append(c2ind[label])
        current_idxs = sorted(current_idxs)

        
        desc_lengths = []
        for idx in current_idxs:
            label = ind2c[idx]
            vector = dv_dict[label]
            # desc = ' '.join([ind2w.get(v,'UNK') for v in vector])
            # print(label,desc)
            desc_lengths.append(len(vector))
            desc_vectors.append(vector)

        ## pad description vector
        # max_desc_length = max(desc_lengths)
        # pad_desc_vectors = []
        # for vector in desc_vectors:
        #     pad_vector = vector + [0] * (max_desc_length - len(vector))
        #     pad_desc_vectors.append(pad_vector)
      
    

        doc_idx = np.array(doc_idx)
        section_num = len(section_length)
        doc_length = sum(section_length) 
        # print(hadm, doc_idx, label_idxs, doc_length, section_num, section_length, pad_desc_vectors)
        return hadm, doc_idx, label_idxs, doc_length, section_num, section_length, desc_vectors, desc_lengths, code_set


    def __len__(self):
        return len(self.notes)

    def display(self):
        num_labels = len(self.dicts['ind2c'])
        num_words = len(self.dicts['ind2w'])

        pass
    
    def get_dataloader(self,
                       batch_size = 32,
                       shuffle = True,
                       num_workers = 4):
        data_loader = DataLoader(
                                dataset=self,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False,
                                pin_memory=True,
                                collate_fn=self.collate_fn)
        return data_loader

    def collate_fn(self, data):
        hadms, docs, labels, doc_lengths, section_nums, section_lengths, desc_vectors, desc_lengths, code_sets = zip(*data)
        max_section_num = max(section_nums)
        ## pad extra for section_num align
        max_doc_length = max(doc_lengths)
        
        
        doc_masks = np.zeros((len(docs), max_doc_length), dtype=np.float32)

        targets = np.zeros((len(docs), max_doc_length), dtype=np.int64)
        ##  pad doc vec
        for i, doc in enumerate(docs):
            targets[i, :len(doc)] = doc
            doc_masks[i, :len(doc)] = 1.0
        
        section_masks = np.zeros((len(docs), max_section_num), dtype=np.float32)
        section_length_targets = np.zeros((len(docs), max_section_num), dtype=np.int64)
        ## pad section vec
        for i, section_num in enumerate(section_nums):
            section_masks[i, :section_num] = 1.0
            section_length_targets[i, :section_num] = section_lengths[i][:section_num]

        hadms = np.stack(hadms, axis=0)
        labels = np.stack(labels, axis=0).astype(np.float32)
        
        # for index, l in enumerate(labels):
        #     l_num = l.sum()
        #     r_num = len(desc_vectors[index])
            # print(l_num, r_num, hadms[index])

        ## pad description vector
        max_desc_length = 0
        for desc_length in desc_lengths:
            if len(desc_length):
                max_desc_length = max(max_desc_length, max(desc_length))
        # max_desc_length = max([max(desc_length) for desc_length in  desc_lengths])
        pad_desc_vectors = []
        for vectors in desc_vectors:
            pad_vectors_i = []
            for vector in vectors:
                pad_vector = vector + [0] * (max_desc_length - len(vector))
                pad_vectors_i.append(pad_vector)
            pad_desc_vectors.append(pad_vectors_i)
        
        batch_code_sets = set()
        for code_set in code_sets:
            batch_code_sets.union(code_set)
        
        

        return hadms,\
               torch.from_numpy(targets),\
               torch.from_numpy(labels),\
               torch.from_numpy(doc_masks),\
               torch.tensor(doc_lengths, dtype=torch.float),\
               torch.from_numpy(section_length_targets),\
               torch.from_numpy(section_masks),\
               pad_desc_vectors, \
               batch_code_sets


def code_reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code

def load_notes(notes_file, args = {}):
    notes = pd.read_csv(notes_file, index_col=None, dtype={'HADM_ID':int})
    notes['TEXT'] = notes['TEXT'].apply(lambda x:ast.literal_eval(x))
    # notes = notes.sort_values(by=['LENGTH'],reverse=True)
    return notes

def load_code_embeddings(embed_file, args={}):
    #also normalizes the embeddings
    W = dict()
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            code = line[0]
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W[code] = vec
    # W = np.array(W, dtype=np.float32)
    return W


def load_code_descriptions(version='mimic3'):
    #load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    ## read diagnosis code
    diagnose_df = pd.read_csv(os.path.join(conf.DATA_DIR,'D_ICD_DIAGNOSES.csv'),dtype={"ICD9_CODE": str,"HADM_ID":int})
    for index, row in diagnose_df.iterrows():
        code = str(row['ICD9_CODE'])
        desc = row['LONG_TITLE']
        desc_dict[code_reformat(code, True)] = desc
    ## read procedure code
    procedure_df = pd.read_csv(os.path.join(conf.DATA_DIR,'D_ICD_PROCEDURES.csv'),dtype={"ICD9_CODE": str,"HADM_ID":int})
    for index, row in procedure_df.iterrows():
        code = str(row['ICD9_CODE'])
        desc = row['LONG_TITLE']
        desc_dict[code_reformat(code, False)] = desc
        
    with open('%s/ICD9_descriptions' % conf.DATA_DIR, 'r') as labelfile:
        for i,row in enumerate(labelfile):
            row = row.rstrip().split()
            code = row[0]
            if code not in desc_dict.keys():
                desc_dict[code] = ' '.join(row[1:])
    return desc_dict

def load_description_vectors(description_file):
    dv_dict = {}
    with open(description_file,'r') as inf:
        ## 
        next(inf)
        for line in inf:
            line = line.rstrip()
            if line != '':
                items = line.split()
                code = items[0]
                vector = [int(v) for v in items[1:]]
                dv_dict[code] = vector
    # desc_dict = load_code_descriptions()
    # dv_dict = {}
    # for code, desc in desc_dict.items():
    #     for word in desc.split

    # dv_dict['36.01'] = dv_dict['V36.01']
    # dv_dict['36.05'] = dv_dict['V36.01']
    return dv_dict

def load_code_relations(relation_file):
    pairs = []
    full_codes = set()
    
    with open(relation_file,'r') as inf:
        for line in inf:
            line = line.rstrip()
            if line != '':
                items = line.split()
                parent = items[0]
                child = items[1]
                pairs.append((parent, child))
                full_codes.add(parent)
                full_codes.add(child)
    full_codes = sorted(full_codes)
    ind2c = {i:code for i, code in enumerate(full_codes)}
    c2ind = {v:k for k,v in ind2c.items()}

    # construct edges
    edges = []
    for (parent, child) in pairs:
        p_idx = c2ind[parent]
        c_idx = c2ind[child]
        edges.append([c_idx, p_idx])
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(full_codes), len(full_codes)),
                        dtype=np.float32)


    return adj, ind2c, c2ind


def load_vocab_dict(vocab_file, args = {}):
    vocab = set()
    with open(vocab_file, 'r') as inf:
        for line in inf:
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())
    ## first word is **PAD**
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    w2ind = {w:i for i,w in ind2w.items()}
    return ind2w, w2ind

def load_code_dict(label_file, args={}):
    codes = []
    with open(label_file, 'r') as inf:
        for line in inf:
            line = line.rstrip()
            if line != '':
                codes.append(line)
    ## remove 36.01 36.02 36.05 48.81 719.70
    ind2c = {i:c for i,c in enumerate(sorted(codes))}
    c2ind = {c:i for i, c in ind2c.items()}
    return ind2c, c2ind


def load_embeddings(embed_file, args={}):
    #also normalizes the embeddings
    W = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
        #UNK embedding, gaussian randomly initialized 
        print("adding unk embedding")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W, dtype=np.float32)
    return W

def load_lookups(args, desc_embd = None, code_embed = None):
    ind2w, w2ind = load_vocab_dict(args.vocab_file, args=args)
    ind2c, c2ind = load_code_dict(args.code_file, args=args)
    desc_dict = load_code_descriptions(version='mimic3')

    if desc_embd:
        dv_dict = load_description_vectors(args.description_file)
    else:
        dv_dict = None
    
    if args.relation_file is not None:
        adj, full_ind2c, full_c2ind = load_code_relations(args.relation_file)
        leaf_idxs = []
        for i in range(len(ind2c)):
            c = ind2c[i]
            leaf_idx = full_c2ind[c]
            leaf_idxs.append(leaf_idx)
        relation  = {'adj':adj,'ind2c':full_ind2c, 'c2ind':full_c2ind, 'leaf_idxs':leaf_idxs}
    else:
        relation = None

    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind, 'desc': desc_dict, 'dv': dv_dict, 'relation':relation}
    return dicts
