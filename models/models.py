import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
from torch.nn.init import xavier_uniform_
from math import floor
import numpy as np
from .utils import normalize, sparse_mx_to_torch_sparse_tensor

class DPCNN(nn.Module):
    pass


class StrutureAwareHCN(nn.Module):
    def __init__(self, num_classes, embed_size, word_kernel_sizes, section_kernel_size, label_kernel_size, num_filter_maps, dropout, vocab_size = None, init_embeds = None, embed_freeze = False, code_embeds = None, lmbda = 0, method=None, use_ontology = False, use_hierarchy = False, total_num_classes = -1):
        super(StrutureAwareHCN, self).__init__()
        modules = []
        for kernel_size in word_kernel_sizes:
            conv_word = nn.Conv1d(embed_size, num_filter_maps, kernel_size, padding=int(floor(kernel_size / 2)))
            xavier_uniform_(conv_word.weight)
            modules.append(conv_word)
    
        self.convs_word = nn.ModuleList(modules=modules)
        self.conv_section = nn.Conv1d(num_filter_maps, num_filter_maps, section_kernel_size, padding=int(floor(section_kernel_size / 2)))
        xavier_uniform_(self.conv_section.weight)

        self.embed_drop = nn.Dropout(p=dropout)
        if init_embeds is None:
            self.embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(init_embeds), freeze = embed_freeze)
            self.embed.padding_idx = 0


        self.text_length_affine = nn.Linear(1, num_classes)
        xavier_uniform_(self.text_length_affine.weight)

        self.final = nn.Linear(num_filter_maps, num_classes)
        xavier_uniform_(self.final.weight)

        self.num_classes = num_classes
        self.num_filter_maps = num_filter_maps
        self.lmbda = lmbda
        self.method = method

        self.use_hierarchy = use_hierarchy
        self.use_ontology = use_ontology

        if use_hierarchy and use_ontology:
            self.word_U = nn.Linear(num_filter_maps, total_num_classes)
            xavier_uniform_(self.word_U.weight)

            ## context vector for section-level
            self.section_U = nn.Linear(num_filter_maps, num_classes)
            xavier_uniform_(self.section_U.weight)

            self.gcn = GCN(num_filter_maps, num_filter_maps, dropout = 0.15)
        
        elif use_hierarchy:
            ## context vector for word-level
            self.word_U = nn.Linear(num_filter_maps, num_classes)
            xavier_uniform_(self.word_U.weight)

            ## context vector for section-level
            self.section_U = nn.Linear(num_filter_maps, num_classes)
            xavier_uniform_(self.section_U.weight)

        elif use_ontology:
            self.word_U = nn.Linear(num_filter_maps, total_num_classes)
            xavier_uniform_(self.word_U.weight)
            self.gcn = GCN(num_filter_maps, num_filter_maps, dropout = 0.15)
        
        else:
            self.word_U = nn.Linear(num_filter_maps, num_classes)
            xavier_uniform_(self.word_U.weight)



        # if self.method in ['caml', 'mvc', 'hierarchy']:
        #     ## context vector for word-level
        #     self.word_U = nn.Linear(num_filter_maps, num_classes)
        #     xavier_uniform_(self.word_U.weight)
        #     if self.method in ['hierarchy']:
        #         ## context vector for section-level
        #         self.section_U = nn.Linear(num_filter_maps, num_classes)
        #         xavier_uniform_(self.section_U.weight)

        # elif self.method in ['hierarchy-gcn']:
        #     self.word_U = nn.Linear(num_filter_maps, total_number_classes)
        #     xavier_uniform_(self.word_U.weight)

        #     ## context vector for section-level
        #     self.section_U = nn.Linear(num_filter_maps, num_classes)
        #     xavier_uniform_(self.section_U.weight)

        #     self.gcn = GCN(num_filter_maps, num_filter_maps, dropout = 0.15)
        

        if code_embeds is not None:
            ## init code embed
            self.code_embed_init(code_embeds)

        if lmbda > 0:
            self.desc_embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)
            self.desc_embed.weight.data = self.embed.weight.data.clone()

            self.conv_label = nn.Conv1d(embed_size, num_filter_maps, kernel_size=label_kernel_size, padding=int(floor(label_kernel_size/2)))
            xavier_uniform_(self.conv_label.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform_(self.label_fc1.weight)
        
        
        # self.gcn = GCN(num_filter_maps, num_filter_maps)

    def embed_propagation(self, input_feats, adj):
        features = self.gcn(input_feats, adj)
        return features
    

    def focal_loss(self, input, target, eps=1e-10, gamma=2):
        probs = torch.clamp(input, eps, 1-eps)
        loss = - (torch.pow((1 - probs),gamma) * target * torch.log(probs) + torch.pow(probs, gamma) * (1 - target) * torch.log(1 - probs))
        loss = loss.sum(1)
        # return loss.mean()
        return loss.mean()
    

    def get_multilabel_loss(self, target, yhat):
        loss = F.binary_cross_entropy_with_logits(yhat, target)
        return loss

        # probs = F.sigmoid(yhat)
        # loss = self.focal_loss(probs, target)
        # return loss
    
    def get_label_reg_loss(self, desc_data, contexts, target):
        """

        """
        # print(desc_data)
        device = contexts.device
        b_batch = self.embed_description_v2(desc_data, device)

        diff = self.compare_label_embeddings(target, b_batch)
        # if diffs is not None:
        #     diffs = torch.stack(diffs).mean()
        return diff


    def caml(self, docs, doc_masks, doc_lengths, adj=None, leaf_idxs=None):
        word_u = self.word_U.weight
        if self.use_ontology:
            word_u_final = self.embed_propagation(word_u, adj)[leaf_idxs]

        device = docs.device
        docs = self.embed(docs)
        docs = self.embed_drop(docs)
        docs = docs.transpose(1, 2)
        multi_word_features = []
        for module in self.convs_word:
            word_features = F.tanh(module(docs))
            multi_word_features.append(word_features)
        multi_word_features = torch.stack(multi_word_features, dim=0).transpose(0,1) ## batch_size * kernel_num *  num_filter_maps * words
        ## max-pooling over kernel
        word_features, _ = torch.max(multi_word_features, dim=1)
        # word_features = multi_word_features[:,0] ## batch_size * num_filter_maps * words
        alpha = F.softmax(word_u[leaf_idxs].matmul(word_features), dim=2)
        context = alpha.matmul(word_features.transpose(1, 2))
        # y = self.final.weight.mul(context).sum(dim=2).add(self.final.bias)
        # print(word_u.shape, context.shape)
        if self.use_ontology:
            y = word_u_final.mul(context).sum(dim=2)
        else:
            y = self.final.weight.mul(context).sum(dim=2).add(self.final.bias)
        # if self.method == 'mvc':
        #     text_affine = self.text_length_affine.weight.mul(doc_lengths.float() / 2500.0).transpose(0, 1) ## batch_size * num_class
        #     y = y + F.sigmoid(text_affine)
        return y, context, None, None

    def forward(self, docs, doc_masks, doc_lengths, section_masks, section_lengths, adj=None, leaf_idxs=None):
        if not self.use_hierarchy:
            return self.caml(docs, doc_masks, doc_lengths, adj=adj, leaf_idxs=leaf_idxs)
        else:
            return self.hierarchy(docs, doc_masks, doc_lengths, section_masks, section_lengths, adj=adj, leaf_idxs=leaf_idxs)

    ## structure-aware attention without label graph propagation
    def hierarchy(self, docs, doc_masks, doc_lengths, section_masks, section_lengths, adj = None, leaf_idxs=None):
        """
            Inputs:
                docs: batch_size * words * embed_size
                doc_masks: batch_size * words
                section_masks: batch_size * section_num
                section_lengths: batch_size * section_num
            Outputs:
                yhat: logits
                context: final document representation
                beta: section weights
                alphas: word weights
        """
        word_u = self.word_U.weight
        section_u = self.section_U.weight
        if self.use_ontology:
            # print('using graph propagation')
            word_u = self.embed_propagation(word_u, adj)[leaf_idxs]
            # section_u = self.embed_propagation(section_u, self.adj)[self.leaf_idxs]
        

        device = section_masks.device
        section_num = section_masks.size(1)
        batch_size = section_masks.size(0)
        doc_words = docs.size(1)
        docs = self.embed(docs)
        docs = self.embed_drop(docs)
        docs = docs.transpose(1, 2)
        multi_word_features = []
        for module in self.convs_word:
            word_features = F.tanh(module(docs))
            multi_word_features.append(word_features)

        multi_word_features = torch.stack(multi_word_features, dim=0).transpose(0,1) ## batch_size * kernel_num *  num_filter_maps * words
        ## max-pooling over kernel
        # word_features = multi_word_features[:,0]
        
        word_features, _ = torch.max(multi_word_features, dim=1) ## batch_size * num_filter_maps * words


        # ##                                 ##
        # ##                                 ##
        # ##                                 ##
        # ##                                 ##
        # alphas_origin = word_u.matmul(word_features) ## batch_size * num_classes * words
        # section_start_indexs = torch.zeros(batch_size, device = device, requires_grad=False, dtype=torch.long)
        # section_features = []


        # for i in range(section_num):
        #     max_section_length = torch.max(section_lengths[:, i])
        #     section_word_masks = torch.zeros((batch_size, max_section_length), device=device, dtype=torch.float)
        #     section_alphas = torch.zeros((batch_size, self.num_classes, max_section_length), device=device, dtype=torch.float)
        #     section_words_features = torch.zeros((batch_size, self.num_filter_maps, max_section_length), device=device, dtype=torch.float)

        #     for batch_idx in range(batch_size):
        #         section_start_index = section_start_indexs[batch_idx]
        #         section_length = section_lengths[batch_idx, i]
        #         if section_length > 0:
        #             tmp = word_features[batch_idx, :, section_start_index:(section_start_index + section_length)]
        #             section_words_features[batch_idx, :, :section_length] = tmp
        #             section_alphas[batch_idx, :, :section_length] = alphas_origin[batch_idx, :, section_start_index:(section_start_index + section_length)]

        #         section_start_indexs[batch_idx] += section_length
        #         section_word_masks[batch_idx, :section_length] = 1.0
            
        #     alpha = F.softmax(section_alphas, dim=2)
        #     alpha = (alpha * section_word_masks.unsqueeze(1))
        #     alpha = alpha + 1e-8
        #     alpha = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        #     # print(section_words_features.shape)
        #     section_feature = alpha.matmul(section_words_features.transpose(1, 2)) ## batch_size * num_classes * num_filter_maps
        #     section_features.append(section_feature)

        # section_features = torch.stack(section_features, dim=0).transpose(0, 1) ## batch_size * section_num * num_classes * num_filter_maps
        # # context,_ = torch.max(section_features, dim=1)
        # # y = self.final.weight.mul(context).sum(dim=2).add(self.final.bias)
        # context = section_features
        # y = self.final.weight.mul(context).sum(-1).add(self.final.bias) ## batch_size * section_num * num_classes
        # y = y * section_masks.unsqueeze(2)
        # y, _ = torch.max(y, dim=1)
        # return y, context, None, None

        ##                                 ##
        ##                                 ##
        ##                                 ##
        ##                                 ##
        






        # section_start_indexs = torch.zeros(batch_size, device = device, requires_grad=False, dtype=torch.long)
        # section_features = []
        # alphas = torch.zeros((batch_size, self.num_classes, doc_words), device=device)

        # for i in range(section_num):
        #     ## section-based attentional pooling over words
        #     max_section_length = torch.max(section_lengths[:, i])
        #     # PAD = torch.zeros((max_section_length, self.num_filter_maps) , device=device, dtype=torch.float)
        #     section_word_masks = torch.zeros((batch_size, max_section_length), device=device, dtype=torch.float)

        #     section_words_features = torch.zeros((batch_size, self.num_filter_maps, max_section_length), device=device, dtype=torch.float)

        #     for batch_idx in range(batch_size):
        #         section_start_index = section_start_indexs[batch_idx]
        #         section_length = section_lengths[batch_idx, i]
        #         if section_length > 0:
        #             tmp = word_features[batch_idx, :, section_start_index:(section_start_index + section_length)]
        #             # section_words_features.append(tmp)
        #             section_words_features[batch_idx, :, :section_length] = tmp
        #         # else:
        #         #     section_words_features.append(PAD)
        #         section_start_indexs[batch_idx] += section_length
        #         section_word_masks[batch_idx, :section_length] = 1.0

        #     # section_word_masks = torch.tensor(section_word_masks, device=device, dtype=torch.float) ## batch_size * words
        #     # section_words_features = torch.stack(section_words_features, dim=0) ## batch_size * num_filter_maps * words

        #     alpha_origin = word_u.matmul(section_words_features)
        #     alpha = F.softmax(alpha_origin, dim=2) ## batch_size * num_classes * words
        #     alpha = (alpha * section_word_masks.unsqueeze(1))
        #     alpha = alpha + 1e-8
        #     alpha = alpha / torch.sum(alpha, dim=-1, keepdim=True)

        #     for batch_idx in range(batch_size):
        #         section_length = section_lengths[batch_idx, i]
        #         section_end_index = section_start_indexs[batch_idx]
        #         if section_length > 0:
        #             alphas[batch_idx, :, (section_end_index - section_length):section_end_index] = alpha[batch_idx, :, :section_length]

        #     section_feature = alpha.matmul(section_words_features.transpose(1, 2)) ## batch_size * num_classes * num_filter_maps
        #     section_features.append(section_feature)

        # section_features = torch.stack(section_features, dim=0).transpose(0, 1).transpose(1, 2) ## batch_size * num_classes * section_num * num_filter_maps
        # section_features = section_features.reshape(-1,section_num, self.num_filter_maps).transpose(1, 2) ## (batch_size * num_classes) * num_filter_maps * section_num

        # ## section feature encode
        # section_representation = F.tanh(self.conv_section(section_features))
        # section_representation = section_representation.reshape(batch_size, -1, self.num_filter_maps, section_num) ## batch_size * num_classes * num_filter_maps * section_num
        # section_representation = section_representation.permute([0, 1, 3, 2]) ## batch_size * num_classes * section_num * num_filter_maps 
        # ## section-level attention

        # """
        #     note there must be class aware
        # """
        # # print(section_representation.shape)
        # # print(self.section_U.weight.unsqueeze(2).shape)
        # beta_origin = torch.matmul(section_representation, section_u.unsqueeze(2)).squeeze(3) ## batch_size * num_classes * section_num
        # beta = F.softmax(beta_origin, dim=2)
        # beta = beta + 1e-8
        # beta = (beta * section_masks.unsqueeze(1))
        # beta = beta / torch.sum(beta, dim=-1, keepdim=True)

        # ## top-down attention
        # """
        #     if beta * (alpha * word_representation)
        # """
        # section_features = section_features.reshape(batch_size, -1, self.num_filter_maps, section_num).permute([0, 1, 3, 2]) ## batch_size * num_classes * section_num * num_filter_maps 
        # context = torch.matmul(beta.unsqueeze(2), section_features).squeeze(2) ## batch_size * num_classes * num_filter_maps
        # # """
        # #     if softmax(beta * alpha_origin)
        # # """
        # # y = self.final(context).squeeze(-1)
        # y = self.final.weight.mul(context).sum(dim=2).add(self.final.bias)
        # return y, context, beta, alphas


    def embed_description(self, descs, device):
        """
            descs: batch_size * 
        """
        b_batch = []
        for inst in descs:
            if len(inst) > 0:
                b_batch = b_batch + inst
        b_batch = np.array(b_batch)
        lt = torch.tensor(b_batch, dtype=torch.long, device=device)
        d = self.desc_embed(lt) ##  number * embed_size
        d = d.transpose(1, 2)
        d = F.tanh(self.conv_label(d))

        d = F.max_pool1d(d, kernel_size=d.size()[2])
        d = d.squeeze(2)
        b_inst = self.label_fc1(d)
        return b_inst

    def embed_description_v2(self, descs, device):
        b_batch = []
        for inst in descs:
            if len(inst) > 0:
                b_batch = b_batch + inst
        b_batch = np.array(b_batch)
        lt = torch.tensor(b_batch, dtype=torch.long, device=device)
        # d = self.desc_embed(lt) ##  number * embed_size
        d = self.embed(lt)
        d = d.transpose(1, 2)
        # d = F.tanh(self.conv_label(d))

        multi_word_features = []
        for module in self.convs_word:
            word_features = F.tanh(module(d))
            multi_word_features.append(word_features)
        multi_word_features = torch.stack(multi_word_features, dim=0).transpose(0,1) ## batch_size * kernel_num *  num_filter_maps * words
        ## max-pooling over kernel
        d, _ = torch.max(multi_word_features, dim=1)
        d = F.max_pool1d(d, kernel_size=d.size()[2])
        d = d.squeeze(2)
        # b_inst = self.label_fc1(d)
        b_inst = d
        return b_inst

        # for inst in descs:
        #     if len(inst) > 0:
        #         lt = torch.tensor(inst, dtype=torch.long, device=device)
        #         d = self.desc_embed(lt) ##  actual_classes * length * embed_size
        #         d = d.transpose(1, 2)
        #         d = self.conv_label(d)
        #         d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
        #         d = d.squeeze(2)
        #         b_inst = self.label_fc1(d)
        #         b_batch.append(b_inst)
        #     else:
        #         b_batch.append([])
        # return b_batch

    def compare_label_embeddings(self, target, b_batch):
        #description regularization loss
        #b_batch is the embedding from description conv (flatten)
        #iterate over batch because each instance has different # labels
        """
            mse(contexts - descs) * target
        """
        diffs = []
        batch_size = target.size(0)
        selected_contexts = torch.zeros((b_batch.size(0), b_batch.size(1)), device=target.device)
        # selected_contexts = []

        inds = torch.nonzero(target)

        if self.method in ['caml','hierarchy']:
            selected_contexts = self.final.weight[inds[:,1],:]
        elif self.method == 'mvc':
            selected_contexts = self.word_U.weight[inds[:,1],:]
        # print(selected_contexts.shape)
        # for i in range(batch_size):
        #     ti = target[i]
        #     inds = torch.nonzero(ti).squeeze()
        #     print(inds)
        #     selected_contexts[current_index:(cu
        # crrent_index + inds.size(0))] = contexts[i, inds]
        #     current_index = current_index + inds.size(0)

        diff = F.mse_loss(selected_contexts, b_batch, size_average=True)
        return diff
        # # print(contexts.shape)
        # for i, bi in enumerate(b_batch):
            
        #     ti = target[i]
        #     inds = torch.nonzero(ti).squeeze()
        #     zi = contexts[i, inds]

        #     # print(bi.shape)
        #     # print(inds)

        #     diff = (zi - bi).mul(zi - bi).mean()

        #     #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
        #     diffs.append(diff * bi.size()[0])
        # return diffs

    def code_embed_init(self, code_embeds):
        """
            code_embeds: num_classes * embed_size
        """
        self.word_U.weight.data = torch.tensor(code_embeds).clone()
        if self.use_hierarchy:
            self.section_U.weight.data = torch.tensor(code_embeds).clone()

class DenseAttnGraph(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout = 0.2):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x

    
