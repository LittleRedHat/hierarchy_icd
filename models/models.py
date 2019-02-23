import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
from torch.nn.init import xavier_uniform_
from math import floor
import numpy as np
from .utils import normalize, sparse_mx_to_torch_sparse_tensor


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x

class DPCNN(nn.Module):
    def __init__(self, num_classes, embed_size, num_filter_maps, dropout, vocab_size = None, init_embeds = None):
        super(DPCNN, self).__init__()
        self.embed_drop = nn.Dropout(p=dropout)
        if init_embeds is None:
            self.embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(init_embeds), freeze = embed_freeze)
            self.embed.padding_idx = 0
        
        self.final = nn.Linear(num_filter_maps, num_classes)
        xavier_uniform_(self.final.weight)

        if use_ontology:
            self.word_U = nn.Linear(num_filter_maps, total_num_classes)
            xavier_uniform_(self.word_U.weight)
            self.gcn = GCN(num_filter_maps, num_filter_maps, dropout = 0.15)
        else:
            self.word_U = nn.Linear(num_filter_maps, num_classes)
            xavier_uniform_(self.word_U.weight)
        
        self.use_ontology = use_ontology
        self.num_filter_maps = num_filter_maps



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate):
        super(_DenseLayer, self).__init__()
        # self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
        #                 growth_rate, kernel_size=1, stride=1, bias=False)),
        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                 kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('conv1', nn.Conv1d(num_input_features, num_output_features, kernel_size=3, stride=1, padding=1))
        self.add_module('bn1', nn.BatchNorm1d(num_output_features))
        self.add_module('relu1', nn.ELU(inplace=True))
        # self.add_module('tanh', nn.Tanh())
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes, embed_size, num_filter_maps, dropout, num_layers = 5, vocab_size = None, init_embeds = None, embed_freeze = False, drop_rate=0.0, use_ontology=False, total_num_classes=None):
        super(MultiScaleCNN, self).__init__()
        self.embed_drop = nn.Dropout(p=dropout)
        if init_embeds is None:
            self.embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(init_embeds), freeze = embed_freeze)
            self.embed.padding_idx = 0
        
        self.trans_layers = nn.ModuleList(modules=[
            nn.Sequential(*[
                nn.Conv1d(embed_size, num_filter_maps, kernel_size=1),
                nn.BatchNorm1d(num_filter_maps),
                nn.ELU(inplace=True)
                # nn.ReLU(inplace=True),
                # nn.Tanh()
            ]),
            # nn.Sequential(*[
            #     nn.Conv1d(num_filter_maps, num_filter_maps, kernel_size=1),
            #     nn.BatchNorm1d(num_filter_maps),
            #     nn.ELU(inplace=True)
            #     # nn.ReLU(inplace=True),
            #     # nn.Tanh()
            # ]),
            # nn.Sequential(*[
            #     nn.Conv1d(num_filter_maps, num_filter_maps, kernel_size=1),
            #     nn.BatchNorm1d(num_filter_maps),
            #     nn.ELU(inplace=True)
            #     # nn.ReLU(inplace=True),
            #     # nn.Tanh()
            # ])
        ])
        self.multi_scale_layers = nn.Sequential(
            *[
                _DenseLayer((len(self.trans_layers) + i) * num_filter_maps, num_filter_maps, drop_rate=drop_rate) for i in range(num_layers)
            ]
        )
        self.total_layers = len(self.trans_layers) + len(self.multi_scale_layers)

        self.final = nn.Linear(num_filter_maps, num_classes)
        xavier_uniform_(self.final.weight)

        if use_ontology:
            self.word_U = nn.Linear(num_filter_maps, total_num_classes)
            xavier_uniform_(self.word_U.weight)
            self.gcn = GCN(num_filter_maps, num_filter_maps, dropout = 0.15)

        else:
            self.word_U = nn.Linear(num_filter_maps, num_classes)
            xavier_uniform_(self.word_U.weight)
        
        self.use_ontology = use_ontology
        self.num_filter_maps = num_filter_maps

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                xavier_uniform_(m.weight)
            else:
                pass

    def get_multilabel_loss(self, target, yhat):
        loss = F.binary_cross_entropy_with_logits(yhat, target)
        return loss

    def forward(self, docs, doc_masks, doc_lengths, section_masks, section_lengths, adj=None, leaf_idxs=None):

        word_u = self.word_U.weight
        if self.use_ontology:
            word_u_final = self.embed_propagation(word_u, adj)[leaf_idxs]

        batch_size = docs.size(0)
        device = docs.device
        docs = self.embed(docs)
        docs = self.embed_drop(docs)
        docs = docs.transpose(1, 2)
        
        transition_layers_output = []
        output = docs
        for layer in self.trans_layers:
            output = layer(output)
            transition_layers_output.append(output)
        
        multi_scale_layer_input = torch.cat(transition_layers_output, dim=1)
        multi_scale_output = self.multi_scale_layers(multi_scale_layer_input) ## batch_size * (total_layers * num_filter_maps) * words
        multi_scale_output = multi_scale_output.reshape(batch_size, -1, self.num_filter_maps, multi_scale_output.size(-1)) ## batch_size * total_layers * num_filter_maps * words
        
        ## calculate scale attention



        ## calculate word attention
        multi_scale_output, _ = torch.max(multi_scale_output, dim=1)
        alpha = F.softmax(word_u.matmul(multi_scale_output), dim=2)
        context = alpha.matmul(multi_scale_output.transpose(1, 2)) 

        if self.use_ontology:
            y = word_u_final.mul(context).sum(dim=2)
        else:
            y = self.final.weight.mul(context).sum(dim=2).add(self.final.bias)
        
        return y, context, None, None
        
        

class StrutureAwareHCN(nn.Module):
    def __init__(self, num_classes, embed_size, word_kernel_sizes, section_kernel_size, label_kernel_size, num_filter_maps, dropout, vocab_size = None, init_embeds = None, embed_freeze = False, code_embeds = None, lmbda = 0, method=None, use_ontology = False, use_hierarchy = False, use_desc = False, total_num_classes = -1):
        super(StrutureAwareHCN, self).__init__()
        modules = []
        for kernel_size in word_kernel_sizes:
            conv_word = nn.Conv1d(embed_size, num_filter_maps, kernel_size, padding=int(floor(kernel_size / 2)))
            xavier_uniform_(conv_word.weight)
            modules.append(conv_word)
    
        self.convs_word = nn.ModuleList(modules=modules)
        if use_hierarchy:
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
            word_features = module(docs)
            # word_features = word_features[:, :self.num_filter_maps] * F.tanh(word_features[:, self.num_filter_maps:])
            word_features = F.tanh(word_features)
            multi_word_features.append(word_features)
        multi_word_features = torch.stack(multi_word_features, dim=0).transpose(0,1) ## batch_size * kernel_num *  num_filter_maps * words
        ###################
        # alpha = word_u.matmul(multi_word_features) ## batch_size * kernel_num * num_classes * words

        # scale_attention = F.softmax(alpha, dim=1) ## batch_size * kernel_num * num_classes * words
        # word_attention = F.softmax(alpha, dim=-1) ## batch_size * kernel_num * num_classes * words
        # final_attention = (scale_attention * word_attention)
        # # final_attention = torch.sum(final_attention, dim=1)
        # context = final_attention.matmul(multi_word_features.transpose(2, 3)) ## batch_size * kernel_num * num_classes * num_filter_maps
        # context = torch.sum(context, dim=1) ## batch_size * num_classes * num_filter
        # if self.use_ontology:
        #     y = word_u_final.mul(context).sum(dim=2)
        # else:
        #     y = self.final.weight.mul(context).sum(dim=2).add(self.final.bias)
        # # if self.method == 'mvc':
        # #     text_affine = self.text_length_affine.weight.mul(doc_lengths.float() / 2500.0).transpose(0, 1) ## batch_size * num_class
        # #     y = y + F.sigmoid(text_affine)
        # return y, context, None, None

        ####################

        ## max-pooling over kernel
        word_features, _ = torch.max(multi_word_features, dim=1) ## batch_size * num_filter_maps * words
        alpha = F.softmax(word_u.matmul(word_features), dim=2)
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
        section_start_indexs = torch.zeros(batch_size, device = device, requires_grad=False, dtype=torch.long)
        section_features = []

        for i in range(section_num):
            max_section_length = torch.max(section_lengths[:, i])
            section_word_masks = torch.zeros((batch_size, max_section_length), device=device, dtype=torch.float)
            section_alphas = torch.zeros((batch_size, self.num_classes, max_section_length), device=device, dtype=torch.float)
            section_words_features = torch.zeros((batch_size, self.num_filter_maps, max_section_length), device=device, dtype=torch.float)

            for batch_idx in range(batch_size):
                section_start_index = section_start_indexs[batch_idx]
                section_length = section_lengths[batch_idx, i]
                if section_length > 0:
                    tmp = word_features[batch_idx, :, section_start_index:(section_start_index + section_length)]
                    section_words_features[batch_idx, :, :section_length] = tmp
                    # section_alphas[batch_idx, :, :section_length] = alphas_origin[batch_idx, :, section_start_index:(section_start_index + section_length)]
            section_feature = F.max_pool1d(section_words_features, kernel_size=max_section_length).squeeze(-1)
            section_features.append(section_feature)

        section_features = torch.stack(section_features, dim=0).transpose(0, 1) ## batch_size * section_num * num_filter_maps




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

    
