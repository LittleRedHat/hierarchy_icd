import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from math import floor
import numpy as np
from .utils import normalize, sparse_mx_to_torch_sparse_tensor
from .layers import luong_gate_attention, DenseLayer, DualPathBlock, ResnetBlock

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

class MultiScaleCNN_Seq2Seq(nn.Module):
    def __init__(self, num_classes, embed_size, vocab_size, hidden_size, label_embed_size, init_embed = None, dropout = 0.15, cell='gru', enc_num_layers = 2, bidirectional = False, dec_num_layers=2, attention_type='luong_gate'):
        super(MultiScaleCNN_Seq2Seq, self).__init__()

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
                nn.SELU(inplace=True)
            ]),
            nn.Sequential(*[
                nn.Conv1d(num_filter_maps, num_filter_maps, kernel_size=1),
                nn.BatchNorm1d(num_filter_maps),
                nn.SELU(inplace=True)
            ])
        ])

        self.multi_scale_layers = nn.ModuleList(
            [
                DenseLayer((i + 1) * num_filter_maps, num_filter_maps, drop_rate=drop_rate, pooling=pooling & (i != num_layers - 1)) for i in range(num_layers)
            ]
        )
        total_layers = 1 + len(self.multi_scale_layers)


        self.attention_affine = nn.Sequential(*[
            nn.Conv1d(total_layers * num_filter_maps, total_layers * num_filter_maps, kernel_size=1, stride=1, groups=total_layers),
            nn.BatchNorm1d(total_layers * num_filter_maps),
            nn.SELU(inplace=True)
        ])


        ## decoder ##
        ## <BOS> -> 0 <EOS> -> last tokens
        self.code_embed = nn.Embedding(num_classes + 2, label_embed_size)
        self.decoder = nn.GRU(input_size=label_embed_size, hidden_size=hidden_size, num_layers = dec_num_layers, dropout=dropout, batch_first=True)
        self.num_classes = num_classes

        ## add <EOS>, <BOS> token
        self.dec_linear = nn.Linear(hidden_size, num_classes + 2)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                xavier_uniform_(m.weight)

    def compute_loss(self, logits, target):
        logprobs = F.log_softmax(logits, dim=-1)

        mask = torch.zeros((target.size(0), target.size(1)), dtype=torch.float, device=logits.device)

        valid_elements = torch.nonzero(target)
        mask[valid_elements[:, 0], valid_elements[:, 1]] = 1.0

        loss = -1 * logprobs.gather(2, target.unsqueeze(2)).squeeze(2) ## bs * cs
        loss = loss * mask
        loss = loss.mean()
        return loss


    def encoder_forward(self, docs, doc_lengths):

       

    def decoder_forward(self, input, state, conv, context):

        embeds = self.code_embed(input)

        output, state = self.decoder(embeds, state)

        output, attn_weights = self.attention(output, conv, context)

        output = self.dec_linear(output)
        return output, state, attn_weights


    def forward(self, docs, doc_masks, doc_lengths, target):

        context, conv, state = self.encoder_forward(docs, doc_lengths)

        batch_size = docs.size(0)

        pad = torch.zeros((batch_size, 1), device=docs.device, dtype=torch.long)

        dec_input = torch.cat((pad, target[:, :-1]), dim=1)
        logits, state, attn = self.decoder_forward(dec_input, state, conv, context)
        return logits
    
    def sample(self, docs, doc_masks, doc_lengths, steps, take_all = False):
        context, conv, enc_state = self.encoder_forward(docs, doc_lengths)
        batch_size = docs.size(0)

        dec_input = torch.zeros((batch_size, 1), device=docs.device, dtype=torch.long)
        dec_state = enc_state
        
        results = torch.zeros((batch_size, steps), dtype=torch.long, device=docs.device)
        probs = torch.zeros((batch_size, steps), dtype=torch.float, device=docs.device)

        for si in range(steps):
            logits, dec_state, attn = self.decoder_forward(dec_input, dec_state, conv, context)

            logits = logits.squeeze(1)
            preds = F.softmax(logits, dim=-1)

            ## ensure get different labels
            prev_ix = torch.nonzero(results)
            if prev_ix.size(0):
                preds[prev_ix[:, 0], prev_ix[:, 1]] = 0.0 ## batch * classes
            ##
            preds[:, 0] = 0.0
            ps, dec_input = torch.max(preds, dim=-1)

            results[:, si] = dec_input
            probs[:, si] = ps

            dec_input = dec_input.unsqueeze(1)
    
        return results, probs

    def beam_sample(self, docs, doc_lengths, beam_size = 5, steps = -1, take_all = False):
        context, conv, enc_state = self.encoder_forward(docs, doc_lengths)
        batch_size = docs.size(0)

        seqlogprobs = torch.tensor((steps, batch_size), dtype=torch.float, device=docs.device)
        seq = torch.tensor((steps, batch_size), dtype=torch.long, device=docs.device)

        done_beams = [[] for _ in range(batch_size)]

        for bi in range(batch_size):
            init_state = torch.expand(enc_state[0][bi], (beam_size, -1)), torch.expand(enc_state[1][bi], (beam_size, -1))
            init_conv = torch.expand(conv[bi], (beam_size, -1, -1))
            init_context = torch.expand(context[bi], (beam_size, -1, -1))

            init_seq = torch.zeros((beam_size, 1), dtype=torch.long, device=docs.device)
            logits, state, attn = self.decoder_forward(init_seq, init_state, init_conv, init_context)
            init_logprobs = F.log_softmax(logits, dim=-1)

            done_beams[bi] = self.beam_search(init_state, init_logprobs, init_conv, init_context, steps, beam_size)

    def beam_search(self, init_state, init_logprobs, init_conv, init_context, steps, beam_size):
        device = init_logprobs.device

        def beam_step(logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            ys, ix = torch.sort(logprobs, dim=1, descending=True)
            rows = beam_size
            cols = min(beam_size, ys.size(1))
            candidates = []
            if t == 0:
                rows = 1
            for c in range(cols): ## for every word
                for r in range(rows): ## for every beam
                    local_logprob = ys[r, c].item()
                    candidate_logprob = beam_logprobs_sum[r] + local_logprob
                    candidates.append({'c':ix[r, c],'r':r, 'p':candidate_logprob, 'l':local_logprob})
            candidates = sorted(candidates, key=lambda x:-x['p'])
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()

            new_state = [_.clone() for _ in state]

            for vix in range(beam_size):
                v = condidates[vix]
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]

                    for state_ix in range(len(new_state)):
                        new_state[state_ix][vix] = state[state_ix][v['q']]
                    
                    beam_seq[t, vix] = v['c']
                    beam_seq_logprobs[t, vix] = v['l']
                    beam_logprobs_sum[vix] = v['p']
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        beam_seq = torch.tensor((steps, beam_size), device=device, dtype=torch.long)
        beam_seq_logprobs = torch.zeros((steps, beam_size), dtype=torch.float, device=device)
        beam_logprobs_sum = torch.zeros(beam_size, dtype=torch.float, device=device)
        state = init_state
        logprobs = init_logprobs
        context = init_context
        conv = init_conv
        for si in range(steps):
            beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates = beam_step(logprobs, beam_size, si, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state)

            input = beam_seq[si].unsqueeze(1) ## beam_size * 1
            if input.sum() == 0:
                break
            logits, state, attn = self.decoder_forward(input, state, conv, context)
            logprobs = F.log_softmax(logits, dim=-1)

class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes, embed_size, num_filter_maps, dropout, num_layers = 5, pooling = False, vocab_size = None, init_embeds = None, embed_freeze = False, drop_rate=0.0, use_ontology=False, use_desc=False, total_num_classes=None):
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
                nn.SELU(inplace=True)
            ]),
            nn.Sequential(*[
                nn.Conv1d(num_filter_maps, num_filter_maps, kernel_size=1),
                nn.BatchNorm1d(num_filter_maps),
                nn.SELU(inplace=True)
            ])
        ])
        # self.multi_scale_layers = nn.ModuleList(
        #     [
        #         DenseLayer((i + 1) * num_filter_maps, num_filter_maps, drop_rate=drop_rate, pooling=pooling & (i != num_layers - 1)) for i in range(num_layers)
        #     ]
        # )
        # total_layers = 1 + len(self.multi_scale_layers)

        in_chs = num_filter_maps
        k_r = 256
        groups = 32
        modules = [
            DualPathBlock(in_chs, k_r, k_r, num_filter_maps, num_filter_maps, groups, block_type='proj', b = False)
        ]
        in_chs += 2 * num_filter_maps
        for i in range(1, num_layers):
            m = DualPathBlock(
                in_chs, k_r, k_r, num_filter_maps, num_filter_maps, groups, block_type='normal', b = False
            )
            in_chs += num_filter_maps

            modules.append(m)
        self.multi_scale_layers = nn.ModuleList(modules)
        total_layers = 2 + len(self.multi_scale_layers)


        self.attention_affine = nn.Sequential(*[
            nn.Conv1d(total_layers * num_filter_maps, total_layers * num_filter_maps, kernel_size=1, stride=1, groups=total_layers),
            nn.BatchNorm1d(total_layers * num_filter_maps),
            nn.SELU(inplace=True)
        ])
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
        self.use_desc = use_desc

        if use_desc > 0:
            label_kernel_size = [1, 3, 5]
            self.desc_embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)
            self.desc_embed.weight.data = self.embed.weight.data.clone()
            self.conv_label = nn.ModuleList([nn.Conv1d(embed_size, num_filter_maps, kernel_size=kernel, padding=int(floor(kernel / 2))) for kernel in label_kernel_size])
            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform_(self.label_fc1.weight)
        

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                xavier_uniform_(m.weight)
    
    def embed_description(self, descs):
        """
            descs: batch_size * 
        """
        d = self.desc_embed(descs) ##  number * embed_size
        d = d.transpose(1, 2)
        ds = []
        for conv in self.conv_label:
            ds.append(F.tanh(conv(d))) ## num_classes * num_filter_maps * words
        ds = torch.stack(ds, dim=0)
        d, _ = torch.max(ds, dim=0)

        # d = F.tanh(self.conv_label(d))
        d = F.max_pool1d(d, kernel_size=d.size()[2])
        d = d.squeeze(2)
        d = self.label_fc1(d)
        return d

    def embed_propagation(self, input_feats, adj):
        features = self.gcn(input_feats, adj)
        return features
    
    def get_multilabel_loss(self, target, logits, eps=1e-8):
        
        # device = target.device
        # yhat = F.sigmoid(logits)
        # yhat = torch.clamp(yhat, eps, 1-eps)
        # y = torch.full((target.size(0), target.size(1)), -1, device=target.device, dtype=torch.long)
        # for bi in range(target.size(0)):
        #     inds = torch.nonzero(target[bi])[:, 0]
        #     y[bi, :inds.size(0)] = inds
        # loss = F.multilabel_margin_loss(yhat, y)
        # return loss


        # probs = F.softmax(logits, dim=-1)
        # probs = torch.clamp(probs, eps, 1-eps)
        # target = target / torch.sum(target, dim = 1, keepdim = True)
        # loss = -1 * torch.sum(torch.log(probs) * target) / probs.shape[0]
        # return loss

        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss

    def forward(self, docs, doc_masks, doc_lengths, adj=None, leaf_idxs=None, code_desc = None, code_set = None):

        word_u = self.word_U.weight
        if self.use_ontology:
            word_u_final = self.embed_propagation(word_u, adj)[leaf_idxs]
            word_u = word_u[leaf_idxs]

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
        
        # multi_scale_layer_input = torch.cat(transition_layers_output, dim=1)
        multi_scale_output = output
        for module in self.multi_scale_layers:
            multi_scale_output = module(multi_scale_output)
        # multi_scale_output = multi_scale_output.reshape(batch_size, -1, self.num_filter_maps, multi_scale_output.size(-1)) ## batch_size * total_layers * num_filter_maps * words
        if isinstance(multi_scale_output, tuple):
            multi_scale_output = torch.cat(multi_scale_output, dim=1)
        
        # print(multi_scale_output.shape)
        ## calculate scale attention
        multi_scale_output = self.attention_affine(multi_scale_output).reshape(batch_size, -1, self.num_filter_maps, multi_scale_output.size(-1)) ## batch_size * total_layers * num_filter_maps * words
        scale = torch.sum(multi_scale_output, dim=2).permute(0, 2 , 1) ## batch_size * words * total_layers
        scale = F.softmax(scale, dim=-1).permute(0, 2, 1) ## batch_size * total_layers * words

        multi_scale_output =  multi_scale_output * scale.unsqueeze(2)
        multi_scale_output = torch.sum(multi_scale_output, dim=1) ## batch_size * num_filter_maps * words

        ## calculate word attention
        # multi_scale_output = multi_scale_output.reshape(batch_size, -1, self.num_filter_maps, multi_scale_output.size(-1)) ## batch_size * total_layers * num_filter_maps * words
        # multi_scale_output, _ = torch.max(multi_scale_output, dim=1)

        if code_set is not None:
            alpha = F.softmax(word_u[code_set].matmul(multi_scale_output), dim=2)
            context = alpha.matmul(multi_scale_output.transpose(1, 2)) 
            if self.use_ontology:
                y = word_u_final[code_set].mul(context).sum(dim=2)
            elif self.use_desc:
                weight = self.embed_description(code_desc[code_set])
                y = weight.mul(context).sum(dim=2)
            else:
                y = self.final.weight[code_set].mul(context).sum(dim=2).add(self.final.bias[code_set])
        else:
            alpha = F.softmax(word_u.matmul(multi_scale_output), dim=2)
            context = alpha.matmul(multi_scale_output.transpose(1, 2))
            
            if self.use_ontology:
                
                y = word_u_final.mul(context).sum(dim=2)
            elif self.use_desc:
               
                weight = self.embed_description(code_desc)
                y = weight.mul(context).sum(dim=2)
            else:
                y = self.final.weight.mul(context).sum(dim=2).add(self.final.bias)

        return y, context, None, None

class CAML(nn.Module):
    def __init__(self, num_classes, embed_size, word_kernel_sizes, label_kernel_sizes, num_filter_maps, dropout, vocab_size = None, init_embeds = None, embed_freeze = False, code_embeds = None, lmbda = 0, method=None, use_ontology = False, use_desc = False, total_num_classes = -1):
        super(CAML, self).__init__()
        modules = []
        for kernel_size in word_kernel_sizes:
            conv_word = nn.Conv1d(embed_size, num_filter_maps, kernel_size, padding=int(floor(kernel_size / 2)))
            xavier_uniform_(conv_word.weight)
            modules.append(conv_word)
    
        self.convs_word = nn.ModuleList(modules=modules)

        self.embed_drop = nn.Dropout(p=dropout)
        if init_embeds is None:
            self.embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(init_embeds), freeze = embed_freeze)
            self.embed.padding_idx = 0

        self.final = nn.Linear(num_filter_maps, num_classes)
        xavier_uniform_(self.final.weight)

        self.num_classes = num_classes
        self.num_filter_maps = num_filter_maps
        self.lmbda = lmbda
        self.method = method
        self.use_desc = use_desc

        self.use_ontology = use_ontology

        if use_ontology:
            self.word_U = nn.Linear(num_filter_maps, total_num_classes)
            xavier_uniform_(self.word_U.weight)
            self.gcn = GCN(num_filter_maps, num_filter_maps, dropout = 0.15)
        
        else:
            self.word_U = nn.Linear(num_filter_maps, num_classes)
            xavier_uniform_(self.word_U.weight)

        if code_embeds is not None:
            ## init code embed
            self.code_embed_init(code_embeds)

        if use_desc:
            self.desc_embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)
            self.desc_embed.weight.data = self.embed.weight.data.clone()
            self.conv_label = nn.ModuleList([nn.Conv1d(embed_size, num_filter_maps, kernel_size=kernel, padding=int(floor(kernel / 2))) for kernel in label_kernel_sizes])
            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform_(self.label_fc1.weight)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                xavier_uniform_(m.weight)

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
    
    def get_label_reg_loss(self, desc_data, contexts, target):
        """
        """
        device = contexts.device
        b_batch = self.embed_description(desc_data, device)
        diff = self.compare_label_embeddings(target, b_batch)
        return diff

    def caml(self, docs, doc_masks, doc_lengths, adj=None, leaf_idxs=None, code_desc=None):
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
            word_features = F.tanh(word_features)
            multi_word_features.append(word_features)
        multi_word_features = torch.stack(multi_word_features, dim=0).transpose(0,1) ## batch_size * kernel_num *  num_filter_maps * words

        ####################
        ## max-pooling over kernel
        word_features, _ = torch.max(multi_word_features, dim=1) ## batch_size * num_filter_maps * words
        alpha = F.softmax(word_u.matmul(word_features), dim=2)
        context = alpha.matmul(word_features.transpose(1, 2))

        if self.use_ontology:
            y = word_u_final.mul(context).sum(dim=2)
        
        elif self.use_desc:
            weight = self.embed_description(code_desc)
            y = weight.mul(context).sum(dim=2)
        else:
            y = self.final.weight.mul(context).sum(dim=2).add(self.final.bias)
        # if self.method == 'mvc':
        #     text_affine = self.text_length_affine.weight.mul(doc_lengths.float() / 2500.0).transpose(0, 1) ## batch_size * num_class
        #     y = y + F.sigmoid(text_affine)
        return y, context, None, None

    def forward(self, docs, doc_masks, doc_lengths, adj=None, leaf_idxs=None, code_desc=None, code_set=None):
        return self.caml(docs, doc_masks, doc_lengths, adj=adj, leaf_idxs=leaf_idxs, code_desc=code_desc)


    def embed_description(self, descs):
        """
            descs: batch_size * 
        """
        d = self.desc_embed(descs) ##  number * embed_size
        d = d.transpose(1, 2)
        ds = []
        for conv in self.conv_label:
            ds.append(F.tanh(conv(d))) ## num_classes * num_filter_maps * words
        ds = torch.stack(ds, dim=0)
        d, _ = torch.max(ds, dim=0)

        # d = F.tanh(self.conv_label(d))
        d = F.max_pool1d(d, kernel_size=d.size()[2])
        d = d.squeeze(2)
        d = self.label_fc1(d)
        return d

    # def embed_description(self, descs, device):
    #     """
    #         descs: batch_size * 
    #     """
    #     b_batch = []
    #     for inst in descs:
    #         if len(inst) > 0:
    #             b_batch = b_batch + inst
    #     b_batch = np.array(b_batch)
    #     lt = torch.tensor(b_batch, dtype=torch.long, device=device)
    #     d = self.desc_embed(lt) ##  number * embed_size
    #     d = d.transpose(1, 2)
    #     d = F.tanh(self.conv_label(d))

    #     d = F.max_pool1d(d, kernel_size=d.size()[2])
    #     d = d.squeeze(2)
    #     b_inst = self.label_fc1(d)
    #     return b_inst

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

