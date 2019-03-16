import os
import argparse
import csv
import sys
import torch
from torch.backends import cudnn
from logger import Logger
import utils
import random
from dataset import MIMICDataset, load_lookups, load_embeddings, load_code_embeddings
from cocob import COCOBBackprop

###################
import logging
logging.basicConfig(level=logging.INFO, format='')
import time
import datetime
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from models.models import CAML, MultiScaleCNN, MultiScaleCNN_Seq2Seq
from models.metrics import all_metrics, all_metrics_ordered
from collections import defaultdict
from tqdm import tqdm
from models.utils import sparse_mx_to_torch_sparse_tensor
from utils import build_optimizer, print_metrics
###################

class Trainer:
    def __init__(self, args={}, init_embed = None, code_embed=None, code_desc = None, relation = None):
        self.train_logger = Logger(args.log_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.args = args
        self.init_embed = init_embed
        self.code_embed = code_embed

        self.model = self._build_model(args)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        if relation is not None:
            adj = relation['adj']
            leaf_idxs = relation['leaf_idxs']
            if torch.cuda.is_available():
                adj = adj.cuda()
            self.adj = adj
            self.leaf_idxs = leaf_idxs
        else:
            self.adj = None
            self.leaf_idxs = None

        if code_desc is not None:
            max_length = max([len(code)for code in code_desc])
            code_desc_array = np.zeros((len(code_desc), max_length), dtype=np.int32)
            for i, code in enumerate(code_desc):
                code_desc_array[i, :len(code)] = code
            self.code_desc = torch.tensor(code_desc_array, dtype=torch.long)
            if torch.cuda.is_available():
                self.code_desc = self.code_desc.cuda()
        else:
            self.code_desc = None
            
    def _build_model(self, args):
        if args.method in ['multiscale']:
            print('use multiscalCNN')
            model = MultiScaleCNN(
                num_classes=args.Y,
                embed_size = args.embed_size,
                num_filter_maps=args.num_filter_maps,
                dropout=args.dropout,
                num_layers= args.num_layers,
                vocab_size=args.vocab_size,
                init_embeds = self.init_embed,
                drop_rate=args.drop_rate,
                use_ontology=args.use_ontology,
                use_desc = args.use_desc,
                pooling=args.pooling,
                total_num_classes=args.total_number_classes
            )
        else:
            model = CAML(num_classes = args.Y, 
                        embed_size = args.embed_size, 
                        word_kernel_sizes=args.word_kernel_sizes, 
                        label_kernel_sizes = args.label_kernel_sizes,
                        num_filter_maps=args.num_filter_maps,
                        dropout=args.dropout,
                        vocab_size=args.vocab_size,
                        init_embeds = self.init_embed,
                        code_embeds = self.code_embed,
                        lmbda=args.lmbda,
                        method=args.method,
                        use_ontology=args.use_ontology,
                        use_desc = args.use_desc,
                        total_num_classes=args.total_number_classes
                    )
        return model
    
    def _build_optimizer(self, parameters, args={}):
        # optimizer = optim.Adam(parameters, lr = args.lr, weight_decay=args.weight_decay)
        # optimizer = COCOBBackprop(parameters)
        optimizer = build_optimizer(parameters, args)
        return optimizer

    def train(self, train_dataloader, dev_dataloader = None, test_dataloader = None):
        parameters = self.model.parameters()
        self.optimizer = self._build_optimizer(parameters, self.args)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones)

        metrics_hist = defaultdict(lambda:[])
        metrics_hist_te = defaultdict(lambda: [])
        metrics_hist_tr = defaultdict(lambda: [])

        model = self.model

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
       
        print('start training .....................')

        for epoch in range(self.args.epochs):
            
            metrics_all = self.train_epoch(epoch, model, train_dataloader, dev_dataloader=dev_dataloader, test_dataloader = test_dataloader, args=self.args)
            if dev_dataloader is not None:
                for name in metrics_all[0].keys():
                    metrics_hist[name].append(metrics_all[0][name])
                for name in metrics_all[1].keys():
                    metrics_hist_te[name].append(metrics_all[1][name])
                for name in metrics_all[2].keys():
                    metrics_hist_tr[name].append(metrics_all[2][name])
                metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)
                utils.save_metrics(metrics_hist_all, self.args.save_dir)

            info_to_save = {
                'model':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'lr':self.optimizer.param_groups[0]['lr'],
                'epoch':epoch
            }
            torch.save(info_to_save,os.path.join(self.args.save_dir,'model_{}.pth.tar'.format(epoch)))
            if self.args.optim != 'adam':
                self.scheduler.step()

    def train_epoch(self, epoch, model, train_dataloader, dev_dataloader = None, test_dataloader = None, args={}):
        print('training epoch {}'.format(epoch))
        start_time = time.time()
        dicts = train_dataloader.dataset.dicts
        ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
        unseen_code_inds = set(ind2c.keys())
        model.train()

        ind2c = train_dataloader.dataset.dicts['ind2c']
        unseen_code_inds = set(ind2c.keys())

        have_done_steps = epoch * len(train_dataloader)
        losses = []
        for step, sample in enumerate(train_dataloader):
            hadms, docs, labels, ordered_labels, doc_masks, doc_lengths, desc_vectors, code_set = sample

            unseen_code_inds = unseen_code_inds.difference(code_set)

            have_done_steps = have_done_steps + 1
            if torch.cuda.is_available():
                docs = docs.cuda()
                labels = labels.cuda()
                ordered_labels = ordered_labels.cuda()
                doc_lengths = doc_lengths.cuda()
                doc_masks = doc_masks.cuda()
                code_set = code_set.cuda()
                
            self.optimizer.zero_grad()
            if not self.args.code_set:
                code_set = None
            else:
                labels = labels[:, code_set]
            
            logits, contexts, _, _  = model(docs, doc_masks, doc_lengths, adj = self.adj, leaf_idxs = self.leaf_idxs, code_desc=self.code_desc, code_set=code_set)
            
            bce_loss = self.model.get_multilabel_loss(labels, logits)
            if args.lmbda > 0:
                reg_loss = self.model.get_label_reg_loss(desc_vectors, contexts, labels)
            else:
                reg_loss = torch.tensor(0.0, device=bce_loss.device)
            loss = bce_loss + args.lmbda * reg_loss
            loss.backward()
            if args.grad_clip_value is not None:
                nn.utils.clip_grad_value_(self.optimizer.param_groups[0]['params'],args.grad_clip_value)

            self.optimizer.step()
            losses.append(loss.cpu().item())

            if step % args.log_frq == 0 or step == len(train_dataloader) - 1:
                log_info = 'epoch {} {}/{} loss {:.4f} reg_loss {:.4f} {} {:.4f}mins'.format(
                    epoch,
                    step,
                    len(train_dataloader),
                    loss.item(),
                    reg_loss.item(),
                    docs.size(1),
                    (time.time() - start_time) / 60.0
                )
                self.logger.info(log_info)
                
        
        loss = np.mean(losses)
        
        if dev_dataloader is not None:
            metrics = self.val_epoch(epoch, model, dev_dataloader = dev_dataloader, args=args, fold='dev')
            print_metrics(metrics)
        else:
            metrics = defaultdict(float)
             
        if test_dataloader is not None:
            metrics_te = self.val_epoch(epoch, model, dev_dataloader = test_dataloader, args=args, fold='test')
        else:
            metrics_te = defaultdict(float)
        
        # if epoch == args.epoches - 1:
        #     metrics_te = val_epoch(epoch, model, dev_dataloader = dev_dataloader, args=args, fold='test')

        metrics_tr = {'loss': loss}
        metrics_all = (metrics, metrics_te, metrics_tr)


        self.train_logger.add_scalar_summary('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
        self.train_logger.add_scalar_summary('train/reg_loss',reg_loss, epoch)
        self.train_logger.add_scalar_summary('train/bce_loss',bce_loss, epoch)
        self.train_logger.add_scalar_summary('train/loss',loss, epoch)
        k = [5] if args.Y == 50 else [8,15]
        for ki in k:
            self.train_logger.add_scalar_summary('dev/p_{}'.format(ki), metrics[f'prec_at_{ki}'], epoch)

        self.train_logger.add_scalar_summary('dev/f1_micro', metrics['f1_micro'], epoch)
        self.train_logger.add_scalar_summary('dev/f1_macro', metrics['f1_macro'], epoch)
        return metrics_all

    def unseen_code_vecs(model, code_inds, dicts, gpu):
        """
            Use description module for codes not seen in training set.
        """
        code_vecs = tools.build_code_vecs(code_inds, dicts)
        code_inds, vecs = code_vecs
        #wrap it in an array so it's 3d
        desc_embeddings = model.embed_descriptions([vecs], gpu)[0]
        #replace relevant final_layer weights with desc embeddings 
        model.final.weight.data[code_inds, :] = desc_embeddings.data
        model.final.bias.data[code_inds] = 0

    
    def val_epoch(self, epoch, model, dev_dataloader, args={}, fold='dev'):
        y, yhat, yhat_raw, hids, losses = [], [], [], [], []
        model.eval()
        with torch.no_grad():
            for step, sample in enumerate(dev_dataloader):
                hadms, docs, labels, ordered_labels, doc_masks, doc_lengths, desc_vectors, code_set = sample
                if torch.cuda.is_available():
                    docs = docs.cuda()
                    labels = labels.cuda()
                    doc_lengths = doc_lengths.cuda()
                    doc_masks = doc_masks.cuda()

                logits, _, _, _  = model(docs, doc_masks, doc_lengths, adj = self.adj, leaf_idxs = self.leaf_idxs, code_desc=self.code_desc, code_set=None)
                loss = self.model.get_multilabel_loss(labels, logits)
                output = F.sigmoid(logits)
                output = output.cpu().numpy() if torch.cuda.is_available() else output.numpy()
                losses.append(loss.cpu().item() if torch.cuda.is_available() else loss.item())
                targets = labels.cpu().numpy() if torch.cuda.is_available() else labels.numpy()

                y.append(targets)
                yhat.append(np.round(output))
                yhat_raw.append(output)
                hids.extend(hadms)

        y = np.concatenate(y, axis=0)
        yhat = np.concatenate(yhat, axis=0)
        yhat_raw = np.concatenate(yhat_raw, axis=0)
        
        dicts = dev_dataloader.dataset.dicts
        ind2c = dicts['ind2c']
        #write the predictions

        preds_file = utils.write_preds(yhat, args.save_dir, epoch, hids, fold, ind2c, yhat_raw)
        #get metrics
        k = 5 if args.Y == 50 else [8,15]
        metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
        metrics['loss_%s' % fold] = np.mean(losses)
        return metrics


class Seq2SeqTrainer:
    def __init__(self, args={}, init_embed = None, code_embed=None, code_desc = None, relation = None):
        self.train_logger = Logger(args.log_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.args = args
        self.init_embed = init_embed
        self.code_embed = code_embed

        self.model = self._build_model(args)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        if relation is not None:
            adj = relation['adj']
            leaf_idxs = relation['leaf_idxs']
            if torch.cuda.is_available():
                adj = adj.cuda()
            self.adj = adj
            self.leaf_idxs = leaf_idxs
        else:
            self.adj = None
            self.leaf_idxs = None

        ## pad code vector
        if code_desc is not None:
            max_length = max([len(code)for code in code_desc])
            code_desc_array = np.zeros((len(code_desc), max_length), dtype=np.int32)
            for i, code in enumerate(code_desc):
                code_desc_array[i, :len(code)] = code
            self.code_desc = torch.tensor(code_desc_array, dtype=torch.long)
            if torch.cuda.is_available():
                self.code_desc = self.code_desc.cuda()
        else:
            self.code_desc = None
            
    def _build_model(self, args):
        model = MultiScaleCNN_Seq2Seq(
                num_classes = args.Y,
                embed_size = args.embed_size,
                vocab_size = args.Y,
                hidden_size = args.hidden_size,
                label_embed_size = args.label_embed_size,
                dropout = args.dropout,
                init_embed = self.init_embed,
                cell = args.cell,
                enc_num_layers = args.enc_num_layers,
                dec_num_layers=args.dec_num_layers
                )
        return model
    
    def _build_optimizer(self, parameters, args={}):
        # optimizer = optim.Adam(parameters, lr = args.lr, weight_decay=args.weight_decay)
        # optimizer = COCOBBackprop(parameters)
        optimizer = build_optimizer(parameters, args)
        return optimizer

    
    def train(self, train_dataloader, dev_dataloader = None, test_dataloader = None):
        parameters = self.model.parameters()
        parameters = [p for p in parameters if p.requires_grad]
        self.optimizer = self._build_optimizer(parameters, self.args)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones)

        metrics_hist = defaultdict(lambda:[])
        metrics_hist_te = defaultdict(lambda: [])
        metrics_hist_tr = defaultdict(lambda: [])

        model = self.model

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
       
        print('start training .....................')

        for epoch in range(self.args.epochs):
            
            metrics_all = self.train_epoch(epoch, model, train_dataloader, dev_dataloader=dev_dataloader, test_dataloader = test_dataloader, args=self.args)
            if dev_dataloader is not None:
                for name in metrics_all[0].keys():
                    metrics_hist[name].append(metrics_all[0][name])
                for name in metrics_all[1].keys():
                    metrics_hist_te[name].append(metrics_all[1][name])
                for name in metrics_all[2].keys():
                    metrics_hist_tr[name].append(metrics_all[2][name])
                metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)
                utils.save_metrics(metrics_hist_all, self.args.save_dir)
            info_to_save = {
                'model':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'lr':self.optimizer.param_groups[0]['lr'],
                'epoch':epoch
            }
            torch.save(info_to_save,os.path.join(self.args.save_dir,'model_{}.pth.tar'.format(epoch)))
            if self.args.optim != 'adam':
                self.scheduler.step()

    def train_epoch(self, epoch, model, train_dataloader, dev_dataloader = None, test_dataloader = None, args={}):
        print('training epoch {}'.format(epoch))
        start_time = time.time()
        dicts = train_dataloader.dataset.dicts
        ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
        unseen_code_inds = set(ind2c.keys())
        model.train()

        ind2c = train_dataloader.dataset.dicts['ind2c']
        unseen_code_inds = set(ind2c.keys())

        have_done_steps = epoch * len(train_dataloader)
        losses = []
        for step, sample in enumerate(train_dataloader):
            hadms, docs, labels, ordered_labels, doc_masks, doc_lengths, desc_vectors, code_set = sample

            unseen_code_inds = unseen_code_inds.difference(code_set)
            have_done_steps = have_done_steps + 1
            if torch.cuda.is_available():
                docs = docs.cuda()
                labels = labels.cuda()
                ordered_labels = ordered_labels.cuda()
                doc_lengths = doc_lengths.cuda()
                doc_masks = doc_masks.cuda()
                code_set = code_set.cuda()
                
            self.optimizer.zero_grad()

            logits = model(docs, doc_masks, doc_lengths, ordered_labels)

            loss = self.model.compute_loss(logits, ordered_labels)
            loss.backward()
            

            if args.grad_clip_value is not None:
                nn.utils.clip_grad_value_(self.optimizer.param_groups[0]['params'],args.grad_clip_value)

            self.optimizer.step()
            losses.append(loss.cpu().item())

            if step % args.log_frq == 0 or step == len(train_dataloader) - 1:
                log_info = 'epoch {} {}/{} loss {:.4f} {} {:.4f}mins'.format(
                    epoch,
                    step,
                    len(train_dataloader),
                    loss.item(),
                    docs.size(1),
                    (time.time() - start_time) / 60.0
                )
                self.logger.info(log_info)
        
        loss = np.mean(losses)
        
        if dev_dataloader is not None:
            metrics = self.val_epoch(epoch, model, dev_dataloader = dev_dataloader, args=args, fold='dev')
            print_metrics(metrics)
        else:
            metrics = defaultdict(float)
             
        if test_dataloader is not None:
            metrics_te = self.val_epoch(epoch, model, dev_dataloader = test_dataloader, args=args, fold='test')
        else:
            metrics_te = defaultdict(float)
        
        # if epoch == args.epoches - 1:
        #     metrics_te = val_epoch(epoch, model, dev_dataloader = dev_dataloader, args=args, fold='test')

        metrics_tr = {'loss': loss}
        metrics_all = (metrics, metrics_te, metrics_tr)


        self.train_logger.add_scalar_summary('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
        self.train_logger.add_scalar_summary('train/loss',loss, epoch)

        # k = [5] if args.Y == 50 else [8,15]
        # for ki in k:
        #     self.train_logger.add_scalar_summary('dev/p_{}'.format(ki), metrics[f'prec_at_{ki}'], epoch)

        self.train_logger.add_scalar_summary('dev/f1_micro', metrics['f1_micro'], epoch)
        self.train_logger.add_scalar_summary('dev/f1_macro', metrics['f1_macro'], epoch)
        return metrics_all

    def unseen_code_vecs(model, code_inds, dicts, gpu):
        """
            Use description module for codes not seen in training set.
        """
        code_vecs = tools.build_code_vecs(code_inds, dicts)
        code_inds, vecs = code_vecs
        #wrap it in an array so it's 3d
        desc_embeddings = model.embed_descriptions([vecs], gpu)[0]
        #replace relevant final_layer weights with desc embeddings 
        model.final.weight.data[code_inds, :] = desc_embeddings.data
        model.final.bias.data[code_inds] = 0

    
    def val_epoch(self, epoch, model, dev_dataloader, args={}, fold='dev'):

        y, yhat, yhat_raw, hids = [], [], [], []
        model.eval()
        with torch.no_grad():
            for step, sample in enumerate(dev_dataloader):
                hadms, docs, labels, ordered_labels, doc_masks, doc_lengths, desc_vectors, code_set = sample
                if torch.cuda.is_available():
                    docs = docs.cuda()
                    labels = labels.cuda()
                    ordered_labels = ordered_labels.cuda()
                    doc_lengths = doc_lengths.cuda()
                    doc_masks = doc_masks.cuda()

                results, probs = model.sample(docs, doc_masks, doc_lengths, steps = args.Y + 1)
                # print(result, probs)

                # loss = self.model.get_multilabel_loss(labels, logits)
                results = results.cpu().numpy() if torch.cuda.is_available() else results.numpy()
                probs = probs.cpu().numpy() if torch.cuda.is_available() else probs.numpy()
                targets = ordered_labels.cpu().numpy() if torch.cuda.is_available() else ordered_labels.numpy()

                pad_targets = np.zeros((targets.shape[0], args.Y + 1))
                pad_targets[:, :targets.shape[1]] = targets

                y.append(pad_targets)
                yhat.append(results)
                yhat_raw.append(probs)
                hids.extend(hadms)

        y = np.concatenate(y, axis=0)
        yhat = np.concatenate(yhat, axis=0)
        yhat_raw = np.concatenate(yhat_raw, axis=0)
        
        dicts = dev_dataloader.dataset.dicts
        ind2c = dicts['ind2c']
        #write the predictions

        # preds_file = utils.write_preds(yhat, args.save_dir, epoch, hids, fold, ind2c, yhat_raw)
        #get metrics
        k = 5 if args.Y == 50 else [8,15]
        metrics = all_metrics_ordered(yhat, y, k=k,  probs=yhat_raw)
        # metrics['loss_%s' % fold] = np.mean(losses)
        return metrics



def train(args):
    dicts = load_lookups(args, desc_embd=True, code_embed=(args.code_embed_file is not None))
    relation = dicts['relation']
    if relation is not None:
        setattr(args, 'total_number_classes', len(relation['ind2c']))
    else:
        setattr(args, 'total_number_classes', len(dicts['ind2c']))

    code_embed = dicts['code_embed']
    if code_embed is not None:
        setattr(args, 'num_filter_maps', len(code_embeds[0]))
    setattr(args, 'vocab_size', len(dicts['ind2w']))
    setattr(args, 'Y', len(dicts['ind2c']))

    
    train_dataset = MIMICDataset(args.train_file, dicts, mode='train', max_length = args.max_length)
    train_dataloader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.nw)

    dev_dataset = MIMICDataset(args.dev_file, dicts=dicts, max_length = args.max_length)
    dev_dataloader = dev_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.nw)

    test_dataset = MIMICDataset(args.test_file, dicts=dicts, max_length = args.max_length)
    test_dataloader = test_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.nw)


    if args.method in ['seq2seq']:
        trainer = Seq2SeqTrainer(args=args, init_embed = dicts['init_embed'])
        trainer.train(train_dataloader, dev_dataloader = dev_dataloader, test_dataloader=test_dataloader)
    else:
        trainer = Trainer(args = args, init_embed = dicts['init_embed'], code_embed = dicts['code_embed'], relation=dicts['relation'], code_desc=dicts['code_desc'])
        trainer.train(train_dataloader, dev_dataloader = dev_dataloader, test_dataloader=test_dataloader)



def main(args):
    utils.delete_path(args.log_dir)
    utils.delete_path(args.save_dir)
    utils.ensure_path(args.save_dir)
    utils.ensure_path(args.log_dir)
    utils.write_dict(vars(args), os.path.join(args.save_dir, 'arguments.csv'))

    torch.manual_seed(args.seed)
    cudnn.benchmark = True 
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train a structure aware hierarchy attention network for clinical note coding")
    parser.add_argument("--vocab_file", type=str, help="path to document vocab file")
    parser.add_argument("--embed_file", type=str, help="path to pretrained word embed file")
    parser.add_argument("--code_file", type=str, help="path to all code label file")
    parser.add_argument("--description_file", type=str, help="path to code description vector file")
    parser.add_argument("--relation_file", type=str, help="relation for codes")
    parser.add_argument("--train_file", type=str, help="path to train file")
    parser.add_argument("--dev_file", type=str, help="path to dev file")
    parser.add_argument("--test_file", type=str, help="path to test file")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--code_embed_file', type=str, help="path to ICD-9 code pretrained label embedding")
    parser.add_argument("--save_dir", type=str, help="dir for persistent")
    

    ##
    parser.add_argument('--log_frq', type=int, default=1, help="every steps to print log info")
    parser.add_argument('--batch_size', type=int, default=32, help="number of samples for one batch (default: 32)")
    parser.add_argument('--nw', type=int, default=4, help="number of worker for dataloader")
    parser.add_argument('--gpus', type=str, default=0, help='gpu ids to use, seperate by comma')
    parser.add_argument('--epochs', type=int, help="number of epochs to train")
    parser.add_argument('--method', type=str, default='caml', help='which model use to train (e.g. caml, hierarchy')
    parser.add_argument('--use_ontology', type=bool, default=False, help="if use knowledge ontology")
    parser.add_argument('--use_desc', type=bool, default=False, help="if use code description for classification weight")
    parser.add_argument('--code_set', type=bool, default=False, help="if only use current code set")
    parser.add_argument('--cell', type=str, default='gru')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--label_embed_size', type=int, default=150)
    parser.add_argument('--enc_num_layers', type=int, default=2)
    parser.add_argument('--dec_num_layers', type=int, default=2)
   
    ## hyper parameters
    parser.add_argument("--num_filter_maps", type=int, default=50, help="size of conv output (default: 50)")
    parser.add_argument("--embed_size", type=int, default=100, help="word embed size (default: 100)")
    parser.add_argument("--word_kernel_sizes", type=str, default="4", help="size of convolution filter for word level. give comma separated integers, e.g. 3,4,5")
    parser.add_argument("--label_kernel_sizes",type=str, default="4", help="size of convolution filter for label conv (default:4)")

    parser.add_argument("--weight_decay", type=float, default=0., help="coefficient for penalizing l2 norm of model weights (default: 0)")
    parser.add_argument("--milestones", type=str, default='20', help="milestone for lr scheduler, seperate by comma")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate to use for training (default: 0.001)")
    parser.add_argument('--optim', type=str, default='adam', help="optimizer type")
    parser.add_argument("--optim_alpha", type=float, default=0.9, help="momentum var for optimizer")
    parser.add_argument("--optim_beta", type=float, default=0.999, help="var for optimizer")
    parser.add_argument("--optim_epsilon", type=float, default=1e-8, help="var for optimizer")

    parser.add_argument("--dropout", type=float, default=0.2,
                        help="optional specification of dropout (default: 0.2)")
    parser.add_argument("--lmbda", type=float, default=0,
                        help="hyperparameter to tradeoff BCE loss and similarity embedding loss. defaults to 0, which won't create/use the description embedding module at all. ")
    parser.add_argument('--max_length', type=int, default=2500, help="max length for document")
    parser.add_argument('--num_layers', type=int, default=5, help='number of dense layers')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='drop rate for multi-scale cnn')
    parser.add_argument('--pooling', type=bool, default=False, help="multi-scale if use pooling")

    ## train policy parameter
    parser.add_argument('--grad_clip_value', type=float, default=3.0, help="parameter grad clip threhold (default: 0.35)")
    parser.add_argument('--seed', type=int, default=-1, help="torch random seed")

    args = parser.parse_args()
    ts = time.time()
    args.milestones = [int(m) for m in args.milestones.split(',')]
    args.save_dir = os.path.join('./outputs/saved_models', args.save_dir, datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H'))
    setattr(args,'log_dir',os.path.join(args.save_dir,'log'))
    args.word_kernel_sizes = [int(kernel_size) for kernel_size in args.word_kernel_sizes.split(',')]
    args.label_kernel_sizes = [int(kernel) for kernel in args.label_kernel_sizes.split(',')]
    if args.seed == -1:
        args.seed = random.randint(-2^30,2^30)
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)
