
from re import S
from torch import nn
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from .data_loader import MRConREDataLoader
from .utils import AverageMeter, Logger

import torch
import time
import numpy as np


class BagRE(nn.Module):

    def __init__(self,
                 model,
                 writer,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 args):

        super().__init__()
        self.args = args
        self.model = model


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # data loader
        # max_bag_size: avoid oom
        self.train_loader = MRConREDataLoader(
            path=train_path,
            rel2id=model.rel2id,
            tokenizer=model.sentence_encoder.tokenize_pair,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            bag_size=self.args.bag_size,
            max_bag_size=self.args.max_bag_size,
            entpair_as_bag=False
        )
        # self.eval_loader = CILDataLoader(
        #     path=val_path,
        #     rel2id=model.rel2id,
        #     tokenizer=model.sentence_encoder.tokenize,
        #     batch_size=self.args.eval_batch_size,
        #     shuffle=False,
        #     bag_size=0,
        #     max_bag_size=self.args.max_bag_size,
        #     entpair_as_bag=True
        # )
        self.test_loader = MRConREDataLoader(
            path=test_path,
            rel2id=model.rel2id,
            tokenizer=model.sentence_encoder.tokenize,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            bag_size=0,
            max_bag_size=0,
            entpair_as_bag=True
        )

        # tensorboard writer
        self.writer = writer

        # criterion
        if self.args.loss_weight:
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # params and optimizer
        params = self.model.named_parameters()

        # bert adam
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in params if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.eps, correct_bias=False)
        self.total_steps = len(self.train_loader) * self.args.max_epoch
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.total_steps)

        self.ckpt = ckpt
        self.global_steps = 0

        if self.args.dont_save_logger:
            self.logger = Logger.get_()
        else:
            prefix = str(time.time())
            self.logger = Logger.get(prefix=prefix)

    def train_model(self):
        best_auc, best_f1 = 0., 0.
        best_p100, best_p200, best_p300 = 0., 0., 0.
        run_time = 0.

        list_loss = []
        list_test_auc = []

        for epoch in range(1, self.args.max_epoch+1):
            self.model.train()
            self.logger.info("=== epoch {} train start ===".format(epoch))
        
            avg_loss = AverageMeter()
            #avg_mlm_loss = AverageMeter()
            avg_cl_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_pos_acc = AverageMeter()


            t = tqdm(self.train_loader)

            start_time = time.time()
            for _, data in enumerate(t):
                for i in range(len(data)):
                    try:
                        data[i] = data[i].to(self.device)
                    except:
                        pass

                self.global_steps += 1

                p = float(self.global_steps) / (self.args.warmup_steps * 2.0)
                alpha = 2. / (1. + np.exp(-2. * p)) - 1
                
                tau_c_max = self.args.tau_c_max
                tau_clean = self.args.tau_clean
                if self.global_steps < self.args.warmup_steps:
                    threshold_clean = min(tau_clean *  self.global_steps / self.args.warmup_steps, tau_clean)
                else:
                    threshold_clean = (tau_c_max - tau_clean) * (self.global_steps - self.args.warmup_steps) / (len(self.train_loader)*self.args.filter_stop_epoch - self.args.warmup_steps) + tau_clean
                    threshold_clean = min(threshold_clean, tau_c_max)

                label, bag_name, scope, bag, args = data[0], data[1], data[2], data[3], data[4:]

                if epoch < self.args.mixmatch_begin_epoch or self.global_steps < self.args.warmup_steps:
                    mlm_loss, cl_loss, mil_loss = self.model(label, scope, *args, bag_size=self.args.bag_size, threshold_clean = threshold_clean)
                else:
                    mlm_loss, cl_loss, mil_loss = self.model(label, scope, *args, bag_size=self.args.bag_size, threshold_clean = threshold_clean, mixmatch = True, epoch = epoch)
                
    
                cl_loss = cl_loss.mean()
                #mlm_loss = mlm_loss.mean()

                cl_loss = 10.0 * cl_loss
                #mlm_loss = 0.1 * mlm_loss

                loss, logits = mil_loss

                list_loss.append(loss.item())

                score, pred = logits.max(-1)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0

                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                #avg_mlm_loss.update(mlm_loss.item(), 1)
                avg_cl_loss.update(cl_loss.item(), 1)
                avg_pos_acc.update(pos_acc, 1)

                t.set_postfix(mil_loss=avg_loss.avg, cl_loss=avg_cl_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg, alpha=alpha, t_c=threshold_clean)

                loss = loss + alpha * cl_loss
                loss = loss / self.args.grad_acc_steps
                loss.backward()

                if self.global_steps % self.args.grad_acc_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.global_steps % self.args.save_steps == 0:
                    self.logger.info("=== steps {} record start ===".format(self.global_steps))
                    result = self.eval_model(self.test_loader)
                    p = result['prec']
                    self.logger.info("auc: %.4f f1: %.4f p@100: %.4f p@200: %.4f p@300: %.4f p@500: %.4f p@1000: %.4f p@2000: %.4f" % (
                        result['auc'], result['f1'], p[100], p[200], p[300], p[500], p[1000], p[2000]))
                    list_test_auc.append(result['auc'])

                    if result['auc'] > best_auc:
                        self.logger.info("auc@best: %.4f auc: %.4f best ckpt and save to %s" % (best_auc, result['auc'], self.ckpt))

                        torch.save(self.model.state_dict(), self.ckpt)
                        best_auc = result['auc']
                        best_f1 = result['f1']
                        best_p100 = p[100]
                        best_p200 = p[200]
                        best_p300 = p[300]
                    else:
                        self.logger.info("auc@best: %.4f auc: %.4f no improvement and skip" % (best_auc, result['auc']))

                    self.model.train()

                    self.logger.info("mil_loss=%.4f cl_loss=%.4f acc==%.4f pos_acc=%.4f" % (avg_loss.avg, avg_cl_loss.avg, avg_acc.avg, avg_pos_acc.avg))
                    self.logger.info("=== steps {} record end ===".format(self.global_steps))

                if self.global_steps == self.args.max_steps:
                    break

            # runtime
            end_time = time.time()
            epoch_time = end_time - start_time
            run_time += epoch_time

            # writer
            if self.writer is not None:
                self.writer.add_scalar('train/loss', avg_loss.avg, epoch)
                self.writer.add_scalar('train/acc', avg_acc.avg, epoch)
                self.writer.add_scalar('train/pos_acc', avg_pos_acc.avg, epoch)
                self.writer.add_scalar('train/run_time', run_time, epoch)
            
            
            self.logger.info("=== epoch %d train end time: %ds avg epoch time: %ds run time: %ds ===" % (epoch, epoch_time, run_time / epoch, run_time))
            

            if self.global_steps == self.args.max_steps:
                break
        self.logger.info("auc@best on eval set: %f" % (best_auc))

    def eval_model(self, eval_loader):
        self.model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            t = tqdm(eval_loader)
            pred_result = []
            for _, data in enumerate(t):
                for i in range(len(data)):
                    try:
                        data[i] = data[i].to(self.device)
                    except:
                        pass

                label, bag_name, scope, bag, args = data[0], data[1], data[2], data[3], data[4:]
                logits = self.model(None, scope, *args, train=False, bag_size=0)

                bag_name = torch.LongTensor(bag_name).to(self.device)



                logits = logits.cpu().numpy()
                label = label.cpu().numpy()
                bag_name = bag_name.cpu().numpy()

                for i in range(len(logits)):
                    for relid in range(self.model.num_class):
                        if relid != 0:
                            pred_result.append({
                                'entpair': bag_name[i][:2],
                                'relation': self.model.id2rel[relid],
                                'score': logits[i][relid]
                            })
            result = eval_loader.dataset.eval(pred_result)
        return result

    def load_model(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(state_dict)

    def print(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)
