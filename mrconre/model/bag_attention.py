from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class BagAttention(nn.Module):
    def __init__(self, sentence_encoder, num_class, rel2id, mil='att', hparams=None):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc_front = nn.Sequential(
            nn.Linear(self.sentence_encoder.hidden_size, self.sentence_encoder.hidden_size // 2),
            nn.ReLU(True),
            nn.Linear(self.sentence_encoder.hidden_size // 2, self.sentence_encoder.hidden_size // 2),
            nn.ReLU(True)
        )
        self.fc = nn.Linear(self.sentence_encoder.hidden_size // 2, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        self.criterion = nn.CrossEntropyLoss()
        self.sup_criterio = SupConLoss()
        self.mil = mil
        self.theta = 0.001
        if hparams is None:
            self.hparams = {
                'temperature': 0.05
            }
        else:
            self.hparams = hparams
        for rel, id in rel2id.items():
            self.id2rel[id] = rel


    def infer(self, bag):
        pass

    def kl_div(self, p, q):
        # p, q is in shape (batch_size, n_classes)
        return (p * p.log2() - p * q.log2()).sum(dim=1)

    def symmetric_kl_div(self,p, q):
        return self.kl_div(p, q) + self.kl_div(q, p)

    def js_div(self,p, q):
        # Jensen-Shannon divergence, value is in (0, 1)
        m = 0.5 * (p + q)
        return 0.5 * self.kl_div(p, m) + 0.5 * self.kl_div(q, m)

    def cl(self, rep, aug_rep, bag_rep):
        # rep: (B, bag, H)
        # aug_rep: (B, bag, H)
        # bag_rep: (B, H)
        temperature = self.hparams.temperature
        # (B, bag, H)
        batch_size, bag_size, hidden_size = rep.size()
        #aug_rep = aug_rep.view(batch_size, bag_size, hidden_size)
        # positive pairs
        # instance ~ augmented instance
        # (B, bag, H) ~ (B, bag, H) - (B, bag)
        pos_sim = F.cosine_similarity(rep, aug_rep, dim=-1)
        pos_sim = torch.exp(pos_sim / temperature)
        # negative pairs
        # instance ~ other bag representation
        # (B, H) - (B, bag, H)
        tmp_bag_rep = bag_rep.unsqueeze(1).repeat(1, bag_size, 1)
        # each instance ~ its own bag representation
        axis_sim = F.cosine_similarity(rep, tmp_bag_rep, dim=-1) # (B, bag)
        # axis_sim = axis_sim.unsqueeze(-1) # (B, bag, 1)
        tmp_bag_rep = bag_rep.unsqueeze(0).repeat(batch_size, 1, 1) # (B, B, H)
        # (B, bag, H) ~ (B, B, H) - (B, bag, B)
        tmp_rep = rep.permute((1, 2, 0)) # (bag, H, B)
        tmp_bag_rep = tmp_bag_rep.permute((1, 2, 0)) # (B, H, B)
        tmp_bag_rep = tmp_bag_rep.unsqueeze(1) # (B, 1, H, B)
        # (bag, H, B) ~ (B, 1, H, B) - (B, bag, B)
        pair_sim = F.cosine_similarity(tmp_rep, tmp_bag_rep, dim=-2) # (B, bag, B)
        # bug sum(2) ? any effect ?
        neg_sim = torch.exp(pair_sim / temperature).sum(2) - torch.exp(axis_sim / temperature)
        pos_sim = pos_sim.view(-1)
        neg_sim = neg_sim.view(-1)
        loss = -1.0 * torch.log(pos_sim / (pos_sim + neg_sim))

        return loss

    
    def mixup_criterion(self, y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b) 
    
    def linear_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

    def semiLoss(self, outputs_x, targets_x, outputs_u, targets_u, epoch=1):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
    
        return Lx, Lu

    def _cal_mixup_loss(self, token, mask, pos1, pos2, \
        noisy_mask=None, \
        targets_x=None, clean_bag_rep=None):

        Lx, Lu = None, None
        
        
        token_u = torch.index_select(token, 0, noisy_mask)
        mask_u = torch.index_select(mask, 0, noisy_mask)
        pos1_u = torch.index_select(pos1, 0, noisy_mask)
        pos2_u = torch.index_select(pos2, 0, noisy_mask)

        with torch.no_grad():
            rep_u, _ = self.sentence_encoder(token_u, mask_u, pos1_u, pos2_u)
            
            outputs_u = self.fc(self.fc_front(self.drop(rep_u)))
            outputs_u2 = self.fc(self.fc_front(self.drop(rep_u)))

            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/self.hparams.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
        
        inputs_x = clean_bag_rep
        inputs_u, _ = self.sentence_encoder(token_u, mask_u, pos1_u, pos2_u)
        
        batch_size = targets_x.size(0)
        targets_x = torch.zeros(batch_size, self.num_class).cuda().scatter_(1, targets_x.view(-1,1).long(), 1)

        # # mixup
        # all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
        # all_targets = torch.cat([targets_x, targets_u], dim=0)

        # l = np.random.beta(self.hparams.alpha, self.hparams.alpha)

        # l = max(l, 1-l)
        # idx = torch.randperm(all_inputs.size(0))
      

        # input_a, input_b = all_inputs, all_inputs[idx]
        # target_a, target_b = all_targets, all_targets[idx]
        l = np.random.beta(self.hparams.alpha, self.hparams.alpha)
        l = max(l, 1-l)

        idx = torch.randint(0, len(targets_u), (len(targets_x),))
        input_a, input_b = inputs_x, inputs_u[idx]
        target_a, target_b = targets_x, targets_u[idx]


        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        
        logits = self.fc(  self.fc_front(mixed_input)  )
        # logits_x = logits[:batch_size]
        # logits_u = logits[batch_size:]

        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        # Lx, Lu = self.semiLoss(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:])
    
        # return Lx, Lu
        return Lx


    
    def forward(self, label, scope, arg1, arg2, arg3, arg4, arg5=None, arg6=None, arg7=None, arg8=None, train=True, bag_size=0, \
        threshold_clean = None, mixmatch = False, epoch = None):
        if bag_size > 0:
            flat = lambda x: x.view(-1, x.size(-1))
            arg1, arg2, arg3, arg4 = flat(arg1), flat(arg2), flat(arg3), flat(arg4)
        else:
            begin, end = scope[0][0], scope[-1][1]
            flat = lambda x: x[:, begin:end, :].view(-1, x.size(-1))
            arg1, arg2, arg3, arg4 = flat(arg1), flat(arg2), flat(arg3), flat(arg4)
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))

        rep, mlm_loss = self.sentence_encoder(arg1, arg2, arg3, arg4, mlm=False)
        #pattern, _ = self.sentence_encoder(arg5, arg6, arg7, arg8, mlm=False)
        with torch.no_grad():
            rep_for_att = self.fc_front(rep)

        items = []
        if train:
            items.append(mlm_loss)
            
            batch_size = label.size(0)
            query = label.unsqueeze(1) # (B, 1)
            att_mat = self.fc.weight.data[query] # (B, 1, H)
            rep_for_att = rep_for_att.view(batch_size, bag_size, -1)
            att_score = (rep_for_att * att_mat).sum(-1) # (B, bag)
            softmax_att_score = self.softmax(att_score) # (B, bag)

            rep = rep.view(batch_size, bag_size, -1)
            bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(1) # (B, bag, 1) * (B, bag, H) -> (B, bag, H) -> (B, H)

            assert arg5 is not None
            flat = lambda x: x.view(-1, x.size(-1))
            arg5, arg6, arg7, arg8 = flat(arg5), flat(arg6), flat(arg7), flat(arg8)
            pattern, _ = self.sentence_encoder(arg5, arg6, arg7, arg8, mlm=False)
            pattern_aug, _ = self.sentence_encoder(arg5, arg6, arg7, arg8, mlm=False)

            pattern = pattern.view(batch_size, bag_size, -1)
            pattern_aug = pattern_aug.view(batch_size, bag_size, -1)
            #att_score_ = (pattern * att_mat).sum(-1)
            #softmax_att_score_ = self.softmax(att_score_)
            #pattern_rep = (softmax_att_score_.unsqueeze(-1) * pattern).sum(1)
            #pattern_rep_repeat = pattern_rep.unsqueeze(1).repeat(1, bag_size, 1)
            #pattern_logits = self.fc(pattern_rep_repeat)
            prob_clean = None
            with torch.no_grad():
                _probs = self.softmax(self.fc(self.fc_front(rep))) # (B, bag, num)
                _targets =  self.softmax(self.fc(self.fc_front(pattern))) # (B, num)
                #_targets =  self.softmax(pattern_logits) # (B, num)
                _probs = _probs.view(batch_size*bag_size, -1)
                _targets = _targets.view(batch_size*bag_size, -1)
                js_value = self.js_div(_probs, _targets).view(batch_size, bag_size) # (B, bag)
                prob_clean = 1.0 - js_value
                
            clean_idx = torch.ge(prob_clean, threshold_clean)
            
            cl_loss = None
            cl_loss_a = self.cl(rep, pattern, bag_rep)
            
            pattern = torch.stack([pattern.sum(1)/3,pattern_aug.sum(1)/3],dim=1)
            cl_loss_b = self.sup_criterio(pattern, label)
            # cl_loss_b = self.cl(pattern, pattern_aug, pattern.index_select(1, torch.arange(1).cuda()).squeeze(1).cuda())     
            
            cl_weight = []
            clean_num = 0
            for i in range(len(clean_idx)):
                temp = []
                count = 0
                for j in range(len(clean_idx[0])):
                    if clean_idx[i][j]:
                        count += 1.0
                        temp.append(1.0)
                    else:
                        temp.append(0.0)
                clean_num += count
                if count == 0:
                    w = 1.0
                else:
                    w = 3.0 / count
                for ele in temp:
                    cl_weight.append(ele * w)
            cl_loss_a = torch.mul(cl_loss_a, (torch.Tensor(cl_weight)).cuda())
            # cl_loss_b = torch.mul(cl_loss_b, (torch.Tensor(cl_weight)).cuda())
            cl_loss = cl_loss_a.mean() + self.theta*cl_loss_b.mean()
            # cl_loss = cl_loss_a.mean()
            
            items.append(cl_loss)

            if mixmatch and clean_num < batch_size * 3:
                noisy_idx = torch.lt(prob_clean, threshold_clean)
                noisy_mask = noisy_idx.view(-1)
                noisy_mask = noisy_mask.long()
                noisy_mask = torch.nonzero(noisy_mask).view(-1)
                
                Lx = self._cal_mixup_loss(arg1, arg2, arg3, arg4,\
                    noisy_mask, \
                    label, bag_rep
                    )

                a = epoch - self.hparams.mixmatch_begin_epoch + 1
                b = self.hparams.max_epoch - self.hparams.mixmatch_begin_epoch + 1
 
                # mil_loss = Lx + self.hparams.lambda_u * self.linear_rampup(a, b) * Lu
                mil_loss = Lx
            else: 
                # mixup
                lam = np.random.beta(0.5, 0.5)
                index = torch.randperm(batch_size).cuda()  
                bag_rep_mixup = lam * bag_rep + (1 - lam) * bag_rep[index,:]   
                y_a, y_b = label, label[index]    
                bag_rep_mixup = self.fc_front(bag_rep_mixup)
                logits = self.fc(bag_rep_mixup) # (B, N)
                loss_func = self.mixup_criterion(y_a, y_b, lam)
                mil_loss = loss_func(self.criterion, logits)
            
            with torch.no_grad():
                bag_logits = self.fc(self.fc_front(bag_rep))

            bag_logits = [mil_loss, bag_logits]

        else:
            if bag_size == 0:
                bag_logits = []
                att_score = torch.matmul(rep_for_att, self.fc.weight.data.transpose(0, 1)) # (nsum, H) * (H, N) -> (nsum, N)
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1)) # (N, n) num_labels
                    rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat) # (N, n) * (n, H) -> (N, H)
                    rep_for_each_rel = self.fc_front(rep_for_each_rel)
                    logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel)) # ((each rel)N, (logit)N)
                    logit_for_each_rel = logit_for_each_rel.diag() # (N)
                    bag_logits.append(logit_for_each_rel)
                bag_logits = torch.stack(bag_logits, 0) # after **softmax**
            else:
                batch_size = rep.size(0) // bag_size
                att_score = torch.matmul(rep, self.fc.weight.data.transpose(0, 1)) # (nsum, H) * (H, N) -> (nsum, N)
                att_score = att_score.view(batch_size, bag_size, -1) # (B, bag, N)
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                softmax_att_score = self.softmax(att_score.transpose(1, 2)) # (B, N, (softmax)bag)
                rep_for_each_rel = torch.matmul(softmax_att_score, rep) # (B, N, bag) * (B, bag, H) -> (B, N, H)
                bag_logits = self.softmax(self.fc(rep_for_each_rel)).diagonal(dim1=1, dim2=2) # (B, (each rel)N)

        items.append(bag_logits)

        items = items if len(items) > 1 else items[0]
        return items

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss