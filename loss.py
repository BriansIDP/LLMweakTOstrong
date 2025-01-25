import torch

class LossFnBase:
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        This function calculates the loss between logits and labels.
        """
        raise NotImplementedError


class product_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for product of predictions and labels.

    Attributes:
    alpha: A float indicating how much to weigh the weak model.
    beta: A float indicating how much to weigh the strong model.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        alpha: float = 1.0,  # how much to weigh the weak model
        beta: float = 1.0,  # how much to weigh the strong model
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.alpha = alpha
        self.beta = beta
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
    ) -> torch.Tensor:
        preds = torch.softmax(logits, dim=-1)
        target = torch.pow(preds, self.beta) * torch.pow(labels, self.alpha)
        target /= target.sum(dim=-1, keepdim=True)
        target = target.detach()
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        return loss.mean()


class logconf_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for log confidence.

    Attributes:
    aux_coef: A float indicating the auxiliary coefficient.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.float()
        # Size: [batch_size, seq_len, tokenizer_size]
        lossmask = (labels == -1)
        # [bs, sl]
        labels = labels.masked_fill(lossmask, 0)
        labels = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()
        # [bs, sl, ts]
        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        preds = torch.log_softmax(logits, dim=-1)
        # pred_entropy = -preds * (- torch.exp(preds) * preds).sum(dim=-1)
        pred_entropy = (- labels * preds).sum(dim=-1) * (~lossmask)
        pred_entropy = pred_entropy.sum(-1) / (~lossmask).sum(dim=-1)
        threshold_mask = pred_entropy > values
        # Threshold_mask: true or false. uttrance level
        lossmask = lossmask | ~threshold_mask.unsqueeze(1)
        maxonehot = torch.nn.functional.one_hot(preds.max(dim=-1)[1], num_classes=logits.size(-1)).float()
        # target = labels * (1 - coef) + maxonehot.detach() * coef
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1, logits.size(-1)),
            reduction="none",
        )
        loss = loss.masked_fill(lossmask.view(-1), 0)
        # loss = loss.sum() / (~lossmask).sum()
        # return loss
        if lossmask.all():
        # if all loss is masked, return 0 as loss
            return loss.sum()
        else:
            return loss.sum() / (~lossmask).sum()


class logconf_step_loss_fn(LossFnBase):

    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
    ) -> torch.Tensor:
        
        logits = logits.float()
        # Size: [batch_size, seq_len, tokenizer_size]
        lossmask = (labels == -1)
        # [bs, sl]
        # labels = labels.masked_fill(lossmask, 0)
        # labels = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()
        # [bs, sl, ts]
        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        
        labels = labels.masked_fill(lossmask, 0)
        labels = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()

        strong_preds = torch.argmax(logits, dim=-1)
        strong_preds[lossmask] = 0
        strong_preds = torch.nn.functional.one_hot(strong_preds, num_classes=logits.size(-1)).float()
        # print("strong_preds", strong_preds.shape)
        targets = labels * (1 - coef) + strong_preds.detach() * coef
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1, logits.size(-1)), 
            reduction="none",
        )
        loss = loss.masked_fill(lossmask.view(-1), 0)
        loss = loss.sum() / (~lossmask).sum()

        return loss
    

class logconf_confer_loss_fn(LossFnBase):

    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
    ) -> torch.Tensor:
        
        logits = logits.float()
        # Size: [batch_size, seq_len, tokenizer_size]
        lossmask = (labels == -1)
        # [bs, sl]
        labels = labels.masked_fill(lossmask, 0)
        labels = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()
        # [bs, sl, ts] one-hot
        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        preds = torch.log_softmax(logits, dim=-1)
        # pred_entropy = -preds * (- torch.exp(preds) * preds).sum(dim=-1)
        pred_entropy = (- labels * preds).sum(dim=-1) * (~lossmask)
        pred_entropy = pred_entropy.sum(-1) / (~lossmask).sum(dim=-1)
        threshold_mask = pred_entropy > values
        loss_mask_weak = lossmask | ~threshold_mask.unsqueeze(1)
        loss_mask_strong = lossmask | threshold_mask.unsqueeze(1)

        strong_preds = torch.argmax(logits, dim=-1).detach()
        strong_preds[lossmask] = 0
        strong_preds = torch.nn.functional.one_hot(strong_preds, num_classes=logits.size(-1)).float()

        loss_weak = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1, logits.size(-1)),
            reduction="none",
        )
        loss_weak = loss_weak.masked_fill(loss_mask_weak.view(-1), 0)
        loss_strong = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            strong_preds.view(-1, logits.size(-1)),
            reduction="none",
        )
        loss_strong = loss_strong.masked_fill(loss_mask_strong.view(-1), 0)

        loss = loss_weak + loss_strong * coef
        # loss = loss_weak + loss_strong
        loss = loss.sum() / (~lossmask).sum()
        return loss


class soft_kl_loss_fn(LossFnBase):
    """
    Attributes:
    aux_coef: A float indicating the auxiliary coefficient.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
        token_score: torch.Tensor,
    ) -> torch.Tensor:
        num_class = logits.size(-1)
        logits = logits.float().view(-1, num_class)
        labels = labels.reshape(-1)
        lossmask = (labels == -1)
        token_score = token_score.exp().view(-1, 1).to(labels.device)

        logits = logits[lossmask == False]
        labels = labels[lossmask == False]
        assert len(token_score) == len(labels)
        
        # 其他category的prob平铺
        # onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        # onehot_num = token_score - (1.0-token_score)/(num_class-1)
        # onehot_label *= onehot_num
        # targets = ((1.0-token_score)/(num_class-1)).expand(labels.size(0), num_class) + onehot_label

        # 在之前的基础上放缩
        prob = torch.softmax(logits, dim=-1)
        label_onehot = torch.nn.functional.one_hot(labels, num_classes=num_class)
        score_before = prob[label_onehot==1].view(-1, 1)
        scale = (1-token_score) / (1-score_before)
        targets = prob * scale
        targets[label_onehot==1] = token_score.view(1, -1)
        
        log_prob = torch.log_softmax(logits, dim=-1)
        loss = torch.nn.functional.kl_div(
            input=log_prob,
            target=targets,
            reduction="none",
            log_target=False,
        ).sum(-1)
        
        return loss.mean()


class soft_step_conf_loss_fn(LossFnBase):
    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.25,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
        token_score: torch.Tensor,
    ) -> torch.Tensor:
        num_class = logits.size(-1)
        logits = logits.float().view(-1, num_class)
        labels = labels.reshape(-1)
        lossmask = (labels == -1)
        token_score = token_score.exp().view(-1, 1).to(labels.device)
        # labels = labels.masked_fill(lossmask, 0).view(-1)
        # labels = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float().view(-1, logits.size(-1))
        # [bs, sl, ts]
        logits = logits[lossmask == False]
        labels = labels[lossmask == False]
        assert len(token_score) == len(labels)

        # onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        # onehot_num = token_score - (1.0-token_score)/(num_class-1)
        # onehot_label *= onehot_num
        # targets = ((1.0-token_score)/(num_class-1)).expand(labels.size(0), num_class) + onehot_label
        
        prob = torch.softmax(logits, dim=-1)
        label_onehot = torch.nn.functional.one_hot(labels, num_classes=num_class)
        score_before = prob[label_onehot==1].view(-1, 1)
        scale = (1-token_score) / (1-score_before)
        targets = prob * scale
        targets[label_onehot==1] = token_score.view(1, -1)

        strong_preds = torch.argmax(logits, dim=-1)
        strong_preds = torch.nn.functional.one_hot(strong_preds, num_classes=logits.size(-1)).float()
        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        targets = targets * (1 - coef) + strong_preds.detach() * coef
        
        log_prob = torch.log_softmax(logits, dim=-1)
        loss = torch.nn.functional.kl_div(
            input=log_prob,
            target=targets,
            reduction="none",
            log_target=False,
        ).sum(-1)
        
        return loss.mean()


class soft_confer_loss_fn(LossFnBase):
    def __init__(
        self,
        aux_coef: float = 1,
        warmup_frac: float = 0.75,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
        token_score: torch.Tensor,
    ) -> torch.Tensor:
        num_class = logits.size(-1)
        logits = logits.float().view(-1, num_class)
        labels = labels.reshape(-1)
        lossmask = (labels == -1)
        token_score = token_score.exp().view(-1, 1).to(labels.device)
        logits = logits[lossmask == False]
        labels = labels[lossmask == False]
        assert len(token_score) == len(labels)

        pred_score, _ = torch.softmax(logits, dim=-1).max(dim=-1, keepdim=True)
        threshold_mask = pred_score > token_score
        # True if strong is more confident than weak.

        onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        onehot_num = token_score - (1.0-token_score)/(num_class-1)
        onehot_label *= onehot_num
        targets = ((1.0-token_score)/(num_class-1)).expand(labels.size(0), num_class) + onehot_label

        strong_preds = torch.argmax(logits, dim=-1).detach()
        strong_preds = torch.nn.functional.one_hot(strong_preds, num_classes=logits.size(-1)).float()
        
        log_prob = torch.log_softmax(logits, dim=-1)
        weak_loss = torch.nn.functional.kl_div(
            input=log_prob,
            target=targets,
            reduction="none",
            log_target=False,
        ).sum(-1)
        weak_loss = weak_loss.masked_fill(threshold_mask.view(-1), 0)
        # strong_loss = torch.nn.functional.cross_entropy(
        #     input=logits,
        #     target=strong_preds,
        #     reduction="none",
        # ).sum(-1)
        strong_loss = torch.nn.functional.cross_entropy(
            input=logits,
            target=strong_preds,
            reduction="none",
        )
        strong_loss = strong_loss.masked_fill(~threshold_mask.view(-1), 0)

        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        loss = weak_loss + strong_loss * coef
        
        return loss.mean()


def dir_loss(outputs, targets, strp_frac=0, epsilon=1e-6, reduction="mean", alpha=0.5):
    '''
    outputs: output logits in shape of Tensor([batch_size*seq_len, tokenizer_size]), dtype=float32
    targets: one-hot tensor in shape of Tensor([batch_size*seq_len, tokenizer_size]), dtype=float32
    '''
    batch_size = targets.shape[0]
    n_class = targets.shape[1]

    alphas = torch.exp(outputs) + epsilon
    # targets = torch.relu(targets - epsilon * (n_class+1)) + epsilon

    dir_logprob = (torch.xlogy(alphas - 1.0, targets).sum(-1)
            + torch.lgamma(alphas.sum(-1))
            - torch.lgamma(alphas).sum(-1))
    
    kl_div = torch.nn.functional.kl_div(
        input=torch.log_softmax(outputs, dim=-1),
        target=targets,
        reduction="none"
    ).sum(-1)

    dir_logprob = torch.clamp(dir_logprob, min=-1e8, max=1e8)

    loss = -dir_logprob * 1e-2 + kl_div
    # loss = kl_div

    return loss


class DirLossFn(LossFnBase):
    def __init__(
        self,
        alpha: float = 0.5,
    ):
        self.alpha = alpha

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
        token_score: torch.Tensor,
    ) -> torch.Tensor:
        
        num_class = logits.size(-1)
        logits = logits.float().view(-1, num_class)
        labels = labels.reshape(-1)
        lossmask = (labels == -1)
        token_score = token_score.exp().view(-1, 1).to(labels.device)

        logits = logits[lossmask == False]
        labels = labels[lossmask == False]
        assert len(token_score) == len(labels)
        
        onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        onehot_num = token_score - (1.0-token_score)/(num_class-1)
        onehot_label *= onehot_num
        targets = ((1.0-token_score)/(num_class-1)).expand(labels.size(0), num_class) + onehot_label

        # [bs, sl, ts]
        loss = dir_loss(
            logits,
            targets,
            reduction="none",
        )
        loss = loss.mean()

        return loss


class Dir_logconf_loss_fn(LossFnBase):
    """
    This class defines a custom loss function for log confidence.

    Attributes:
    aux_coef: A float indicating the auxiliary coefficient.
    warmup_frac: A float indicating the fraction of total training steps for warmup.
    """

    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
        alpha: float = 0.5,
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac
        self.alpha = alpha

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.float()
        # Size: [batch_size, seq_len, tokenizer_size]
        lossmask = (labels == -1)
        # [bs, sl]
        labels = labels.masked_fill(lossmask, 0)
        labels = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()
        # [bs, sl, ts]
        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        preds = torch.log_softmax(logits, dim=-1)
        # pred_entropy = -preds * (- torch.exp(preds) * preds).sum(dim=-1)
        pred_entropy = (- labels * preds).sum(dim=-1) * (~lossmask)
        pred_entropy = pred_entropy.sum(-1) / (~lossmask).sum(dim=-1)
        threshold_mask = pred_entropy > values
        # Threshold_mask: true or false. uttrance level
        lossmask = lossmask | ~threshold_mask.unsqueeze(1)
        # target = labels * (1 - coef) + maxonehot.detach() * coef
        loss = dir_loss(
            logits.view(-1, logits.size(-1)),
            labels.view(-1, logits.size(-1)),
            reduction="none",
            alpha=self.alpha,
        )
        loss = loss.masked_fill(lossmask.view(-1), 0)
        if lossmask.all():
        # if all loss is masked, return 0 as loss
            return loss.sum()
        else:
            return loss.sum() / (~lossmask).sum()
        

def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, num_classes, step_frac, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(step_frac, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    # kl_alpha = alpha * (1 - y) + 1
    # kl_alpha = alpha * (1 - y)
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device)

    kl_div = torch.clamp(kl_div, min=-2e5, max=2e5)
    
    return A + kl_div * 1e-4
    # return A, kl_div
    # return A


def edl_log_loss(output, target, step_frac, num_classes=2):
    device = output.device
    # evidence = F.relu(output)
    # evidence = torch.nn.functional.softmax(output, dim=1)
    evidence = torch.exp(output)
    # evidence = torch.nn.functional.softplus(output)
    alpha = evidence + 1
    # alpha = evidence
    # loss = torch.mean(
    #     edl_loss(
    #         torch.log, target, alpha, num_classes, step_frac, device
    #     )
    # )
    loss = edl_loss(torch.log, target, alpha, num_classes, step_frac, device)
    return loss.squeeze()
    # xent, kl_div = edl_loss(torch.log, target, alpha, num_classes, step_frac, device)
    # return xent.squeeze(), kl_div.squeeze()


class edl_log_loss_fn(LossFnBase):
    def __init__(
        self,
        one_hot: bool=False,
    ):
        self.one_hot = one_hot

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
        token_score: torch.Tensor,
    ) -> torch.Tensor:
        
        num_class = logits.size(-1)
        logits = logits.float().view(-1, num_class)
        labels = labels.reshape(-1)
        lossmask = (labels == -1)
        token_score = token_score.exp().view(-1, 1).to(labels.device)

        logits = logits[lossmask == False]
        labels = labels[lossmask == False]
        assert len(token_score) == len(labels)

        # targets = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()
        
        # onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        # onehot_num = token_score - (1.0-token_score)/(num_class-1)
        # onehot_label *= onehot_num
        # targets = ((1.0-token_score)/(num_class-1)).expand(labels.size(0), num_class) + onehot_label

        # onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        # targets = onehot_label * token_score
        if self.one_hot:
            targets = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()
        else:
            prob = torch.softmax(logits, dim=-1)
            label_onehot = torch.nn.functional.one_hot(labels, num_classes=num_class)
            score_before = prob[label_onehot==1].view(-1, 1)
            scale = (1-token_score) / (1-score_before)
            targets = prob * scale
            targets[label_onehot==1] = token_score.view(1, -1)

        # loss = edl_log_loss(logits, labels, step_frac)
        loss = edl_log_loss(
            output=logits,
            target=targets,
            step_frac=step_frac,
            num_classes=num_class,
        )

        loss = loss.mean()

        return loss

        # xent, kl_div = edl_log_loss(
        #     output=logits.view(-1, num_classes),
        #     target=labels.view(-1, num_classes),
        #     step_frac=step_frac,
        #     num_classes=num_classes,
        # )
        # xent = xent.masked_fill(lossmask.view(-1), 0)
        # kl_div = kl_div.masked_fill(lossmask.view(-1), 0)
        # xent = xent.sum() / (~lossmask).sum()
        # kl_div = kl_div.sum() / (~lossmask).sum()
        # return xent, kl_div


class edl_logconf_loss_fn(LossFnBase):
    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.1,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
        token_score: torch.Tensor,
    ) -> torch.Tensor:
        num_class = logits.size(-1)
        logits = logits.float().view(-1, num_class)
        labels = labels.reshape(-1)
        lossmask = (labels == -1)
        token_score = token_score.exp().view(-1, 1).to(labels.device)

        logits = logits[lossmask == False]
        labels = labels[lossmask == False]
        assert len(token_score) == len(labels)

        pred_score, _ = torch.softmax(logits, dim=-1).max(dim=-1, keepdim=True)
        threshold_mask = pred_score > token_score
        
        onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        onehot_num = token_score - (1.0-token_score)/(num_class-1)
        onehot_label *= onehot_num
        targets = ((1.0-token_score)/(num_class-1)).expand(labels.size(0), num_class) + onehot_label
        
        loss = edl_log_loss(
            output=logits,
            target=targets,
            step_frac=step_frac,
            num_classes=num_class,
        )

        loss = loss.masked_fill(threshold_mask.view(-1), 0)
        if threshold_mask.all():
            return loss.sum()
        else:
            return loss.sum() / (~threshold_mask).sum()

        

class edl_logconf_step_loss_fn(LossFnBase):

    def __init__(
        self,
        aux_coef: float = 0.25,
        warmup_frac: float = 0.25,  # in terms of fraction of total training steps
        one_hot: bool=False,
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac
        self.one_hot = one_hot

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
        token_score: torch.Tensor,
    ) -> torch.Tensor:
        
        num_class = logits.size(-1)
        logits = logits.float().view(-1, num_class)
        labels = labels.reshape(-1)
        lossmask = (labels == -1)
        token_score = token_score.exp().view(-1, 1).to(labels.device)
        
        logits = logits[lossmask == False]
        labels = labels[lossmask == False]
        assert len(token_score) == len(labels)

        # onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        # onehot_num = token_score - (1.0-token_score)/(num_class-1)
        # onehot_label *= onehot_num
        # targets = ((1.0-token_score)/(num_class-1)).expand(labels.size(0), num_class) + onehot_label

        # onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        # targets = onehot_label * token_score
        if self.one_hot:
            onehot_label = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()
            targets = onehot_label * token_score
        else:
            prob = torch.softmax(logits, dim=-1)
            label_onehot = torch.nn.functional.one_hot(labels, num_classes=num_class)
            score_before = prob[label_onehot==1].view(-1, 1)
            scale = (1-token_score) / (1-score_before)
            targets = prob * scale
            targets[label_onehot==1] = token_score.view(1, -1)

        # targets = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()

        strong_preds = torch.argmax(logits, dim=-1)
        strong_preds = torch.nn.functional.one_hot(strong_preds, num_classes=logits.size(-1)).float().detach()
        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        targets = targets * (1 - coef) + strong_preds.detach() * coef
        
        log_prob = torch.log_softmax(logits, dim=-1)
        loss = edl_log_loss(
            output=logits,
            target=targets,
            step_frac=step_frac,
            num_classes=num_class,
        )

        # weak_loss = edl_log_loss(logits, targets, step_frac=step_frac, num_classes=num_class)
        # strong_loss = torch.nn.functional.cross_entropy(
        #     input=logits,
        #     target=strong_preds,
        #     reduction="none",
        # )
        # loss = weak_loss * (1-coef) + strong_loss * coef
        
        return loss.mean()


class edl_logconf_confer_loss_fn(LossFnBase):

    def __init__(
        self,
        aux_coef: float = 1.5,
        warmup_frac: float = 0.5,  # in terms of fraction of total training steps
    ):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step_frac: float,
        values: torch.Tensor,
        token_score: torch.Tensor,
    ) -> torch.Tensor:

        num_class = logits.size(-1)
        logits = logits.float().view(-1, num_class)
        labels = labels.reshape(-1)
        lossmask = (labels == -1)
        token_score = token_score.exp().view(-1, 1).to(labels.device)
        logits = logits[lossmask == False]
        labels = labels[lossmask == False]
        assert len(token_score) == len(labels)

        pred_score, _ = torch.softmax(logits, dim=-1).max(dim=-1, keepdim=True)
        threshold_mask = pred_score > token_score
        # True if strong is more confident than weak.

        onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        onehot_num = token_score - (1.0-token_score)/(num_class-1)
        onehot_label *= onehot_num
        targets = ((1.0-token_score)/(num_class-1)).expand(labels.size(0), num_class) + onehot_label

        # onehot_label = torch.nn.functional.one_hot(labels, num_classes=num_class).float()
        # targets = onehot_label * token_score

        # prob = torch.softmax(logits, dim=-1)
        # label_onehot = torch.nn.functional.one_hot(labels, num_classes=num_class)
        # score_before = prob[label_onehot==1].view(-1, 1)
        # scale = (1-token_score) / (1-score_before)
        # targets = prob * scale
        # targets[label_onehot==1] = token_score.view(1, -1)

        strong_preds = torch.argmax(logits, dim=-1).detach()
        strong_preds = torch.nn.functional.one_hot(strong_preds, num_classes=num_class).float()
        
        # log_prob = torch.log_softmax(logits, dim=-1)
        weak_loss = edl_log_loss(
            output=logits,
            target=targets,
            step_frac=step_frac,
            num_classes=num_class,
        )
        weak_loss = weak_loss.masked_fill(threshold_mask.view(-1), 0)
        strong_loss = torch.nn.functional.cross_entropy(
            input=logits,
            target=strong_preds,
            reduction="none",
        )

        strong_loss = strong_loss.masked_fill(~threshold_mask.view(-1), 0)

        coef = 1.0 if step_frac > self.warmup_frac else step_frac
        coef = coef * self.aux_coef
        loss = weak_loss + strong_loss * coef
        
        return loss.mean()