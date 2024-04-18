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
        # lossmask *= threshold_mask.unsqueeze(1)
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
            return loss.sum()
        else:
            return loss.sum() / (~lossmask).sum()


class logconf_step_loss_fn(LossFnBase):

    def __init__(
        self,
        aux_coef: float = 0.5,
        warmup_frac: float = 0.05,  # in terms of fraction of total training steps
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
    

class logconf_stepmask_loss_fn(LossFnBase):
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
        # lossmask *= threshold_mask.unsqueeze(1)
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
            return loss.sum() / lossmask.sum()
        else:
            return loss.sum() / (~lossmask).sum()