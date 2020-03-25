import torch


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

class MultipleChoiceLossCompute:
    "A Loss compute and train function for multiple choice tasks."

    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef = lm_coef
        self.opt = opt

    def __call__(self, X, Y, M, clf_logits, lm_logits=None, only_return_losses=False):
        # Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:, :, 1:, 0].contiguous().view(-1)  # Shape: 252
            M = M.view(-1, M.size(2))
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            lm_losses = lm_losses.view(X.size(0) * X.size(1), X.size(2) - 1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        clf_losses = self.clf_criterion(clf_logits, Y)
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None else clf_losses

        if self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() + self.lm_coef * lm_losses.sum()
        else:
            train_loss = clf_losses.sum()
        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))





class ClassificationLossCompute:
    "A Loss compute and train function for classification tasks."

    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion  = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef       = lm_coef
        self.opt           = opt



        
    def __call__(self, X, Y, M, clf_logits, lm_logits=None, only_return_losses=False):
        # Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:, 1:, 0].contiguous().view(-1)
            M         = M.view(-1, M.size(-1))
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            lm_losses = lm_losses.view(X.size(0), X.size(-2) - 1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses_return = lm_losses.sum()
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        if clf_logits is not None:
            clf_losses = self.clf_criterion(clf_logits, Y)
        if only_return_losses and clf_logits is None:
            return lm_losses_return            
        if only_return_losses and lm_logits is None:
            return clf_losses.sum()
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None else clf_losses

        if clf_logits is None:
            train_loss = self.lm_coef * lm_losses.sum()
        elif self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() + self.lm_coef * lm_losses.sum()
        else:
            train_loss = clf_losses.sum()
        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()


class ClassificationLossComputeStructured:
    "A Loss compute and train function for classification tasks."

    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion  = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef       = lm_coef
        self.opt           = opt




    def _score_para(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).cuda()
        tags = torch.cat([torch.tensor([0], dtype=torch.long).cuda(), tags])
        for i, feat in enumerate(feats):
            score = score  + feat[tags[i + 1]]
        return score

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, 6), 0.).cuda()
        # START_TAG has all of the score.
        #init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(6):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, 6)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                #trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var +  emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var
        alpha = log_sum_exp(terminal_var)
        return alpha


    def __call__(self, X, Y, M, clf_logits, lm_logits=None, only_return_losses=False):
        # Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:, 1:, 0].contiguous().view(-1)
            M         = M.view(-1, M.size(-1))
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            lm_losses = lm_losses.view(X.size(0), X.size(-2) - 1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses_return = lm_losses.sum()
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        if clf_logits is not None:
            forward = self._forward_alg(clf_logits)
            gold = self._score_para(clf_logits, Y)
            clf_losses = forward-gold
        if only_return_losses and clf_logits is None:
            return lm_losses_return            
        if only_return_losses and lm_logits is None:
            return clf_losses.sum()
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None else clf_losses

        if clf_logits is None:
            train_loss = self.lm_coef * lm_losses.sum()
        elif self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() +lm_losses.sum()
        else:
            train_loss = clf_losses.sum()
        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()


class ClassificationLossComputeLM:
    "A Loss compute and train function for classification tasks."

    def __init__(self, lm_criterion, lm_coef, opt=None):
        self.lm_criterion  = lm_criterion
        self.lm_coef       = lm_coef
        self.opt           = opt

    def __call__(self, X, M, lm_logits, only_return_losses=False):
        # Language modeling loss

        x_shifted = X[:, 1:, 0].contiguous().view(-1)
        M         = M.view(-1, M.size(-1))
        lm_losses = self.lm_criterion(lm_logits, x_shifted)
        lm_losses = lm_losses.view(X.size(0), X.size(-2) - 1)
        lm_losses = lm_losses * M[:, 1:]
        lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        if only_return_losses:
            return lm_losses

       
        train_loss = self.lm_coef * lm_losses.sum()

        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()
# TODO Implement a LossCompute class for similiraty tasks.
