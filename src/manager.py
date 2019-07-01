import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

from src.modules import make_baseline_model


class ModelManager:
    """
    Manages PyTorch nn.Module instance

        - Train Loop
        - Evaluate Loop
        - TODO: early stopping, save/load checkpoint, better logging
    """

    # noinspection PyUnusedLocal
    def __init__(self, src_vocab, tgt_vocab, pad_idx=0):
        # Seq2Seq model
        self.model = None
        # optimizer
        self.optimizer = None
        # loss function criterion
        self.criterion = None
        # gradient clip value
        self.clip_value = None

    def train(self, iterator):
        """
        Training loop for the model

        :param iterator: PyTorch DataIterator instance

        :return: average epoch loss
        """
        # enable training of model layers
        self.model.train()

        epoch_loss = 0

        for i, batch in tqdm(enumerate(iterator), total=len(iterator),
                             desc='training loop'):
            # get source and target data
            src, src_lengths = batch.src
            trg, trg_lengths = batch.trg

            output, _ = self.model(src, trg, src_lengths, trg_lengths)

            # compute logits
            output = F.log_softmax(output, dim=-1)

            # flatten output, trg tensors and ignore <sos> token
            # output -> [(seq_len - 1) * batch x output_size] (2D logits)
            # trg -> [(seq_len - 1) * batch] (1D targets)
            y_pred = output[:, 1:].contiguous().view(-1, output.size(-1))
            y = trg[:, 1:].contiguous().view(-1)

            # compute loss
            loss = self.criterion(y_pred, y)

            # backward pass
            loss.backward()

            # clip the gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)

            # update the parameters
            self.optimizer.step()

            # zero gradients for next batch
            self.optimizer.zero_grad()

            epoch_loss += loss.item()

        # return the average loss
        return epoch_loss / len(iterator)

    def evaluate(self, iterator):
        """
        Evaluation loop for the model

        :param iterator: PyTorch DataIterator instance

        :return: average epoch loss
        """
        # disable training of model layers
        self.model.eval()

        epoch_loss = 0

        # don't update model parameters
        with torch.no_grad():
            for i, batch in tqdm(enumerate(iterator), total=len(iterator),
                                 desc='evaluation loop'):
                # get source and target data
                src, src_lengths = batch.src
                trg, trg_lengths = batch.trg

                output, _ = self.model(src, trg, src_lengths, trg_lengths)

                # compute logits
                output = F.log_softmax(output, dim=-1)

                # reshape same as train loop
                y_pred = output[:, 1:].contiguous().view(-1, output.size(-1))
                y = trg[:, 1:].contiguous().view(-1)

                # compute loss
                loss = self.criterion(y_pred, y)

                epoch_loss += loss.item()

        # return the average loss
        return epoch_loss / len(iterator)


class BaselineModelManager(ModelManager):
    def __init__(self, src_vocab, tgt_vocab, pad_idx=0):
        super(BaselineModelManager, self).__init__(src_vocab, tgt_vocab)
        # Seq2Seq model
        self.model = make_baseline_model(src_vocab, tgt_vocab)
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters())
        # loss function criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        # gradient clip value
        self.clip_value = 10
