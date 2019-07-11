import numpy as np
import sacrebleu
import torch
from torch import nn, optim
from tqdm import tqdm

from src.modules import make_baseline_model, make_ps_model


class ModelManager:
    """
    Manages PyTorch nn.Module instance

        - Train Loop
        - Evaluate Loop
        - TODO: early stopping, better logging
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
        # model save path
        self.path = None
        # vocab to lookup words
        self.vocab = src_vocab.itos

        # internal variables
        self.loss = None

        # seed first
        self.seed()

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

            # flatten output, trg tensors and ignore <sos> token
            # output -> [(seq_len - 1) * batch x output_size] (2D logits)
            # trg -> [(seq_len - 1) * batch] (1D targets)
            y_pred = output[:, 1:].contiguous().view(-1, output.size(-1))
            y = trg[:, 1:].contiguous().view(-1)

            # compute loss
            self.loss = self.criterion(y_pred, y)

            # backward pass
            self.loss.backward()

            # clip the gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)

            # update the parameters
            self.optimizer.step()

            # zero gradients for next batch
            self.optimizer.zero_grad()

            epoch_loss += self.loss.item()

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
        accuracy = 0

        # don't update model parameters
        with torch.no_grad():
            for i, batch in tqdm(enumerate(iterator), total=len(iterator),
                                 desc='evaluation loop'):
                # get source and target data
                src, src_lengths = batch.src
                trg, trg_lengths = batch.trg

                output, _ = self.model(src, trg, src_lengths, trg_lengths)
                decoded_output = self.model.decoder.decode_mechanism(output)

                # reshape same as train loop
                y_pred = output[:, 1:].contiguous().view(-1, output.size(-1))
                y = trg[:, 1:].contiguous().view(-1)

                # compute loss
                loss = self.criterion(y_pred, y)

                epoch_loss += loss.item()

                # using BLEU score for machine translation tasks
                accuracy += sacrebleu.raw_corpus_bleu(sys_stream=self.lookup_words(decoded_output),
                                                      ref_streams=[self.lookup_words(trg)]).score

        # return the average loss
        return epoch_loss / len(iterator), accuracy / len(iterator)

    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }

        torch.save(checkpoint, self.path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']

    @staticmethod
    def seed(_seed=0):
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        torch.backends.cudnn.deterministic = True

    def lookup_words(self, batch):
        batch = [[self.vocab[ind] if ind < len(self.vocab) else self.vocab[0] for ind in ex]
                 for ex in batch]  # denumericalize

        def filter_special(tok):
            return tok not in "<pad>"

        batch = [filter(filter_special, ex) for ex in batch]

        return [' '.join(ex) for ex in batch]


class BaselineModelManager(ModelManager):
    def __init__(self, src_vocab, tgt_vocab, pad_idx=0):
        super(BaselineModelManager, self).__init__(src_vocab, tgt_vocab, pad_idx)
        # Seq2Seq model
        self.model = make_baseline_model(len(src_vocab), len(tgt_vocab), pad_idx=pad_idx)
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters())
        # loss function criterion
        self.criterion = nn.NLLLoss(ignore_index=pad_idx)
        # gradient clip value
        self.clip_value = 1
        # model save path
        self.path = 'models/baseline.pt'


class PointerSoftmaxModelManager(ModelManager):
    def __init__(self, src_vocab, tgt_vocab, pad_idx=0):
        super(PointerSoftmaxModelManager, self).__init__(src_vocab, tgt_vocab, pad_idx)
        # Seq2Seq model
        self.model = make_ps_model(len(src_vocab), len(tgt_vocab), pad_idx=pad_idx)
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters())
        # loss function criterion
        self.criterion = nn.NLLLoss(ignore_index=pad_idx)
        # gradient clip value
        self.clip_value = 1
        # model save path
        self.path = 'models/pointer_softmax.pt'
