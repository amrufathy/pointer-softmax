import torch
from torch import nn
from torch.nn import functional as F

from src.utils import sequence_mask


class EncoderRNN(nn.Module):
    """
    Applies a bi-directional GRU to encode a sequence.
    """

    def __init__(self, input_size, emb_size, hidden_size, pad_idx=0, embedding_matrix=None):
        """
        :param input_size: vocab size of source (input) sentences
        :param emb_size: desired embedding size for embedding layer
        :param hidden_size: number of hidden units of GRU
        :param embedding_matrix: optional, matrix of pre-trained embedding vectors
        """
        super(EncoderRNN, self).__init__()

        # embedding layer (with ability to add pre-trained vectors)
        self.embed = nn.Embedding(input_size, emb_size, padding_idx=pad_idx)

        if embedding_matrix is not None:
            self.embed.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embed.weight.requires_grad = True  # train the embeddings layer

        # 1-layer bi-GRU
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, src):
        """
        :param src: batch of input sentences, each sentence is a
            vector of word indices in some dictionary.
                shape: [batch x seq_len]

        :return:
            - output: output features from the last layer of the GRU
                shape: [batch x seq_len x directions * hidden_size]
            - hidden_concat: concatenated hidden state of last layer
                shape: [batch x directions * hidden_size]
        """
        embedded = self.embed(src)  # [batch x seq_len x emb]

        # output [batch x seq_len x directions*hidden_size], hidden [batch x directions x hidden_size]
        output, hidden = self.rnn(embedded)

        # concat the final hidden state vector
        hidden_fwd, hidden_bwd = torch.chunk(hidden, 2, 0)
        # hidden_concat [batch x directions*hidden_size]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=2).squeeze(0)

        return output, hidden_concat


class Attention(nn.Module):
    """
    Bahdanau MLP attention
    """

    def __init__(self, hidden_size):
        """
        :param hidden_size: number of hidden units for all layers
        """
        super(Attention, self).__init__()

        # bi-directional encoder -> 2*hidden_size
        self.enc_state_layer = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.dec_state_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, dec_state, enc_state, src_length):
        """
        :param dec_state: previous decoder hidden state
        :param enc_state: encoder hidden features for attention
        :param src_length: lengths of source sequences to create masks for attention

        :return:
            - context: context vector calculated for a word
                shape: [batch x 1 x 2 * hidden_size]
            - alphas: attention probabilities of each word
                shape: [batch x 1 x src_len]
        """
        # compute energies using non-linear mapping
        enc = self.enc_state_layer(enc_state)  # [batch x src_len x hidden_size]
        dec = self.dec_state_layer(dec_state)  # [batch x 1 x hidden_size]

        energies = self.energy_layer(torch.tanh(dec + enc))  # [batch x src_len x 1]
        energies = energies.squeeze(2).unsqueeze(1)  # [batch x 1 x src_len]
        mask = sequence_mask(lengths=src_length)
        energies.data.masked_fill_(~mask.unsqueeze(1), -float('inf'))

        # normalize using softmax
        alphas = torch.softmax(energies, dim=-1)  # [batch x 1 x src_len]
        context = torch.bmm(alphas, enc_state)  # [batch x 1 x 2*hidden_size]

        return context, alphas


class Decoder(nn.Module):
    """
    Base class for a generic RNN decoder that unrolls one
        step at a time.
    """

    def __init__(self, input_size, emb_size, hidden_size, pad_idx=0):
        """
        :param input_size: vocab size of target (output) sentences
        :param emb_size: desired embedding size for embedding layer
        :param hidden_size: number of hidden units of GRU
            NOTE: emb_size, hidden_size are currently the same as the encoder's
        """
        super(Decoder, self).__init__()

        self.attention = Attention(hidden_size)
        self.embed = nn.Embedding(input_size, emb_size, padding_idx=pad_idx)
        # input to GRU is the embedded target sequences + encoder's bi-directional hidden states
        self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, batch_first=True)

        # to initialize from the final encoder state of last layer
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True)

        self.teacher_forcing_ratio = 0.5

    def _forward_step(self, *_input):
        """
        Performs a single decoder step (1 word)

        :param prev_embed: embedded previous word (in target sequence)
        :param encoder_hidden: encoder hidden features for attention
        :param src: source sequences (used only in pointer softmax)
        :param src_length: lengths of source sequences to create masks for attention
        :param hidden: previous decoder hidden state

        :return:
            - output: decoder output for one word
                shape: [batch x 1 x output_size]
            - hidden: new decoder hidden state
                shape: [1 x batch x hidden_size]
        """
        raise NotImplementedError

    def forward(self, src, trg, encoder_hidden, encoder_final, src_length, trg_length):
        """
        Unroll the decoder one step at a time.

        :param src: source target sequences (used only for pointer softmax decoder)
        :param trg: target input sequences
        :param encoder_hidden: hidden states from the encoder
        :param encoder_final: last state from the encoder
        :param src_length: lengths of source sequences to create masks for attention
        :param trg_length: lengths of target sequences to get RNN unroll steps

        :return:
            - outputs: concatenated decoder outputs for whole sequence
                shape: [batch x max_steps x output_size]
            - hidden: last decoder hidden state
                shape: [1 x batch x hidden_size]
        """
        # maximum number of steps to unroll the RNN
        max_steps = trg_length.max()

        # initialize decoder hidden state
        hidden = self.init_hidden(encoder_final)  # [1 x batch x hidden_size]

        # store output vectors
        outputs = []

        trg_embedded = self.embed(trg)  # [batch x max_steps x emb]

        # unroll the decoder RNN for max_steps
        # TODO: use teacher forcing
        for i in range(max_steps):
            prev_embed = trg_embedded[:, i].unsqueeze(1)  # [batch x 1 x emb]
            output, hidden = self._forward_step(
                prev_embed, encoder_hidden, src, src_length, hidden)
            outputs.append(output)

        # [batch x max_steps x output_size]
        outputs = torch.cat(outputs, dim=1)

        return outputs, hidden

    def init_hidden(self, encoder_final):
        """
        Returns the initial decoder state, conditioned on the final encoder state.

        :param encoder_final: final state from the last layer of the encoder
            shape: [batch x directions * hidden_size]

        :return: initial hidden state for GRU
            shape: [1 x batch x hidden_size]
        """
        return torch.tanh(self.bridge(encoder_final)).unsqueeze(0)

    @staticmethod
    def decode_mechanism(logits):
        """
        Helper function to decode the logits to a sequence of words from the vocab

        :param logits: probability distribution of all words in vocab

        :return: index of most probable word in the vocab
        """
        _, indices = torch.max(logits, dim=2)

        return indices


class DecoderRNN(Decoder):
    """
    Decodes a sequence of words given an initial context vector
        representing the input sequence and using a Bahdanau (MLP) attention.
    """

    def __init__(self, input_size, emb_size, hidden_size, dropout=.3, pad_idx=0):
        """
        :param dropout: dropout applied to output of RNN
        """
        super(DecoderRNN, self).__init__(input_size, emb_size, hidden_size, pad_idx)

        # output size is the same as input size
        output_size = input_size

        self.dropout_layer = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size,
                                      output_size, bias=False)

    def _forward_step(self, prev_embed, encoder_hidden, src, src_length, hidden):
        # compute context vector using attention mechanism
        prev_dec_state = hidden.squeeze().unsqueeze(1)  # [batch x 1 x hidden_size]
        context, _ = self.attention(
            dec_state=prev_dec_state, enc_state=encoder_hidden,
            src_length=src_length)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        # here i override the output variable with actual output value
        #   instead of RNN hidden values
        output = torch.cat([prev_embed, output, context], dim=2)
        # [batch x 1 x output_size]
        output = torch.log_softmax(self.output_layer(self.dropout_layer(output)), dim=-1)

        return output, hidden


class DecoderPS(Decoder):
    """
    Attention decoder with pointer softmax layer
    """

    def __init__(self, input_size, emb_size, hidden_size, dropout=.3, pad_idx=0):
        super(DecoderPS, self).__init__(input_size, emb_size, hidden_size, pad_idx)

        # output size is the same as input size
        output_size = input_size

        # pointer softmax layers
        self.switching_layer = nn.Linear(2 * hidden_size + hidden_size, 1, bias=False)
        # normal output layer with output dim = vocab size
        self.dropout_layer = nn.Dropout(p=dropout)
        self.shortlist_softmax = nn.Linear(hidden_size + 2 * hidden_size + emb_size, output_size, bias=False)

    def _forward_step(self, prev_embed, encoder_hidden, src, src_length, hidden):
        # compute context vector using attention mechanism
        prev_dec_state = hidden.squeeze().unsqueeze(1)  # [batch x 1 x hidden_size]
        context, attn_probs = self.attention(
            dec_state=prev_dec_state, enc_state=encoder_hidden,
            src_length=src_length)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        # switching layer is conditioned on context vector and hidden state of the decoder
        switch_input = torch.cat([prev_dec_state, context], dim=2)
        switch_prob = torch.ge(torch.sigmoid(self.switching_layer(switch_input)), 0.5).to(torch.float32)  # [batch x 1 x 1]

        # vocab output
        v_in = torch.cat([prev_embed, output, context], dim=2)
        # [batch x 1 x output_size]
        vocab_output = torch.log_softmax(self.shortlist_softmax(self.dropout_layer(v_in)), dim=-1)

        # pointer output
        # use attn_probs tensor to calculate the location of word to copy
        log_attn = torch.log(attn_probs).squeeze()  # [batch x src_len]

        _, a_indices = torch.max(log_attn, dim=1)
        a_indices = a_indices.unsqueeze(1)  # [batch x 1]

        # get word indices from a_indices
        src_indices = src.gather(dim=1, index=a_indices)  # [batch x 1]

        # create one-hot-encoded vectors from word indices
        pointer_output = F.one_hot(src_indices, num_classes=vocab_output.size(-1)).to(torch.float32)  # [batch x 1 x output_size]

        # select based on switching variable
        output = torch.where(switch_prob == 1, vocab_output, pointer_output)  # [batch x 1 x output_size]

        return output, hidden


class Seq2Seq(nn.Module):
    """
    Standard Encoder-Decoder architecture.
    """

    def __init__(self, src_vocab, trg_vocab, emb_size, hidden_size, dropout=.3, pad_idx=0, embedding_matrix=None):
        """
        :param src_vocab: vocab size of source (input) sentences
        :param trg_vocab: vocab size of target (output) sentences
        :param emb_size: desired embedding size for embedding layer (used in both encoder and decoder)
        :param hidden_size: number of hidden units of any layer
        :param dropout: dropout applied to output of decoder RNN
        :param embedding_matrix: optional, matrix of pre-trained embedding vectors
        """
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(src_vocab, emb_size, hidden_size, pad_idx, embedding_matrix)
        self.decoder = DecoderRNN(trg_vocab, emb_size, hidden_size, dropout, pad_idx)

    def forward(self, src, trg, src_length, trg_length):
        """
        :param src: source sequences
            shape: [batch x src_len]
        :param trg: target sequences
            shape: [batch x trg_len]
        :param src_length: lengths of source sequences
        :param trg_length: lengths of target sequences

        :return: decoder's output
        """
        encoder_hidden, encoder_final = self.encoder(src)
        return self.decoder(src, trg, encoder_hidden, encoder_final, src_length, trg_length)


class PSSeq2Seq(Seq2Seq):
    """
    Encoder-Decoder architecture with a pointer softmax decoder
    """

    def __init__(self, src_vocab, trg_vocab, emb_size, hidden_size, dropout=.3, pad_idx=0, embedding_matrix=None):
        super(PSSeq2Seq, self).__init__(src_vocab, trg_vocab, emb_size, hidden_size, dropout, pad_idx, embedding_matrix)
        # overwrite decoder
        self.decoder = DecoderPS(trg_vocab, emb_size, hidden_size, dropout, pad_idx)


def make_baseline_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, dropout=.3, pad_idx=0, embedding_matrix=None):
    """
    Create baseline seq2seq model from given parameters and pass it to GPU
    """
    if embedding_matrix is not None:
        err_msg = 'Embedding matrix has inconsistent dimensions'
        assert embedding_matrix.size(0) == src_vocab and embedding_matrix.size(1) == emb_size, err_msg

    return Seq2Seq(src_vocab, tgt_vocab, emb_size, hidden_size, dropout, pad_idx, embedding_matrix).cuda()


def make_ps_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, dropout=.3, pad_idx=0, embedding_matrix=None):
    """
    Create pointer softmax seq2seq model from given parameters and pass it to GPU
    """
    if embedding_matrix is not None:
        err_msg = 'Embedding matrix has inconsistent dimensions'
        assert embedding_matrix.size(0) == src_vocab and embedding_matrix.size(1) == emb_size, err_msg

    return PSSeq2Seq(src_vocab, tgt_vocab, emb_size, hidden_size, dropout, pad_idx, embedding_matrix).cuda()
