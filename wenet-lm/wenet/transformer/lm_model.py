# Copyright 2021 NWPU(kun wei)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sequential implementation of Recurrent Neural Network Language Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, reverse_pad_list,
                                th_accuracy)


class SequentialRNNLM(torch.nn.Module):
    """Sequential RNNLM.

    See also:
        https://github.com/pytorch/examples/blob/4581968193699de14b56527296262dd76ab43557/word_language_model/model.py

    """
    def __init__(self, configs, type='gru', n_vocab=3233, unit=256, layer=2, dropout_rate=0.2):
        """Initialize class.

        Args:
            n_vocab (int): The size of the vocabulary
            args (argparse.Namespace): configurations. see py:method:`add_arguments`

        """
        torch.nn.Module.__init__(self)
        unit = configs['unit']
        layer = configs['layer']
        self._setup(
            rnn_type=type.upper(),
            ntoken=n_vocab,
            ninp=unit,
            nhid=unit,
            nlayers=layer,
            dropout=dropout_rate,
        )
        self.vocab = n_vocab

    def _setup(
        self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False
    ):
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    "An invalid option for `--model` was supplied, "
                    "options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"
                )
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers:
        #  A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self._init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def _init_weights(self):
        # NOTE: original init in pytorch/examples
        # initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # NOTE: our default.py:RNNLM init
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        """Compute LM loss value from buffer sequences.

        Args:
            x (torch.Tensor): Input ids. (batch, len)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of
                loss to backward (scalar),
                negative log-likelihood of t: -log p(t) (scalar) and
                the number of elements in x (scalar)

        Notes:
            The last two return values are used
            in perplexity: p(t)^{-n} = exp(-log p(t) / n)

        """
        sos = self.vocab - 1
        eos = self.vocab - 1
        ignore_id = 0
        x, y = add_sos_eos(x, sos, eos, ignore_id)
        t = y
        y = self._before_loss(x, None)[0]
        mask = (x != 0).to(y.dtype)
        loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        logp = loss * mask.view(-1)
        logp = logp.sum()
        count = mask.sum()
        return logp / count, logp, count

    def _before_loss(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_state(self, x):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        bsz = 1
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid),
            )
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def score(self, y, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y, new_state = self._before_loss(y[-1].view(1, 1), state)
        logp = y.log_softmax(dim=-1).view(-1)
        return logp, new_state
