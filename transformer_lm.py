
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn, optim
import random


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_indexer):
        self.indexer = vocab_indexer
        self.vocab_size = len(self.indexer)
        self.model = RNNLM(self.vocab_size, 32, 32)
        self.softmax = nn.LogSoftmax()

    def get_next_char_log_probs(self, context):
        context_idx = [self.indexer.index_of(char) for char in context]
        h_n = self.model(torch.tensor([context_idx]))[1].squeeze()
        hidden_state = h_n
        log_probability = self.softmax(hidden_state)
        result = log_probability.detach().numpy()
        return result

    def get_log_prob_sequence(self, next_chars, context):
        context_idx = [self.indexer.index_of(char) for char in context]
        next_chars_idx = [self.indexer.index_of(char) for char in next_chars]
        all_idx = context_idx + next_chars_idx
        output = self.model(torch.tensor([all_idx]))[0].squeeze()
        # print('multiple', output)
        log_prob_sum = 0
        for i, next_char_idx in enumerate(next_chars_idx):
            hidden_state = output[len(context)-1+i, :]
            log_probability = self.softmax(hidden_state)
            log_probability = log_probability.squeeze()[next_char_idx]

            log_prob_sum += log_probability

        result = log_prob_sum.detach().numpy().item()

        return result

class RNNLM(nn.Module):

    def __init__(self, vocab_size, inp, hid):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, inp)
        self.W = nn.Linear(hid, vocab_size)
        nn.init.xavier_uniform_(self.W.weight)
        self.rnn = nn.GRU(inp, hid, batch_first=True)
        # This loss funtion is better for this question
        self.loss_fcn = nn.CrossEntropyLoss()

    def forward(self, x, h_0=None):
        embeddings = self.embeddings(x)
        output, h_n = self.rnn(embeddings, h_0) if h_0 else self.rnn(embeddings)
        h_n = h_n.squeeze()
        return self.W(output), self.W(h_n)


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    clf = NeuralLanguageModel(vocab_index)
    initial_learning_rate = 0.01
    optimizer = optim.Adam(clf.model.parameters(), lr=initial_learning_rate)
    batch_x = []
    batch_y = []
    train_exs = []
    chunk_size = 22
    chunked_text = [list(train_text)[i:i+chunk_size] for i in range(0, len(train_text), chunk_size)]
    for chunk in chunked_text:
        while len(chunk) < chunk_size:
            chunk.append(' ')
        chunk_idx = [vocab_index.index_of(char) for char in chunk]
        train_exs.append([[vocab_index.index_of(' ')]+chunk_idx[:-1], chunk_idx])
    epochs = 20
    batch_size = 250
    for epoch in range(epochs):
        random.shuffle(train_exs)
        total_loss = 0.0
        for ex in train_exs:
            if len(batch_x) < batch_size:
                batch_x.append(ex[0])
                batch_y.append(ex[1])
            else:
                target = torch.tensor(batch_y)
                clf.model.zero_grad()
                probability, h_n = clf.model(torch.tensor(batch_x))
                probability = probability.view(batch_size*chunk_size, len(vocab_index))
                target = target.view(batch_size*chunk_size)
                loss = clf.model.loss_fcn(probability, target)
                total_loss += loss
                loss.backward()
                optimizer.step()
                batch_x = []
                batch_y = []
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return clf