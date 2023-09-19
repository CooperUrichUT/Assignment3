
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
        self.model = Transformer(self.vocab_size, 32, 32)
        self.softmax = nn.LogSoftmax()

    def get_next_char_log_probs(self, context):
        context_index = [self.indexer.index_of(char) for char in context]
        index = self.model(torch.tensor([context_index]))[1].squeeze()
        return self.softmax(index).detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        context_index = [self.indexer.index_of(char) for char in context]
        next_chars_index = [self.indexer.index_of(char) for char in next_chars]
        output = self.model(torch.tensor([(context_index + next_chars_index)]))[0].squeeze()
        # print('multiple', output)
        log_prob_sum = 0
        for i, next_char_idx in enumerate(next_chars_index):
            hidden_state = output[len(context)-1+i, :]
            log_prob_sum += self.softmax(hidden_state).squeeze()[next_char_idx]

        result = log_prob_sum.detach().numpy().item()

        return result

class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, d_internal):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.W_Matrix = nn.Linear(d_internal, vocab_size)
        nn.init.xavier_uniform_(self.W_Matrix.weight)
        self.rnn = nn.GRU(d_model, d_internal, batch_first=True)
        # This loss funtion is better for this question
        self.loss_fcn = nn.CrossEntropyLoss()

    def forward(self, x, indices=None):
        embeddings = self.embeddings(x)

        if indices:
            output, index = self.rnn(embeddings, indices)
        else:
            output, index = self.rnn(embeddings)

        index = index.squeeze()
        return self.W_Matrix(output), self.W_Matrix(index)

    
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # d_k matrix
        self.internal = d_model
        # all matrices and their weights
        self.W_Q_Matrix = nn.Linear(d_internal, d_model)
        self.W_Q_Weight = self.W_Q_Matrix.weight
        self.W_K_Matrix = nn.Linear(d_internal, d_model)
        self.W_K_Weight = self.W_K_Matrix.weight
        self.W_V_Matrix = nn.Linear(d_internal, d_model)
        self.W_V_Weight = self.W_V_Matrix.weight

        # single linear layer
        self.linear_layer = nn.Linear(d_internal, d_internal)
        # self.Lin2 = nn.Linear(d_internal, d_internal)




# https://stackoverflow.com/questions/57793574/self-attention-explainability-of-the-output-score-matrix
    def forward(self, input_vecs):
        # X * W^Q
        queries = torch.matmul(input_vecs, self.W_Q_Weight)
        # X * W^K
        keys = torch.matmul(input_vecs, self.W_K_Weight)
        # X * W^V
        values = torch.matmul(input_vecs, self.W_V_Weight)
        # Get the attention value
        attention = self.attention_algorithm(queries,keys,values, self.internal)
        # apply ReLU
        relu = torch.nn.functional.relu(attention) 
        # Step forward
        step = self.linear_layer(relu) 
        return step, attention

    
    def attention_algorithm(self, queries, keys, values, d_k, mask=None, dropout=None):
        # multiply Q x kT, divide by the square root of d_k
        inner = torch.matmul(queries, keys.transpose(-2, -1)) /  math.sqrt(d_k)
        # take softmax
        softmax = torch.nn.functional.softmax(inner, dim = -1)
        # multiply by values matrix
        attention = torch.matmul(softmax, values)
        return attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)

def chunk_text(text, chunk_size):
    # Chunk the input text into chunks of the specified size
    chunks = [list(text[i:i + chunk_size]) for i in range(0, len(text), chunk_size)]
    for chunk in chunks:
        while len(chunk) < chunk_size:
            chunk.append(' ')
    return chunks

def create_batch(batch_x, batch_y, train_exs):
    batch_x.append(train_exs[-1][0])
    batch_y.append(train_exs[-1][1])

    return batch_x, batch_y

def process_batch(batch_x, batch_y, nlm_model, optimizer, vocab_index, total_loss):
    chunk_size = 32
    batch_size = 250
    target = torch.tensor(batch_y)
    nlm_model.model.zero_grad()
    prob, _ = nlm_model.model(torch.tensor(batch_x))
    prob = prob.view(batch_size * chunk_size, len(vocab_index))
    target = target.view(batch_size * chunk_size)
    loss = nlm_model.model.loss_fcn(prob, target)
    total_loss += loss
    loss.backward()
    optimizer.step()

    return total_loss, optimizer, nlm_model



def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # Combine train_text and dev_text for training on both datasets
    train_and_dev_text = train_text + dev_text

    # Create the NeuralLanguageModel instance
    nlm_model = NeuralLanguageModel(vocab_index)
    # Define training parameters
    optimizer = optim.Adam(nlm_model.model.parameters(), lr=1e-2)
    batch_x = []
    batch_y = []
    train_exs = []
    chunk_size = 32
    batch_size = 250

    # Preprocess the combined text into chunks
    chunked_text = chunk_text(train_and_dev_text, chunk_size)
    
    # Training loop
    epochs = 15
    for epoch in range(epochs):
        random.shuffle(chunked_text)
        total_loss = 0.0
        for chunk in chunked_text:
            chunk_idx = [vocab_index.index_of(char) for char in chunk]
            train_exs.append([[vocab_index.index_of(' ')] + chunk_idx[:-1], chunk_idx])

            if len(batch_x) < batch_size:
                batch_x, batch_y = create_batch(batch_x, batch_y, train_exs)
            else:
                total_loss, optimizer, nlm_model= process_batch(batch_x, batch_y, nlm_model, optimizer, vocab_index, total_loss)
                batch_x = []
                batch_y = []

        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    return nlm_model
