# transformer.py
import math
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *

class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.positions = PositionalEncoding(d_model)
        self.transformer = TransformerLayer(d_model, d_internal)
        self.W = nn.Linear(d_internal, num_classes)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        z = self.embeddings(indices)
        a = self.positions.forward(z)
        
        b, att = self.transformer.forward(a)
        c = self.W(b)
        e = torch.nn.functional.log_softmax(c, dim=-1)

        return e, att


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.internal = d_model
        self.WQ = nn.Linear(d_internal, d_model)
        self.WK = nn.Linear(d_internal, d_model)
        self.WV = nn.Linear(d_internal, d_model)
        self.Lin = nn.Linear(d_internal, d_internal)

    def forward(self, input_vecs):
        q = torch.matmul( input_vecs, self.WQ.weight)
        k = torch.matmul( input_vecs, self.WK.weight)
        v = torch.matmul( input_vecs, self.WV.weight)

        att = self.attention(q,k,v, self.internal)
        #lin lin lin relu lin lin softmax

        c = torch.nn.functional.relu(att) 
        d = self.Lin(c) 
        # e = torch.nn.functional.relu(d)
        # f = self.Lin2(e) 
        # print("att:",att)
        return d, att

    
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
    
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        scores = torch.nn.functional.softmax(scores, dim = -1)
            
        output = torch.matmul(scores, v)
        return output


# Implementation of positional encoding that you can use in your network
#Undo mods
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

            


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    vocab_size = 27
    num_positions = 20
    d_model = 100
    d_internal = 50
    model = Transformer(vocab_size, num_positions, d_model, d_internal, 3, 1)

    negative_log_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for t in range(0, num_epochs):
        total_loss = 0.0
        random.shuffle(ex_idxs)
        
        ex_idxs = [i for i in range(0, len(train))]

        for x in ex_idxs:
            ex = train[x]
            model.zero_grad()
            result, _ = model.forward(ex.input_tensor)
            loss = negative_log_loss(result, ex.output_tensor) 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(total_loss)
    model.eval()
    return model
    



####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        print(predictions)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                # print(f"Attention Map Shape: {attn_map.shape}")  # Add this line
                # print(f"Shape of the first attention map: {attn_maps[0].shape}")
                fig, ax = plt.subplots()
                # im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
                print(f"After plotting: Attention Map Shape: {attn_map.shape}")
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
