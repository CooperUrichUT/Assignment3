# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
from torch.utils.data import DataLoader, TensorDataset


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
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
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)  # Output dimension matches the vocabulary size

    def forward(self, indices):
        x = self.embedding(indices)
        x = self.positional_encoding(x)

        attentions = []
        for layer in self.transformer_layers:
            x = layer(x)

        # Predict log probabilities with log-softmax activation function
        log_probs = nn.functional.log_softmax(self.fc(x), dim=-1)

        return log_probs, attentions



class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, num_heads=4):
        super(TransformerLayer, self).__init__()

        self.num_heads = num_heads
        self.head_dim = d_internal // num_heads

        self.q_linear = nn.Linear(d_model, d_internal)
        self.k_linear = nn.Linear(d_model, d_internal)
        self.v_linear = nn.Linear(d_model, d_internal)

        self.out_linear = nn.Linear(d_internal, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Linear(d_internal, d_model)
        )

    def forward(self, input_vecs):
        if len(input_vecs.shape) == 2:
            # Handle 2D input: add batch and sequence length dimensions
            input_vecs = input_vecs.unsqueeze(1)  # Add sequence length dimension

        batch_size, seq_len, d_model = input_vecs.size()
        q = self.q_linear(input_vecs)
        k = self.k_linear(input_vecs)
        v = self.v_linear(input_vecs)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_linear(attn_output)

        attention_output = self.norm1(input_vecs + attn_output)

        ff_output = self.feed_forward(attention_output)

        output = self.norm2(attention_output + ff_output)

        return output

# Implementation of positional encoding that you can use in your network
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
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0).type(x.dtype)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed).type(x.dtype)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    vocab_size = 27
    model = Transformer(
        vocab_size=vocab_size,
        num_positions=20,
        d_model=128,
        d_internal=512,
        num_classes=3,  # Three classes: 0, 1, 2
        num_layers=4
    )

    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Convert data into DataLoader for batching
    train_data = TensorDataset(torch.stack([example.input_tensor for example in train]),
                               torch.stack([example.output_tensor for example in train]))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_input, batch_target in train_loader:
            optimizer.zero_grad()

            # Forward pass
            log_probs, _ = model(batch_input)  # Model returns log probabilities

            # Reshape log_probs to (batch_size * seq_len, 3)
            log_probs_flat = log_probs.view(-1, 3)

            # Expand batch_target to match the number of classes (3)
            batch_target_flat = batch_target.view(-1, 1).expand(-1, 3).contiguous().view(-1)

            # Compute the loss
            loss = loss_fn(log_probs_flat, batch_target_flat)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    # Evaluate the model on the dev set (similar to your decode function)
    model.eval()
    dev_input_tensors = torch.stack([example.input_tensor for example in dev])
    dev_output_tensors = torch.stack([example.output_tensor for example in dev])
    with torch.no_grad():
        dev_outputs, _ = model(dev_input_tensors)  # Model returns log probabilities

    # You can compute metrics or do further analysis on dev_outputs here

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
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
