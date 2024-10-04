import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
# Sample dialogue data (replace with your actual dataset)
data = [
("Hello", "Hi there!"),
("How are you doing?", "I'm doing well, thanks for asking!"),
("What's the weather like?", "It's sunny today!")
]
# Separate input and output sentences
input_sentences = [data[i][0] for i in range(len(data))]
output_sentences = [data[i][1] for i in range(len(data))]
# Tokenization (converting text to numerical representation)
# (This is a simplified example, more robust tokenization is needed in practice)
word2idx = {"<PAD>": 0} # Padding token
idx2word = {0: "<PAD>"}
cur_idx = 1
for sentence in input_sentences + output_sentences:
    for word in sentence.split():
        if word not in word2idx:
            word2idx[word] = cur_idx
            idx2word[cur_idx] = word
            cur_idx += 1
            
# Convert sentences to sequences of indices
input_seqs = [[word2idx[word] for word in sentence.split()] for sentence in input_sentences]
output_seqs = [[word2idx[word] for word in sentence.split()] for sentence in output_sentences]

# Pad sequences to ensure they have the same length (optional for this example)
input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=word2idx["<PAD>"])
output_seqs_padded = pad_sequence(output_seqs, batch_first=True, padding_value=word2idx["<PAD>"])

