import torch


# Step 1: The first step is to take in the input file and let the python program read it

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Step 2: Collect all the unique characters to get vocabulory
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Step 3: Build a mapping between characters and an array of integers
stoi = {ch : i for i,ch in enumerate(chars)}
itos = {i : ch for i,ch in enumerate(chars)}

# Step 4: Functions for moving back and forth between array of integers and characters
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])


# Quick sanity check
#test = encode("Multi Headed Attention")
#print(test)
#print(decode(test))


# Step 5:  We now take the dataset and turn it into a long array of integers
data = torch.tensor(encode(text),dtype = torch.long)

# Step 6:  We split the data set into two parts out of which the first is used for trainig and the latter is used for testing
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Step 7: We try several block sizes

#block_size = 8
#block_size = 16
block_size = 32


# Step 8: The batch size is mainly for running things on GPU efficiently
#batch_size = 4
#batch_size = 8 
#batch_size = 16
#batch_size = 32
batch_size = 64

# Step 9: We write the function for splitting the input into batches starting with random initial positions
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


if __name__ == "__main__":

    xb, yb = get_batch('train')
    print('inputs:')
    print(xb.shape)
    print(xb)
    print('targets:')
    print(yb.shape)
    print(yb)
