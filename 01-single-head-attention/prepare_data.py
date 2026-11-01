import torch

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#print(f"Length of dataset in characters: {len(text)}")
#print(text[:500])  # peek at the first 500 characters

# Get all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(f"Vocabulary: {''.join(chars)}")
#print(f"Vocab size: {vocab_size}")

# Build the Mappings
stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i : ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Quick sanity check
test = encode("First Citizen")
#print(test)
#print(decode(test))


# Making data into a long array of numbers
data = torch.tensor(encode(text),dtype = torch.long)
#print(data.shape, data.dtype)
#print(data[:100])  # peek at first 100 tokens as numbers

#time to chop off the data; we keep most of the data for testing and then rest for training
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# defining a block size of 8 characters. Sort of an active memory of the engine. Also called max context length.
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
#    print(f"when input is {context.tolist()}, target is {target}")

batch_size = 64  # how many independent sequences per batch

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
