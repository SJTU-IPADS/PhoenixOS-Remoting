import torch
from transformers import BertTokenizer, BertForMaskedLM
import time
import sys
import ctypes
import os

# load remoting bottom library
remoting_bottom_lib_path = os.getenv('REMOTING_BOTTOM_LIBRARY')
if remoting_bottom_lib_path is not None:
    cpp_lib = ctypes.CDLL(remoting_bottom_lib_path)
    start_trace = cpp_lib.startTrace
    end_trace = cpp_lib.endTrace

log_breakpoint_lib_path = os.getenv('LOG_BREAKPOINT_LIBRARY')
if log_breakpoint_lib_path is not None:
    cpp_lib = ctypes.CDLL(log_breakpoint_lib_path)
    breakpoint = cpp_lib.log_breakpoint

if(len(sys.argv) < 3):
    print('Usage: python3 inference.py num_iter batch_size [model_path]')
    sys.exit()
    
num_iter = int(sys.argv[1])
batch_size = int(sys.argv[2])
model_path = 'bert-base-uncased'
if len(sys.argv) > 3:
    model_path = sys.argv[3]
    print('model_path:', model_path)
else:
    print('Using remote model:', model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path).to(device)

masked_sentences = ["The primary [MASK] of the United States is English." for _ in range(batch_size)]
pos_masks = [3 for _ in range(batch_size)]

# remove initial overhead
torch.cuda.empty_cache()
for i in range(2):
    encoded_inputs = tokenizer(masked_sentences, return_tensors='pt', padding='max_length', max_length=20).to(device)
    outputs = model(**encoded_inputs)
    most_likely_token_ids = [torch.argmax(outputs[0][i, pos, :]) for i, pos in enumerate(pos_masks)]
    unmasked_tokens = tokenizer.decode(most_likely_token_ids).split(' ')
    unmasked_sentences = [masked_sentences[i].replace('[MASK]', token) for i, token in enumerate(unmasked_tokens)]

if log_breakpoint_lib_path is not None:
    breakpoint()

if remoting_bottom_lib_path is not None:
    start_trace()

print("begin trace")
sys.stdout.flush()

T1 = time.time()

for i in range(num_iter):
    encoded_inputs = tokenizer(masked_sentences, return_tensors='pt', padding='max_length', max_length=20).to(device)
    outputs = model(**encoded_inputs)
    most_likely_token_ids = [torch.argmax(outputs[0][i, pos, :]) for i, pos in enumerate(pos_masks)]
    unmasked_tokens = tokenizer.decode(most_likely_token_ids).split(' ')
    unmasked_sentences = [masked_sentences[i].replace('[MASK]', token) for i, token in enumerate(unmasked_tokens)]
    
T2 = time.time()
print('time used: ', T2-T1)

if remoting_bottom_lib_path is not None:
    end_trace()

# print(unmasked_sentences)
