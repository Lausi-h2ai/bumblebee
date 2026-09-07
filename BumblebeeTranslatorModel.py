import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from Processing import translate_sentence, bleu
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
from Bumblebee import Bumblebee
import numpy as np
import random


#spacy de and en vocab building
spacy_src = spacy.load('de')
spacy_trg = spacy.load('en')


def tokenize_src(text):
    return [tok.text for tok in spacy_src.tokenizer(text)]


def tokenize_trg(text):
    return [tok.text for tok in spacy_trg.tokenizer(text)]


src = Field(tokenize=tokenize_src, lower=True, init_token="<sos>", eos_token="<eos>", batch_first=True)

trg = Field(
    tokenize=tokenize_trg, lower=True, init_token="<sos>", eos_token="<eos>", batch_first=True
)
#Data set of Flickr descriptions
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(src, trg),root = 'data'
)


src.build_vocab(train_data, max_size=10000, min_freq=2)
trg.build_vocab(train_data, max_size=10000, min_freq=2)


def evaluate_with_iterator(
        model: nn.Module,
        iterator,
        criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    optimizer.zero_grad()

    with torch.no_grad():

        for batch_idx, batch in enumerate(iterator):
            src, trg = batch.src.to(device), batch.trg.to(device)

            output = model(src, trg[:, :-1])

            #output = model(src, trg[:, :-1]).permute(1,0,2) #(sentence_length, batch_size, embed_size)
            output = output.reshape(-1, output.shape[2]) #(sentence_length*batch_size, embed_size)
            trg = trg[:,1:].reshape(-1) #remove bos token

            loss = criterion(output, trg)

            
            epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(iterator)
    return epoch_loss

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch_idx, batch in enumerate(iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        #input targets without eos token
        output = model(inp_data, target[:, :-1])

        output = output.reshape(-1, output.shape[2]) #(sent_len*batch_size, vocab_size)

        #remove sos token
        target = target[:,1:].reshape(-1) #(sent_len*batch_size)

        optimizer.zero_grad()

        loss = criterion(output, target)

        epoch_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()

    
    average_loss = epoch_loss / len(iterator)
    #print(mean_loss)
    scheduler.step(average_loss)
    return average_loss


import argparse
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--src_data', type=str, default='PHP.de-en.de',
help='location of the data corpus')
parser.add_argument('--trg_data', type=str, default='PHP.de-en.en',
help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=512,
help='size of word embeddings')
parser.add_argument('--n_layers', type=int, default=3,
help='amount of layers in the transformer')
parser.add_argument('--dropout_p', type=float, default=0.10,
help='dropout percentage')
parser.add_argument('--n_heads', type=int, default=8,
help='number of heads in multiheaded attention')
parser.add_argument('--expansion_factor', type=int, default=4,
help='multiplication factor for the forward expansion step')
parser.add_argument('--lr', type=float, default=3e-4,
help='initial learning rate')
parser.add_argument('--clip', type=float, default=1,
help='gradient clipping')
parser.add_argument('--prepare_data', type=bool, default=True,
help='whether to prepare the input and output data (padding, sos, eos...)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
help='batch size')
parser.add_argument('--seed', type=int, default=1122334455,
help='random seed')
parser.add_argument('--save', type=str, default='model.pt',
help='path to save the final model')
args = parser.parse_args("")

#torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load = False
save = True

#Hyperparameters
num_epochs = 5
learning_rate = args.lr
batch_size = args.batch_size

src_vocab_size = len(src.vocab)
trg_vocab_size = len(trg.vocab)
embedding_size = args.emsize
num_heads = args.n_heads
num_encoder_layers = args.n_layers
dropout = args.dropout_p
max_len = 100
forward_expansion = args.expansion_factor
src_pad_idx = trg.vocab.stoi["<pad>"]


#Hyperparameter Grid Search
# emsizes = [256, 512, 1024]
# n_layers = [1, 3, 6, 8]
# n_heads = [8, 16]
# lrs = [3e-4, 1e-4]

emsizes = [1024]
n_layers = [1,2,3,6]
n_heads = [8]
lrs = [0.0001]


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    device=device,
)

#Best found parameters:
####THIS IS BEST: emsize=1024, n_layer=6, n_head=8, lr=0.0001
# New best model on tests data with hyperparametes emsize=1024, n_layer=3, n_head=8, lr=0.0003
# emsize=1024, n_layer=6, n_head=8, lr=0.0001
# New best model found! Bleu Score is 0.3938163856934448, test_loss is 1.6889834739267826 on tests data with hyperparametes emsize=256, n_layer=6, n_head=8, lr=0.0003
best_bleu = 0.39
best_model = None
for emsize in emsizes:
    for n_layer in n_layers:
        for n_head in n_heads:
            for lr in lrs:
                model = Bumblebee(
                    device, trg_vocab_size, src_vocab_size, emsize, max_len, n_layer, dropout, n_head, forward_expansion, src_pad_idx
                ).to(device)

                optimizer = optim.Adam(model.parameters(), lr=lr)

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.1, patience=10, verbose=True
                )

                pad_idx = trg.vocab.stoi["<pad>"]
                criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

                if load:
                    with open('model.pt', 'rb') as f:
                        model = torch.load(f).to(device)

                sentence = "ein kleiner junge kauft sich ein eis"

                best_val_loss = np.inf
                previous_val_loss = np.inf
                train_losses = list()
                val_losses = list()
                print('Starting training with hyperparametes emsize={}, n_layer={}, n_head={}, lr={}'.format(emsize, n_layer, n_head, lr))
                for epoch in range(num_epochs):
                    print(f"[Epoch {epoch+1} / {num_epochs}]")

                    model.eval()
                    

                    epoch_train_loss = train(model, train_iterator, optimizer, criterion, args.clip)
                    epoch_valid_loss = evaluate_with_iterator(model, valid_iterator, criterion)

                    translated_sentence = translate_sentence(
                        model, sentence, src, trg, device, max_length=50, printIt=True
                    )
                    train_losses.append(epoch_train_loss)
                    val_losses.append(epoch_valid_loss)
                    print(f'\tTrain Loss: {epoch_train_loss:.3f}')
                    print(f'\tVal. Loss: {epoch_valid_loss:.3f}')
                    if (not best_val_loss or epoch_valid_loss < best_val_loss) and save:
                            with open(args.save, 'wb') as f:
                                torch.save(model, f)
                            best_val_loss = epoch_valid_loss
                            best_model = model
                    
                with open('manual_attention_train_losses.txt', 'a') as f:
                    f.write('hyperparameters: emsize={}, n_layer={}, n_head={}, lr={} \n'.format(emsize, n_layer, n_head, lr))
                    f.write("train losses: \n")
                    for loss in train_losses:
                        f.write("{:.3f} \n".format(loss))

                    f.write("val losses: \n")
                    for loss in val_losses:
                        f.write("{:.3f} \n".format(loss))


                test_loss = evaluate_with_iterator(best_model, test_iterator, criterion)


                # running on entire test data takes a while, so I sampled some randomly for calculating bleu. Not accurate, but faster
                score = bleu(random.sample(list(test_data),200), None, src, trg, device)
                # score = bleu(test_data, None, src, trg, device, printIt=True)
                print(f"Bleu score {score * 100:.2f} and test loss {test_loss:.3f}")
                if score > best_bleu:
                    print('New best model on tests data with hyperparameters emsize={}, n_layer={}, n_head={}, lr={}'.format(emsize, n_layer, n_head, lr))
                    # with open('best_model.pt', 'wb') as f:
                    #     torch.save(model, f)    
                    best_bleu = score
                    with open('data.txt', 'a') as f:
                        f.write('New best model found! Bleu Score is {}, test_loss is {} on tests data with hyperparametes emsize={}, n_layer={}, n_head={}, lr={}'.format(best_bleu, test_loss, emsize, n_layer, n_head, lr))