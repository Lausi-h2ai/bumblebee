
import argparse
import os
from torchtext.legacy.data import Field
import spacy
import Processing as pr
import torch
from torchtext.legacy.datasets import Multi30k


parser = argparse.ArgumentParser(description='translate some sentence using an existing model')
parser.add_argument('--model', type=str, default='best_model.pt',
help='path to model file')
parser.add_argument('--sentence', type=str, default='ein junges kind kauft ein eis',
help='sentence you want to translate')
parser.add_argument('--bleu', type=bool, default=False,
help='if you want to calculate bleu score')
args = parser.parse_args()


def tokenize_src(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_trg(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

    
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #spacy de and en vocab building
    spacy_ger = spacy.load('de')
    spacy_eng = spacy.load('en')

    source = Field(tokenize=tokenize_src, lower=True, init_token="<sos>", eos_token="<eos>", batch_first=True)

    target = Field(
        tokenize=tokenize_trg, lower=True, init_token="<sos>", eos_token="<eos>", batch_first=True
    )


    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(source, target),root = 'data'
    )

    source.build_vocab(train_data, max_size=10000, min_freq=2)
    target.build_vocab(train_data, max_size=10000, min_freq=2)
    try:
        with open(args.model, 'rb') as f:
            model = torch.load(f).to(device)
    except:
        print('Could not find model {} in {}'.format(args.model, os.getcwd()))

    if args.bleu:
        pr.bleu(test_data, None, source, target, device, printIt=True)
    else:
        pr.translate_sentence(model, args.sentence, source, target, device, printIt=True)

