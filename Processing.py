
import os
import torch
#from torch.utils import data
#from Model import Corpus
import spacy
from torchtext.data.metrics import bleu_score


def create_data(name_in, name_out, startl, endl, target, target2):
    max_len = 0
    with open('training//'+'temp_'+name_in, 'w', encoding="utf-8") as outf:
        with open('training//'+'temp_'+name_out, 'w', encoding="utf-8") as train_out:
            with open(target, 'r', encoding="utf8") as f:
                data = f.readlines()


            with open(target2, 'r', encoding="utf8") as f:
                data2 = f.readlines()

            print("padding data")
            for lineno, line in enumerate(data):
                    #print(lineno)
                    if lineno < startl:
                        continue
                    if lineno > endl:
                        break
                    if startl <= lineno <= endl:
                        input_sentence = line.strip() + ' <eos> ' + data2[lineno]
                        #print(res)
                        if len(input_sentence.split()) > max_len:
                            max_len = len(input_sentence.split())               
                        outf.write(input_sentence)
                        
                        pad = '<pad> '
                        for i in range(len(line.split())-1):
                            pad += ' <pad> '
                        output_sentence = pad + data2[lineno].strip() + ' <eos> \n'
                        train_out.write(output_sentence)
                        
                        if (lineno % round((endl-startl)/100) == 0):
                            print('Percent: '+str(lineno/(endl-startl)))
                  
    with open('training//'+name_in, 'w', encoding="utf-8") as outf:
        with open('training//'+'temp_'+name_in, 'r', encoding="utf8") as f:
            for line in f:
                line = line.strip()
                while len(line.split()) < max_len:
                    line += ' <pad> '
                outf.write(line+'\n')
                
    with open('training//'+name_out, 'w', encoding="utf-8") as outf:  
        with open('training//'+'temp_'+name_out, 'r', encoding="utf8") as f:
            for line in f:
                line = line.strip()
                while len(line.split()) < max_len:
                    line += ' <pad> '
                outf.write(line+'\n')
                
    os.remove('training//'+'temp_'+name_in)
    os.remove('training//'+'temp_'+name_out)

     
    return max_len
    


def split_validation_data(filename_in, filename_out, percent, target_in, target_out):
    with open('training//'+target_in, 'r', encoding="utf8") as f:
                input_data = f.readlines()
            
    with open('training//'+target_out, 'r', encoding="utf8") as f:
                output_data = f.readlines()
    nlines = len(input_data)
    split = round(nlines*percent/100)
    test_split = (nlines - split) // 2


    with open('training//'+filename_in, 'w', encoding="utf-8") as outf:
        for line in input_data[split:]:
            outf.write(line)
            
            
    with open('training//'+filename_out, 'w', encoding="utf-8") as outf:
        for line in output_data[split:]:
            outf.write(line)
            
    with open('training//'+target_in, 'w', encoding="utf-8") as outf:
        for line in input_data[:split]:
            outf.write(line)
            
            
    with open('training//'+target_out, 'w', encoding="utf-8") as outf:
        for line in output_data[:split]:
            outf.write(line)

def split_data(src_file, trg_file, percent):
    with open('training//'+src_file, 'r', encoding="utf8") as f:
                src_data = f.readlines()
            
    with open('training//'+trg_file, 'r', encoding="utf8") as f:
                trg_data = f.readlines()

    nlines = len(src_data)
    split = round(nlines*percent/100)
    test_split = ((nlines - split) // 2)+split


    with open('training//train_src', 'w', encoding="utf8") as outf:
        for line in src_data[:split]:
            outf.write(line)
            
            
    with open('training//train_trg.', 'w', encoding="utf8") as outf:
        for line in trg_data[:split]:
            outf.write(line)
            
    with open('training//val_src', 'w', encoding="utf8") as outf:
        for line in src_data[split:test_split]:
            outf.write(line)
            
            
    with open('training//val_trg', 'w', encoding="utf8") as outf:
        for line in trg_data[split:test_split]:
            outf.write(line)

    with open('training//test_src', 'w', encoding="utf8") as outf:
        for line in src_data[test_split:]:
            outf.write(line)
            
            
    with open('training//test_trg', 'w', encoding="utf8") as outf:
        for line in trg_data[test_split:]:
            outf.write(line)

    return ['training//train_src.txt', 'training//train_trg.txt'], ['training//val_src.txt', 'training//val_trg.txt'], ['training//test_src.txt', 'training//test_trg.txt']



def add_sos_token(file, output_name):
    with open('training/'+output_name, 'w', encoding="utf-8") as outf:
        with open('training//'+file, 'r',  encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()
                    line = '<sos> '+line
                    outf.write(line+'\n')
    #os.remove(file)


def get_max_line_length(file):
    max_len = 0
    with open('training//'+file, 'r', encoding="utf-8") as f:
        for line in f:
            if len(line.split()) > max_len:
                 max_len = len(line.split()) 
    f.close()
    return max_len

def add_padding(file, max_len):
    
    with open('training//'+'temp_'+file, 'w', encoding="utf-8") as outf:
        with open('training//'+file, 'r', encoding="utf8") as f:
            data = f.readlines()

        for lineno, line in enumerate(data):
            line = line.strip()+' '
            while len(line.split()) < max_len:
                line = line + '<pad> '
            outf.write(line+'\n')

        #for line in outf:
            
def add_eos_token(file, output_name):
    with open('training/'+output_name, 'w', encoding="utf-8") as outf:
        with open('training//'+file, 'r',  encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()
                    line = line + ' <eos>'
                    outf.write(line+'\n')   

    #os.remove(file)  


def preprocess_data():
   #print(os.getcwd())
    #sentence_len = create_data('train_in.txt','train_out.txt', 0, 30000,'training//PHP.de-en.de','training//PHP.de-en.en')
    split_validation_data('valid_in.txt', 'valid_out.txt', 80, 'train_in.txt', 'train_out.txt')


def translate_sentence(model, sentence, german, english, device, max_length=50, printIt=False):
    # src tokenizer
    spacy_ger = spacy.load("de")

    # tokenize input using spacy if it is a string
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    #addind sos and eos
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    #print the input sentence
    if(printIt):
        print(tokens)
    text_to_indices = [german.vocab.stoi[token] for token in tokens]


    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(0).to(device)

    outputs = [english.vocab.stoi["<sos>"]]

    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[:, -1].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    if printIt:
        print(translated_sentence[1:])
    return translated_sentence[1:]


def bleu(data, model, german, english, device, printIt=False):
    targets = []
    outputs = []

    if model is None:
        with open('model.pt', 'rb') as f:
            model = torch.load(f).to(device)

    print('calculating bleu score...')
    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device, printIt=printIt)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

#if __name__ == "__main__":
    #preprocess_data()
    # max_len = get_max_line_length('PHP.de-en.de')
    # add_padding('PHP.de-en.de', max_len)
    # add_sos_token('temp_PHP.de-en.de', 'test.txt')
    # add_eos_token('test.txt', 'train_in.txt')


    # max_len = get_max_line_length('PHP.de-en.en')
    # add_padding('PHP.de-en.en', max_len)
    # add_sos_token('temp_PHP.de-en.en', 'test.txt')
    # add_eos_token('test.txt', 'train_target.txt')

    #split_validation_data('valid_src.txt', 'valid_trg.txt', 10, 'train_in.txt', 'train_target.txt')

    #translate_sentence()
    #split_data('PHP.de-en.de', 'PHP.de-en.en', 80)