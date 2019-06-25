import json
import sys
sys.path.insert(0, '../bert')
from run_squad import read_squad_examples
import tokenization
from konlpy.tag import Mecab

if __name__=="__main__":
    vocab = set()
    mecab = Mecab('../mecab-ko-dic-2.1.1-20180720')
    train_examples = read_squad_examples('./KorQuAD_v1.0_train.json',
                                         is_training=True)
    dev_examples = read_squad_examples('./KorQuAD_v1.0_dev.json',
                                       is_training=True)
    tokenizer = tokenization.FullTokenizer(vocab_file='./vocab.txt', do_lower_case=False) 
    def add_to_vocab(vocab, tokenizer, examples):
        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer.tokenize(example.question_text)
            vocab |= set(query_tokens)
            for (i, token) in enumerate(example.doc_tokens):
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    vocab |= set(sub_tokens)
    print("starting build vocab")
    add_to_vocab(vocab, tokenizer, train_examples)
    add_to_vocab(vocab, tokenizer, dev_examples)
    print("finished adding vocabs")
    with open('./vocab.txt', 'w') as file:
        for word in vocab:
            file.write(word + "\n")
        
