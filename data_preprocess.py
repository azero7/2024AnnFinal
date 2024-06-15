import json
import jieba
import nltk
from collections import Counter
import torch
import numpy as np

# nltk.download('punkt')

# 读取数据
def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 数据清洗和截断
def clean_and_truncate_data(data, max_len_en=41, max_len_zh=48):
    cleaned_data = []
    for item in data:
        en_tokens = nltk.word_tokenize(item['en'])
        zh_tokens = list(jieba.cut(item['zh']))
        
        if len(en_tokens) > max_len_en - 2:  # 减去特殊标记长度
            en_tokens = en_tokens[:max_len_en - 2]
        
        if len(zh_tokens) > max_len_zh - 2:  # 减去特殊标记长度
            zh_tokens = zh_tokens[:max_len_zh - 2]
        
        cleaned_data.append({'en': en_tokens, 'zh': zh_tokens, 'index': item['index']})
    return cleaned_data

def load_glove_embeddings(glove_path, vocab, embedding_dim=50):
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    word_index = {word: idx for idx, word in enumerate(vocab.keys())}

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            if word in word_index:
                embeddings[word_index[word]] = vector
    
    return torch.tensor(embeddings, dtype=torch.float32)


# 分词（已在clean_and_truncate_data中完成）
def tokenize(data):
    return data

# 构建词典
def build_vocab_and_embeddings(tokenized_data, min_freq=1, max_size=1000000):
    counter_en = Counter()
    counter_zh = Counter()
    for item in tokenized_data:
        counter_en.update(item['en'])
        counter_zh.update(item['zh'])

    specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    vocab_en = {special: idx for idx, special in enumerate(specials)}
    vocab_zh = {special: idx for idx, special in enumerate(specials)}

    max_size_en = max_size - len(vocab_en)
    max_size_zh = max_size - len(vocab_zh)

    for word, freq in counter_en.items():
        if freq >= min_freq and word not in vocab_en:
            vocab_en[word] = len(vocab_en)
        if len(vocab_en) >= max_size_en:
            break

    for word, freq in counter_zh.items():
        if freq >= min_freq and word not in vocab_zh:
            vocab_zh[word] = len(vocab_zh)
        if len(vocab_zh) >= max_size_zh:
            break

    # embeddings_en = np.zeros((len(vocab_en), embedding_dim))
    # with open(glove_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         values = line.split()
    #         word = values[0]
    #         vector = np.array(values[1:], dtype='float32')
    #         if word in vocab_en:
    #             embeddings_en[vocab_en[word]] = vector

    print(f"Vocabulary size (English): {len(vocab_en)}")
    print(f"Vocabulary size (Chinese): {len(vocab_zh)}")

    return vocab_en, vocab_zh



# 转换为索引
def convert_to_indices(data, vocab_en, vocab_zh):
    UNK_IDX_EN = vocab_en['<unk>']
    UNK_IDX_ZH = vocab_zh['<unk>']
    
    indexed_data = []
    for item in data:
        if not isinstance(item['en'], list) or not isinstance(item['zh'], list):
            raise ValueError(f"Expected list type for 'en' and 'zh' fields, but got {type(item['en'])} and {type(item['zh'])}")

        indexed_item = {
            'en': [vocab_en.get(token, UNK_IDX_EN) for token in ['<bos>'] + item['en'] + ['<eos>']],
            'zh': [vocab_zh.get(token, UNK_IDX_ZH) for token in ['<bos>'] + item['zh'] + ['<eos>']],
            'index': item['index']
        }

        # Debug
        for idx in indexed_item['zh']:
            if idx >= len(vocab_zh) or idx < 0:
                print(f"Error in Chinese sentence: {item['zh']}")
                raise ValueError(f"Index {idx} out of range for vocab size {len(vocab_zh)} in Chinese sentence {item['zh']}")
        for idx in indexed_item['en']:
            if idx >= len(vocab_en) or idx < 0:
                print(f"Error in English sentence: {item['en']}")
                raise ValueError(f"Index {idx} out of range for vocab size {len(vocab_en)} in English sentence {item['en']}")

        indexed_data.append(indexed_item)
    return indexed_data


# 检查索引是否在词汇表范围内
def check_indices_within_vocab(indexed_data, vocab_en, vocab_zh):
    for item in indexed_data:
        for idx in item['zh']:
            if idx >= len(vocab_zh) or idx < 0:
                raise ValueError(f"Index {idx} out of range for zh vocab size {len(vocab_zh)} in sentence {item['zh']}")
        for idx in item['en']:
            if idx >= len(vocab_en) or idx < 0:
                raise ValueError(f"Index {idx} out of range for en vocab size {len(vocab_en)} in sentence {item['en']}")

if __name__ == '__main__':
    # glove_path = 'glove.6B.50d.txt'
    # train_data = read_data('dataset/augmented_train.jsonl')
    train_data = read_data('dataset/train_100k.jsonl')
    valid_data = read_data('dataset/valid.jsonl')
    test_data = read_data('dataset/test.jsonl')

    train_data = clean_and_truncate_data(train_data)
    valid_data = clean_and_truncate_data(valid_data)
    test_data = clean_and_truncate_data(test_data)

    vocab_en, vocab_zh = build_vocab_and_embeddings(train_data)

    train_data = convert_to_indices(train_data, vocab_en, vocab_zh)
    valid_data = convert_to_indices(valid_data, vocab_en, vocab_zh)
    test_data = convert_to_indices(test_data, vocab_en, vocab_zh)

    check_indices_within_vocab(train_data, vocab_en, vocab_zh)
    check_indices_within_vocab(valid_data, vocab_en, vocab_zh)
    check_indices_within_vocab(test_data, vocab_en, vocab_zh)

    # 保存处理后的数据
    torch.save((train_data, valid_data, test_data, vocab_en, vocab_zh), 'preprocessed_data.pth')
