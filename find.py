import json
import matplotlib.pyplot as plt
import jieba

def analyze_lengths(file_path):
    en_lengths = []
    zh_lengths = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            en_lengths.append(len(data['en'].split()))  # 英文使用空格分词
            zh_tokens = list(jieba.cut(data['zh']))  # 使用jieba进行中文分词
            zh_lengths.append(len(zh_tokens))  # 计算分词后的中文长度

    return en_lengths, zh_lengths

en_lengths, zh_lengths = analyze_lengths('dataset/train_100k.jsonl')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(en_lengths, bins=50, alpha=0.7)
plt.title('English Sentence Lengths')
plt.xlabel('Length')
plt.ylabel('Number of Sentences')

plt.subplot(1, 2, 2)
plt.hist(zh_lengths, bins=50, alpha=0.7)
plt.title('Chinese Sentence Lengths')
plt.xlabel('Length')
plt.ylabel('Number of Sentances')

plt.tight_layout()
plt.savefig('sentence_lengths.png')

import numpy as np
print("英文句子平均长度：", np.mean(en_lengths))
print("中文句子平均长度：", np.mean(zh_lengths))
print("英文句子95%长度：", np.percentile(en_lengths, 95))
print("中文句子95%长度：", np.percentile(zh_lengths, 95))
