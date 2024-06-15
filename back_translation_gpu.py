import time
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import concurrent.futures
import json
import random
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_mbart_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt').to(device)
    return tokenizer, model, device

def translate(texts, tokenizer, model, src_lang, tgt_lang, device, max_length=512):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens]
    return translated_texts

def back_translate(sentences, tokenizer, model, device, max_length=512):
    start_time = time.time()
    # Translate to English and then back to Chinese
    en_trans = translate(sentences, tokenizer, model, 'zh_CN', 'en_XX', device, max_length)
    zh_back = translate(en_trans, tokenizer, model, 'en_XX', 'zh_CN', device, max_length)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to back-translate batch: {elapsed_time:.2f} seconds")
    return zh_back

def back_translation(data, tokenizer, model, device, batch_size=6, max_workers=6):
    sentences = [item['zh'] for item in data]
    augmented_data = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            futures.append(executor.submit(back_translate, batch, tokenizer, model, device))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                zh_back_batch = future.result()
                for j, zh_back in enumerate(zh_back_batch):
                    idx = (futures.index(future) * batch_size) + j
                    augmented_data.append({'en': data[idx]['en'], 'zh': zh_back, 'index': data[idx]['index']})
                    # print(f"Original: {data[idx]['zh']}")
                    # print(f"Back Translated: {zh_back}")
                    # print(f"English: {data[idx]['en']}\n")
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
            finally:
                # 清理 GPU 内存缓存和变量
                torch.cuda.empty_cache()
                gc.collect()

    return augmented_data

def main():
    data_path = 'dataset/train_10k.jsonl'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    tokenizer, model, device = load_mbart_models()
    augmented_data = back_translation(data, tokenizer, model, device, batch_size=2, max_workers=1)

    full_data = data + augmented_data
    random.shuffle(full_data)

    with open('dataset/augmented_train.jsonl', 'w', encoding='utf-8') as f:
        for item in full_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    main()
