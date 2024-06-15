import torch
import matplotlib.pyplot as plt
from model import EncoderRNN, AttnDecoderRNN, Seq2SeqModel
import os
from pypinyin import pinyin, Style

# 创建保存图像的目录
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 将中文句子转换为拼音, 因为这个系统没有安装中文拼写，这里将中文转换成拼音来做图表的x轴
def convert_to_pinyin(sentence):
    pinyin_sentence = pinyin(sentence, style=Style.NORMAL)
    return [word[0] for word in pinyin_sentence]

# 调用训练好的模型来翻译句子并返回注意力
def translate_and_return_attention(model, src_sentence, src_vocab, trg_vocab, device, max_len=50):
    model.eval()
    with torch.no_grad():
        src_indexes = [src_vocab[word] for word in src_sentence]
        src_tensor = torch.tensor(src_indexes).unsqueeze(0).to(device)

        encoder_outputs, hidden = model.encoder(src_tensor)

        trg_indexes = [trg_vocab['<bos>']]
        attentions = torch.zeros(max_len, len(src_indexes)).to(device)

        for i in range(max_len):
            trg_tensor = torch.tensor([trg_indexes[-1]]).to(device)
            with torch.no_grad():
                output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs)
            attentions[i] = attention.squeeze(0)

            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            if pred_token == trg_vocab['<eos>']:
                break

        trg_tokens = [list(trg_vocab.keys())[list(trg_vocab.values()).index(i)] for i in trg_indexes]
        return trg_tokens[1:], attentions[:len(trg_tokens)-1]

# 可视化注意力机制
def visualize_attention(model, src_sentence, src_vocab, trg_vocab, device, save_path='attention_visualization.png'):
    trg_sentence, attentions = translate_and_return_attention(model, src_sentence, src_vocab, trg_vocab, device)

    src_tokens = convert_to_pinyin(src_sentence)
    trg_tokens = trg_sentence

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    cax = ax.matshow(attentions.cpu().numpy(), cmap='viridis')
    
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(trg_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90)
    ax.set_yticklabels(trg_tokens)
    
    plt.xlabel('Source Sequence')
    plt.ylabel('Target Sequence')
    plt.title('Attention Mechanism Visualization')
    
    plt.savefig(save_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预处理后的数据
    data_path = 'preprocessed_data.pth'
    train_data, valid_data, test_data, vocab_en, vocab_zh = torch.load(data_path)

    # 确保词汇表大小与训练时的一致
    input_size = len(vocab_zh)
    output_size = len(vocab_en)

    # 初始化模型
    hidden_size = 256
    n_layers = 2
    dropout_p = 0.3

    encoder = EncoderRNN(input_size, hidden_size, n_layers, dropout_p)
    decoder = AttnDecoderRNN(hidden_size, output_size, attn_model='dot', n_layers=n_layers, dropout_p=dropout_p)
    model = Seq2SeqModel(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load('best_model.pth'))

    # 输入的中文句子
    src_sentence = ["我", "是", "中国", "人"]
    
    # 可视化注意力机制
    visualize_attention(model, src_sentence, vocab_zh, vocab_en, device, 'attention_visualization.png')

if __name__ == '__main__':
    main()
