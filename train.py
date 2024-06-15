import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import EncoderRNN, AttnDecoderRNN, Seq2SeqModel
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import time

SOS_token = 2
EOS_token = 3

# 自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['zh'], self.data[idx]['en']  # 确保中文在前，英文在后

# 将单个样本组合成批次，并进行适当的填充，使得批次中的所有样本长度一致。
def collate_fn(batch):
    zh_batch = [item[0] for item in batch]
    en_batch = [item[1] for item in batch]
    
    zh_lengths = [len(seq) for seq in zh_batch]
    en_lengths = [len(seq) for seq in en_batch]
    
    zh_max_length = max(zh_lengths)
    en_max_length = max(en_lengths)
    
    zh_padded = torch.zeros((len(zh_batch), zh_max_length), dtype=torch.long)
    en_padded = torch.zeros((len(en_batch), en_max_length), dtype=torch.long)
    
    for i, (zh_seq, en_seq) in enumerate(zip(zh_batch, en_batch)):
        zh_padded[i, :len(zh_seq)] = torch.tensor(zh_seq, dtype=torch.long)
        en_padded[i, :len(en_seq)] = torch.tensor(en_seq, dtype=torch.long)
    
    return zh_padded, en_padded


# teacher_forcing_ratio=1表示总是使用 teacher_forcing，=0表示总是使用 Free Running
def train_model(model, train_loader, criterion, optimizer, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    for src, trg in train_loader:
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

        output = output[:, 1:].contiguous()
        trg = trg[:, 1:].contiguous()

        output = output.view(-1, output.shape[-1]).float()  # [batch_size * seq_len, vocab_size]
        trg = trg.view(-1)  # [batch_size * seq_len]

        loss = criterion(output, trg)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)



# torch.backends.cudnn.enabled = False
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def evaluate_model_greedy(model, valid_loader, criterion, device, sos_token=SOS_token, eos_token=EOS_token):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in valid_loader:
            src = src.to(device)
            trg = trg.to(device)
            output = model.greedy_decode(src, max_len=trg.shape[1], sos_token=sos_token, eos_token=eos_token)
            # Output shape: (batch_size, max_len, output_dim)
            # Trg shape: (batch_size, max_len)
            output = output[:, 1:].contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            # Ensure correct data types
            output = output.float()
            trg = trg.long()
            
            loss = criterion(output, trg)
            total_loss += loss.item()

    return total_loss / len(valid_loader)

def evaluate_model_beam_search(model, valid_loader, criterion, device, beam_width=3):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in valid_loader:
            src = src.to(device)
            trg = trg.to(device)

            output = model.beam_search_decode(src, max_len=trg.shape[1], sos_token=SOS_token, eos_token=EOS_token, beam_width=beam_width)

            # print(f"output shape before reshape: {output.shape}")
            # print(f"trg shape after slicing: {trg.shape}")

            # slicing to match the lengths
            output = output[:, 1:].contiguous()
            trg = trg[:, 1:].contiguous()

            # print(f"output shape after slicing: {output.shape}")
            # print(f"trg shape after slicing: {trg.shape}")

            # Ensure output is of type float for log_softmax and loss calculation
            output = output.view(-1, output.size(-1)).float()
            trg = trg.view(-1).long()

            # print(f"output shape after reshape: {output.shape}")
            # print(f"trg shape after reshape: {trg.shape}")

            if output.size(0) != trg.size(0):
                print(f"Shape mismatch after reshape: output {output.shape}, trg {trg.shape}")
                continue

            # # Print some example values
            # print(f"output example values: {output[:5]}")
            # print(f"trg example values: {trg[:5]}")

            loss = criterion(output, trg)
            print(f"loss: {loss.item()}")  # Print the loss for each batch

            total_loss += loss.item()

    return total_loss / len(valid_loader)



def build_itos(vocab):
    itos = {index: token for token, index in vocab.items()}
    return itos

def evaluate_model_bleu(model, valid_loader, device, itos_en):
    model.eval()
    references = []
    hypotheses = []
    with torch.no_grad():
        for src, trg in valid_loader:
            src = src.to(device)
            trg = trg.to(device)

            output = model.greedy_decode_bleu(src, max_len=trg.shape[1], sos_token=SOS_token, eos_token=EOS_token)

            trg_words = [[itos_en[token] for token in sent if token in itos_en and token != 1] for sent in trg.cpu().numpy()]
            output_words = [[itos_en[token] for token in sent if token in itos_en and token != 1] for sent in output.cpu().numpy()]

            references.extend([[ref] for ref in trg_words])  # 双层列表
            hypotheses.extend(output_words)

    cc = SmoothingFunction()
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=cc.method1)
    return bleu_score * 100

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / self.cls
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预处理后的数据
    data_path = 'preprocessed_data.pth'
    train_data, valid_data, test_data, vocab_en, vocab_zh = torch.load(data_path)

    # 超参数
    input_size = len(vocab_zh)  # 注意这里是中文词汇表的大小
    output_size = len(vocab_en)  # 这里是英文词汇表的大小
    hidden_size = 256
    n_layers = 2
    dropout_p = 0.3
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 300
    patience = 5
    decode_strategy = 'beam'
    beam_width = 3

    # 加载数据
    train_dataset = TranslationDataset(train_data)
    valid_dataset = TranslationDataset(valid_data)
    test_dataset = TranslationDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型、损失函数和优化器
    encoder = EncoderRNN(input_size, hidden_size, n_layers, dropout_p).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_size, attn_model='dot', n_layers=n_layers, dropout_p=dropout_p).to(device)
    model = Seq2SeqModel(encoder, decoder, device).to(device)

    # Xavier初始化的目的是让输入和输出的方差相同，以防止信号在前向传播中逐层放大或消失，主要用于激活函数为sigmoid或tanh的神经网络
    # def init_weights(m):
    #     for name, param in m.named_parameters():
    #         if 'weight' in name and 'embedding' not in name:
    #             nn.init.xavier_uniform_(param.data)
    #         elif 'bias' in name:
    #             nn.init.constant_(param.data, 0)

    # Kaiming初始化的目的是使前向传播时各层的输出具有适当的方差，特别适用于ReLU及其变体激活函数
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                nn.init.kaiming_uniform_(param.data)  # 使用Kaiming初始化
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    encoder.apply(init_weights)
    decoder.apply(init_weights)

    # criterion = nn.CrossEntropyLoss(ignore_index=vocab_en['<unk>']) # 0
    criterion = LabelSmoothingLoss(classes=output_size, smoothing=0.1) # 1

    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # 2
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5) # 3

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5) # 4
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # 5是学习率保持的批次大小 # 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) # 10是余弦周期 # 6

    best_valid_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()  # 记录开始时间

        teacher_forcing_ratio = max(0.75 * (0.99 ** epoch), 0.1)  # 初始值设置为 0.75，然后逐步减少到 0.1

        train_loss = train_model(model, train_loader, criterion, optimizer, device, teacher_forcing_ratio)
        
        if (decode_strategy == 'greedy') : 
            valid_loss = evaluate_model_greedy(model, valid_loader, criterion, device)
        elif (decode_strategy == 'beam') :
            valid_loss = evaluate_model_beam_search(model, valid_loader, criterion, device, beam_width)
        else :
            print ("Unsupported decode strategy")

        end_time = time.time()  # 记录结束时间
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)  # 计算耗时

        print(f'Time cost {epoch_mins} mins {epoch_secs:.3f} secs.')
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()}')  # 打印当前学习率

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # scheduler.step(valid_loss) # 和ReduceLROnPlateau.step方法绑定
        scheduler.step()

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1} due to no improvement in validation loss.')
            break

        torch.cuda.empty_cache()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), 'best_model.pth')
        print('Loaded best model state with validation loss:', best_valid_loss)
        print('Model saved as best_model.pth')

    itos_en = build_itos(vocab_en)
    bleu_score = evaluate_model_bleu(model, test_loader, device, itos_en)
    print(f'BLEU Score: {bleu_score:.4f}')

if __name__ == '__main__':
    main()
