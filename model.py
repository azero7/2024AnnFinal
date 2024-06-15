import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 2
EOS_token = 3
MAX_LENGTH = 10

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method

        # 初始化一个线性层 self.attn 
        if method == "general":
            self.attn = nn.Linear(hidden_size, hidden_size)

        # 多初始化一个参数向量 self.v
        elif method == "concat":
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.rand(hidden_size))

    def dot_score(self, hidden, encoder_output):
        # 确保 hidden 的维度是 [batch_size, 1, hidden_size]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        
        # 扩展 hidden 到 encoder_output 的序列长度
        hidden = hidden.expand(-1, encoder_output.size(1), -1)
        
        return torch.sum(hidden * encoder_output, dim=2)


    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
            
            batch_size = encoder_output.size(0)
            seq_len = encoder_output.size(1)
            
            # 确保 hidden 维度为 [batch_size, hidden_size]
            if hidden.dim() == 3:  # 当 hidden 是 [num_layers, batch_size, hidden_size]
                hidden = hidden[-1]  # 只取最后一层的隐藏状态
            
            hidden_size = hidden.size(1)
            
            # 将 hidden 从 [batch_size, hidden_size] 扩展到 [batch_size, seq_len, hidden_size]
            hidden = hidden.unsqueeze(1).expand(batch_size, seq_len, hidden_size)

            # 拼接 hidden 和 encoder_output
            concatenated = torch.cat((hidden, encoder_output), 2)

            # 通过全连接层计算注意力能量
            energy = self.attn(concatenated).tanh()

            # 计算最终的注意力分数
            attn_energies = torch.sum(self.v * energy, dim=2)

            return attn_energies


    def forward(self, hidden, encoder_outputs):

        hidden = hidden.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # dot product
        if self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)
        # multiplicative
        elif self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        # additive
        elif self.method == "concat":
            hidden = hidden[-1]  # 只取最后一层的隐藏状态
            attn_energies = self.concat_score(hidden, encoder_outputs)
        
        attn_weights = F.softmax(attn_energies, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context, attn_weights

# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, dropout_p=0.1, embeddings=None):
        '''
            input_size: 输入词汇表的大小，输入序列中可能出现的不同词汇的数量。
            hidden_size: GRU的隐藏层大小,每个GRU单元的输出向量的维度。
            n_layers: GRU的层数。
            dropout_p: Dropout的概率,防止过拟合。
            embeddings: 预训练的嵌入层，如果提供则使用，否则创建一个新的嵌入层
        '''
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings)
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        '''
            输入：形状为 [batch_size, seq_len] 的张量(batch_size 是一个批次中的序列数量, seq_len 是每个序列的长度)
            self.embedding(input)：将输入序列中的词汇索引转换为对应的嵌入向量。输出形状为 [batch_size, seq_len, hidden_size]。
            self.dropout(embedded):嵌入向量上应用dropout。
            self.gru(embedded):将嵌入向量传递给GRU层。
            output:形状为 [batch_size, seq_len, hidden_size]，表示每个时间步的输出。
            hidden:形状为 [n_layers, batch_size, hidden_size]，表示每一层最后一个时间步的隐藏状态。
        '''
        if input.max().item() >= self.embedding.num_embeddings:
            raise ValueError(f"Input index {input.max().item()} is out of range for embedding size {self.embedding.num_embeddings}")
        # Embedding & dropout layer
        embedded = self.dropout(self.embedding(input))
        # GRU layer
        output, hidden = self.gru(embedded)
        return output, hidden

# Decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, attn_model='dot', n_layers=2, dropout_p=0.1, embeddings=None):
        # 参数含义类似encoder
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attn_model = attn_model
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # unsupportive for embedding yet
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings)
        else:
            self.embedding = nn.Embedding(output_size, hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.attention = Attention(self.hidden_size, method=attn_model)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, n_layers, dropout=self.dropout_p, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        '''
        输入：
            input: 当前时间步的输入，形状为 [batch_size]。
            last_hidden: 上一个时间步的隐藏状态，形状为 [n_layers, batch_size, hidden_size]。
            encoder_outputs: 编码器的所有输出，形状为 [batch_size, seq_len, hidden_size]。
        输出：
            output:形状为 [batch_size, output_size]，表示当前时间步的预测输出。
            hidden: 形状为 [n_layers, batch_size, hidden_size], 表示GRU的隐藏状态。
            attn_weights: 形状为 [batch_size, seq_len]，表示注意力权重。
        '''

        # 嵌入层和dropout层：输入通过嵌入层转换为嵌入向量，通过dropout层得到 embedded，形状为 [batch_size, 1, hidden_size]
        batch_size = input.size(0)
        embedded = self.embedding(input).view(batch_size, 1, -1)
        embedded = self.dropout(embedded)

        # 对于 concat 注意力机制，使用 last_hidden 和 encoder_outputs 计算上下文向量 context 和注意力权重 attn_weights，
        # 并将 context 和 embedded 拼接，作为GRU的输入。
        if self.attn_model == 'concat':
            context, attn_weights = self.attention(last_hidden, encoder_outputs)
            context = context.squeeze(1)
            # CAT
            rnn_input = torch.cat((embedded, context.unsqueeze(1)), 2)
            # GRU
            output, hidden = self.gru(rnn_input, last_hidden)
            # SOFTMAX
            output = F.log_softmax(self.out(output.squeeze(1)), dim=1)

            return output, hidden, attn_weights

        # dot & general attention
        context, attn_weights = self.attention(last_hidden[-1], encoder_outputs)

        context = context.unsqueeze(1)
        context = context.squeeze(2)
        rnn_input = torch.cat((embedded, context), 2)

        output, hidden = self.gru(rnn_input, last_hidden)
        output = F.log_softmax(self.out(output.squeeze(1)), dim=1)
        return output, hidden, attn_weights

# Seq2Seq Model
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5, return_attention=False):
        '''
            src: 输入序列，形状为 [batch_size, src_len]。
            trg: 目标序列，形状为 [batch_size, trg_len]。
            teacher_forcing_ratio: 使用教师强制的比例。1即是完全使用teacher-forcing
            return_attention: 是否返回注意力权重, 主要是用于在visualize中展现attention热图
        '''
        batch_size = src.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        input = trg[:, 0]

        attentions = torch.zeros(batch_size, max_len - 1, src.shape[1]).to(self.device) if return_attention else None

        for t in range(1, max_len):
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            if return_attention:
                attentions[:, t - 1] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)

        if return_attention:
            return outputs, attentions
        else:
            return outputs
        
    def greedy_decode_bleu(self, src, max_len, sos_token=SOS_token, eos_token=EOS_token):
        '''
            src: 输入序列。
            max_len: 目标序列的最大长度。
            sos_token: 开始符号的索引。
            eos_token: 结束符号的索引。
        '''
        batch_size = src.shape[0]

        # 存储每一步的输出索引
        outputs = torch.zeros(batch_size, max_len).to(self.device).long()
        encoder_outputs, hidden = self.encoder(src)

        input = torch.tensor([sos_token] * batch_size).to(self.device).long()

        for t in range(max_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            top1 = output.argmax(1)  # 获取最大概率的索引
            outputs[:, t] = top1
            input = top1
            if (top1 == eos_token).all():
                break

        return outputs

    def greedy_decode(self, src, max_len, sos_token=SOS_token, eos_token=EOS_token):
        # 相比于bleu版本的，这里返回的是每个时间步的完整输出概率分布，而不是最大概率的索引。
        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        input = torch.tensor([sos_token] * batch_size).to(self.device)

        for t in range(1, max_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = top1
            if (top1 == eos_token).all():
                break

        return outputs
    
    def beam_search_decode(self, src, max_len, sos_token=SOS_token, eos_token=EOS_token, beam_width=3):
        batch_size = src.size(0)
        trg_vocab_size = self.decoder.output_size
        encoder_outputs, hidden = self.encoder(src)

        # 初始化概率张量
        log_probs_all = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)

        # 初始化beam
        beams = [[(torch.LongTensor([sos_token]).to(self.device), hidden[:, i, :].unsqueeze(1).contiguous(), 0, False)] for i in range(batch_size)]
        completed_beams = [[] for _ in range(batch_size)]

        for step in range(max_len):
            # 初始化候选序列：每个时间步保留 beam_width 数量的候选序列
            all_candidates = []
            for i in range(batch_size):
                candidates = []
                for seq, hidden, score, finished in beams[i]:
                    if finished:
                        completed_beams[i].append((seq, hidden, score, finished))
                        continue
                    
                    # 调用解码器：获取当前时间步的输出和隐藏状态。
                    output, hidden, _ = self.decoder(seq[-1].view(1, -1), hidden.contiguous(), encoder_outputs[i].unsqueeze(0).contiguous())
                    log_probs = F.log_softmax(output, dim=1).squeeze(0)
                    top_log_probs, top_indices = log_probs.topk(beam_width)

                    # 收集对数概率
                    log_probs_all[i, step, :] = log_probs

                    for k in range(beam_width):
                        new_seq = torch.cat([seq, top_indices[k].view(1).to(self.device)])
                        new_score = score + top_log_probs[k].item()
                        new_finished = top_indices[k].item() == eos_token
                        candidates.append((new_seq, hidden, new_score, new_finished))

                # 选择最佳候选：选择得分最高的 beam_width 数量的候选序列。
                candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]
                beams[i] = candidates
                all_candidates.extend(candidates)

            if all([len(b) == 0 for b in beams]):
                break

        # 返回所有时间步的对数概率
        return log_probs_all




def main():
    input_size = 5000  # Example input vocabulary size
    output_size = 5000  # Example output vocabulary size
    hidden_size = 256
    n_layers = 2

    encoder = EncoderRNN(input_size, hidden_size, n_layers).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_size, attn_model='dot', n_layers=n_layers).to(device)
    model = Seq2SeqModel(encoder, decoder, device).to(device)

    # Example forward pass (replace with actual training loop and data)
    src = torch.randint(0, input_size, (32, MAX_LENGTH)).to(device)  # Example source batch
    trg = torch.randint(0, output_size, (32, MAX_LENGTH)).to(device)  # Example target batch

    outputs = model(src, trg)
    print(outputs.shape)

    # Perform Beam Search decoding
    decoded_outputs = model.beam_search_decode(src, MAX_LENGTH, beam_width=3)
    print(decoded_outputs)

if __name__ == '__main__':
    main()
