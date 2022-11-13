import torch
import torch.nn as nn
import numpy as np
INF=1e12

def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
    """稀疏多标签交叉熵损失
    说明：
        1. 输入：
           y_true[batch_size, num_classes, max_entity_number, head_and_tail_of_entity(因此长度为2)]
           y_pred[batch_size, num_classes, sequence_len, sequence_len] （各类型的抽取的实体的评分矩阵）
           y_true对应正例下标，而非one-hot向量。
        2. mask_zero = True时，y_true采用零填充使得y_true.shape相同，
           此时，保证y_pred展平后的shape=[...,num_classes+1],
           y_true中正例下标的范围为1~num_classes而非0~num_classes，
           否则填充的0将对损失函数的计算产生影响。
        3. 请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，
           尤其是不能加sigmoid或者softmax（原因见4）。
        4. 预测阶段则输出y_pred大于0的类。
    参考：
        损失函数的计算公式：https://kexue.fm/archives/7359，https://kexue.fm/archives/8888
        Tensorflow实现：https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272
    """
    shape = y_pred.shape
    # y_true变为[batch_size, num_classes, max_entity_number, index_of_pred]（实体在展平的评分矩阵中对应的下标）
    y_true = y_true[..., 0] * shape[2] + \
        y_true[..., 1]  # 实体头*矩阵的横坐标+实体尾，即在矩阵中对应的下标
    # y_pred变为y_pred[batch_size, num_classes, sequence_len*sequence_len]（展平的评分矩阵）
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))

    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)  # 加e^{0},dim=-1（最后一维）

    # 将y_pred[...,0]替换为infs，使得利用y_true的0填充索引查找y_pred时对应为infs，
    # e^{-infs}约等于0，不影响损失函数的最终计算
    if mask_zero:
        infs = torch.ones_like(zeros) * float('inf')
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

    # 正例的损失函数，1+\sum e^{-s}
    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)  # 所有正例的分数
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)

    # 负例的损失函数，a + \log(1-e^{b-a})
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)  # a
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss  # b-a
    aux_loss = torch.clip(1-torch.exp(aux_loss), 1e-16, 1)  # 1-e^{b-a}
    neg_loss = all_loss + torch.log(aux_loss)

    loss = torch.mean(torch.sum(pos_loss + neg_loss))
    return loss


class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, ent_type_size, head_size, RoPE=True, tril_mask=True):
        '''GlobalPointer的初始化
        参数：
            1. ent_type_size：类型数目
            2. head_size：muti-head的维度
            3. RoPE：旋转式位置编码
            4. tri_mask：下三角遮罩，抽取subject、object可以设置为True（head一定在tail之前）
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.ent_type_size = ent_type_size
        self.head_size = head_size
        self.RoPE = RoPE
        self.tri_mask = tril_mask
        # 对于每个类型，会分别给出每个位置作为start向量、end向量的分数
        self.dense = nn.Linear(
            hidden_size, self.ent_type_size * self.head_size * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        '''Sinusoidal位置编码（为了辅助RoPE编码的生成）
        目的：
            生成sin(m\theta_{i})、cos(m\theta_{i})
            其中，\theta_{i} = 1000^{-2*i/d}
        参数：
            d对应output_dim：位置向量的维度
            i对应indices: 范围为0到output_dim/2（2i：sin，2i+1：cos）
        相关知识参考：
            https://kexue.fm/archives/8265/comment-page-1#comments
        '''
        # 生成位置信息m
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1) 
        # 计算\theta_{i} 
        indices = torch.arange(0, output_dim // 2, dtype=torch.float) 
        indices = torch.pow(10000, -2 * indices / output_dim)  
        # 生成m\theta_{i}，即包含\theta的embedding
        embeddings = position_ids * indices 
        # 生成\sin(m\theta),\cos(m\theta)，在最后一维进行堆叠
        # embeddings.shape=[seq_len, output_dim//2, 2]，其中，2对应sin,cos
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1) 
        # 将生成的位置编码，扩展到batch_size个
        # embeddings.shape=[batch_size, seq_len, output_dim//2,2]
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape)))) 
        # embeddings.shape=[batch_size, seq_len, output_dim]
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device) 
        return embeddings

    def forward(self,context_outputs, attention_mask):
        self.device = attention_mask.device
        # last_hidden_state.shape=[batch_size, seq_len, hidden_size]
        last_hidden_state = context_outputs[0]
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # output.shape=[batch_size, seq_len, ent_type_size * head_size *2]
        output = self.dense(last_hidden_state)
        output = torch.split(output, self.head_size*2, dim=-1)
        # output.shape=[batch_size, seq_len, ent_type_size, head_size*2]
        output = torch.stack(output,dim=-2)
        # qw,kw.shape=[batch_size, seq_len, ent_type_size, head_size]
        qw, kw = output[..., :self.head_size], output[..., self.head_size:]
        
        #RoRE编码
        if self.RoPE:
            # pos_emb.shape = [batch_size, seq_len, head_size]
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.head_size) 
            # cos_pos, sin_pos = [batch_size, seq_len, 1, head_size]
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1) # 抽取奇数列信息(cosm\theta)，并复制两遍
            sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1) # 抽取偶数列信息(sinm\theta)，并复制两遍
            
            # 根据公式，得到start向量(qw)的相对位置编码
            qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1) # 将q的奇数行加上负号，得到与(sin)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos 

            # 类似地，得到end向量(kw)的相对位置编码
            kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积，得到评分结果
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw) # 先求qw的转置，再求qw与kw的矩阵乘法结果
        
        # 排除padding
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size,self.ent_type_size,seq_len,seq_len)
        logits = logits * pad_mask - (1-pad_mask)*INF

        # 排除下三角
        if self.tri_mask:
            mask = torch.tril(torch.ones_like(logits),diagonal=-1) # torch.tri:返回一个矩阵主对角线以下的下三角矩阵
            logits = logits - mask * INF

        return logits/self.head_size ** 0.5


class GPLinker(nn.Module):
    '''关系抽取模型GPLinker
    由3个GlobalPointer构成，分别用于：
        1. mention_detect：subject和object对应的实体抽取
        2. s_o_head：对于每种关系，subject和object的head
        3. s_o_tail：对于每种关系，subject和object的tail
    相关知识参考：
        https://kexue.fm/archives/8888
    '''
    def __init__(self, encoder, len_schema):
        super().__init__()
        self.device = encoder.device
        self.encoder = encoder
        self.mention_detect = GlobalPointer(hidden_size=1024,ent_type_size=2,head_size=64).to(self.device)
        self.s_o_head = GlobalPointer(hidden_size=1024,ent_type_size=len_schema,head_size=64,RoPE=False,tril_mask=False).to(self.device)
        self.s_o_tail = GlobalPointer(hidden_size=1024,ent_type_size=len_schema,head_size=64,RoPE=False,tril_mask=False).to(self.device)

    def forward(self,batch_token_ids,batch_mask_ids):
        encoder_output= self.encoder(batch_token_ids,batch_mask_ids)
        mention_detect_output = self.mention_detect(encoder_output, batch_mask_ids)
        so_head_output = self.s_o_head(encoder_output,batch_mask_ids)
        so_tail_output = self.s_o_tail(encoder_output,batch_mask_ids)
        return mention_detect_output, so_head_output, so_tail_output
