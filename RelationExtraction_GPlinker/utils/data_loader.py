import json
import numpy as np
import torch
from torch.utils.data import Dataset


def load_name(filename):
    '''五元组标签的提取
    输入：原始文本
    输出：文本、spo五元组(subject, predicate, object, subject_type, object_type)，用于后续标签的生成与准确率的评估
        例：{'text': '皮肤鳞状细胞癌@环孢A可以促进免疫缺陷小鼠的原发皮肤癌生长，以及体外角质形成细胞生长。', 
            'spo_list': [('皮肤鳞状细胞癌', '病因', '环孢A', '疾病', '社会学')]}
    '''
    D = []
    with open(filename, 'r', encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            D.append({
                "text": line["text"],
                "spo_list": [(spo["subject"], spo["predicate"], spo["object"]["@value"], spo["subject_type"], spo["object_type"]["@value"])for spo in line["spo_list"]]
            })
        return D


def sequence_padding(inputs, value=0, seq_dim=1):
    '''将序列padding到同一长度
    '''
    length = np.max([np.shape(x)[:seq_dim] for x in inputs], axis=0)  # 最大维度
    slices = [np.s_[:length[i]] for i in range(seq_dim)]  # 切片
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dim):
            pad_width[i] = (0, length[i]-np.shape(x)[i])
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        # pad_width：每个axis需要填充的数值数目；constant_values：填充值
        outputs.append(x)
    return np.array(outputs)


def pattern2index(pattern, sequence):
    '''在sequence序列中寻找子串pattern
    如果找到，返回第一个下标，否则返回-1。
    '''
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i+n] == pattern:
            return i
    return -1


class data_generator(Dataset):
    '''编码器
    继承自Dataset，需要重写init、len和getitem，供dataloader调用
    1. __len__：数据集的容量
    2. __getitem__：返回一条数据，其中，对数据的编码通过encoder函数实现
    '''

    def __init__(self, data, tokenizer, max_len, schema):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema

    def __len__(self):  # 数据集的容量
        return len(self.data)

    def encoder(self, item):
        '''编码器
        输入：load_name后的数据
        输出：原始文本，编码后的标签，编码后的句子和mask
        '''
        text = item["text"]  # 文本
        encoder_text = self.tokenizer(
            text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]  # 句子编码
        attention_mask = encoder_text["attention_mask"]  # mask
        spoes = set()
        # 编码后的三元组
        for s, p, o, s_t, o_t in item["spo_list"]:
            # add_special_tokens:特殊占位符
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = self.schema[s_t+"_"+p+"_"+o_t]
            o = self.tokenizer.encode(o, add_special_tokens=False)
            s_h = pattern2index(s, input_ids)
            o_h = pattern2index(o, input_ids)
            if s_h != -1 and o_h != -1:
                spoes.add((s_h, s_h+len(s)-1, p, o_h, o_h+len(o)-1))
        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        # 将编码后的三元组转为GPLinker所需要的标签形式
        for s_h, s_t, p, o_h, o_t in spoes:
            # 实体抽取：subject和object
            entity_labels[0].add((s_h, s_t))
            entity_labels[1].add((o_h, o_t))
            # head_labels：subject和object的head，tail_labels同理
            head_labels[p].add((s_h, o_h))
            tail_labels[p].add((s_t, o_t))
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0, 0))
        # 将标签对齐
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return text, entity_labels, head_labels, tail_labels, input_ids, attention_mask

    def __getitem__(self, index):  # 返回一条数据
        item = self.data[index]
        return self.encoder(item)

    @staticmethod
    def collate(examples):
        '''mini-batch生成方式
        将数据以batch的形式训练，并且在padding后转为tensor的形式
        '''
        batch_token_ids, batch_mask_ids, text_list, batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], [], [], [],[]
        # 生成batch
        for item in examples:
            # encoder函数的一条输出
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask = item
            # 标签
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            # 预测的文本
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            # 原始文本
            text_list.append(text)
        # 对batch进行padding，并转为tensor
        batch_token_ids = torch.tensor(
            sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_entity_labels = torch.tensor(
            sequence_padding(batch_entity_labels, seq_dim=2)).long()
        batch_head_labels = torch.tensor(
            sequence_padding(batch_head_labels, seq_dim=2)).long()
        batch_tail_labels = torch.tensor(
            sequence_padding(batch_tail_labels, seq_dim=2)).long()
        return text_list, batch_token_ids, batch_mask_ids,  batch_entity_labels, batch_head_labels, batch_tail_labels
