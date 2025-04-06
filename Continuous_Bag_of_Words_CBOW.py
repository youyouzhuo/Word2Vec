# Continuous_Bag_of_Words_CBOW.py
import os
import sys
import argparse
import json
import time
from collections import defaultdict, Counter
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ======================== 路径配置 ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "frankenstein_with_splits.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model_storage/ch5/cbow")


# ======================== 工具函数 ========================
def validate_file(path):
    """验证文件是否存在"""
    if not os.path.exists(path):
        print(f"错误：文件 '{path}' 不存在！")
        print("请检查：1. 文件路径 2. 文件名拼写 3. 文件扩展名")
        sys.exit(1)
    return path


# ======================== 数据预处理 ========================
class Vocabulary:
    def __init__(self, token_to_idx=None, unk_token="<UNK>", mask_token="<MASK>", add_unk=True, add_mask=True):
        self._token_to_idx = token_to_idx or {}
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._unk_token = unk_token
        self._mask_token = mask_token

        self.unk_index = self.add_token(unk_token) if add_unk else -1
        self.mask_index = self.add_token(mask_token) if add_mask else -1

    def add_token(self, token):
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        index = len(self._token_to_idx)
        self._token_to_idx[token] = index
        self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        return self._idx_to_token.get(index, self._unk_token)

    def __len__(self):
        return len(self._token_to_idx)


# ======================== 数据集类 ========================
class CBOWDataset(Dataset):
    def __init__(self, cbow_df, vectorizer, max_seq_length=-1):
        self.vectorizer = vectorizer
        self.max_seq_length = max_seq_length

        # 计算最大序列长度
        if max_seq_length < 0:
            self.max_seq_length = max(len(context.split())
                                      for context in cbow_df['context'])

        # 保存数据
        self.target_df = cbow_df[['target']]
        self.context_df = cbow_df[['context']]

    @classmethod
    def load_dataset_and_make_vectorizer(cls, csv_path):
        """加载数据集并创建向量化器"""
        csv_path = validate_file(csv_path)
        cbow_df = pd.read_csv(csv_path)
        train_cbow_df = cbow_df[cbow_df.split == 'train']
        return cls(train_cbow_df, CBOWVectorizer.from_dataframe(train_cbow_df))

    def get_vectorizer(self):
        return self.vectorizer

    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, index):
        target_row = self.target_df.iloc[index]
        context_row = self.context_df.iloc[index]

        context_vector = self.vectorizer.vectorize(context_row['context'], self.max_seq_length)
        target_index = self.vectorizer.cbow_vocab.lookup_token(target_row['target'])

        return {
            'x_data': context_vector,
            'y_target': target_index
        }


# ======================== 向量化器 ========================
class CBOWVectorizer:
    def __init__(self, cbow_vocab):
        self.cbow_vocab = cbow_vocab

    def vectorize(self, context, vector_length=-1):
        indices = [self.cbow_vocab.lookup_token(token) for token in context.split()]
        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.cbow_vocab.mask_index

        return out_vector

    @classmethod
    def from_dataframe(cls, cbow_df):
        vocab = Vocabulary()
        for index, row in cbow_df.iterrows():
            for token in chain(row['context'].split(), [row['target']]):
                vocab.add_token(token)
        return cls(vocab)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({'vocab': self.cbow_vocab._token_to_idx}, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath) as f:
            vocab_data = json.load(f)
        vocab = Vocabulary(token_to_idx=vocab_data['vocab'])
        return cls(vocab)


# ======================== 模型定义 ========================
class CBOWClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=padding_idx)
        self.fc1 = nn.Linear(in_features=embedding_size,
                             out_features=vocab_size)

    def forward(self, x_in):
        x_embedded_sum = self.embedding(x_in).sum(dim=1)
        y_out = self.fc1(x_embedded_sum)
        return y_out


# ======================== 训练工具 ========================
def make_train_state(args):
    return {
        'stop_early': False,
        'early_stopping_step': 0,
        'early_stopping_best_val': float('inf'),
        'epoch_index': 0,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }


def update_train_state(args, model, train_state):
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False
    elif train_state['epoch_index'] >= 1:
        loss_tm1 = train_state['val_loss'][-2]
        loss_t = train_state['val_loss'][-1]
        if loss_t >= loss_tm1:
            train_state['early_stopping_step'] += 1
        else:
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = loss_t
                train_state['early_stopping_step'] = 0
        train_state['stop_early'] = (train_state['early_stopping_step'] >= args.early_stopping_criteria)
    return train_state


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


# ======================== 主函数 ========================
def main(args):
    # 初始化环境
    os.makedirs(MODEL_DIR, exist_ok=True)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA: {args.device.type == 'cuda'}")

    # 加载数据集
    print("Loading dataset and creating vectorizer")
    dataset = CBOWDataset.load_dataset_and_make_vectorizer(DEFAULT_CSV_PATH)
    vectorizer = dataset.get_vectorizer()

    # 初始化模型
    classifier = CBOWClassifier(
        vocab_size=len(vectorizer.cbow_vocab),
        embedding_size=args.embedding_size,
        padding_idx=vectorizer.cbow_vocab.mask_index
    ).to(args.device)

    # 训练配置
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

    # 训练状态
    train_state = make_train_state(args)
    train_state['model_filename'] = os.path.join(MODEL_DIR, "model.pth")

    # 训练循环
    try:
        for epoch_index in range(args.num_epochs):
            # 训练阶段
            classifier.train()
            batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)

            running_loss = 0.0
            running_acc = 0.0

            for batch_index, batch_dict in enumerate(tqdm(batch_generator)):
                optimizer.zero_grad()

                y_pred = classifier(batch_dict['x_data'])
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss.backward()
                optimizer.step()

                running_loss += (loss.item() - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # 更新训练状态
            train_state = update_train_state(args, classifier, train_state)
            if train_state['stop_early']:
                break

            print(f"Epoch {epoch_index + 1}/{args.num_epochs}")
            print(f"Train Loss: {running_loss:.3f} | Acc: {running_acc:.1f}%")

    except KeyboardInterrupt:
        print("训练被用户中断")

    # 保存最终模型和向量化器
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.json")
    vectorizer.save(vectorizer_path)
    print(f"模型和向量化器已保存至：{MODEL_DIR}")


def generate_batches(dataset, batch_size, device="cpu"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbow_csv", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--model_storage", type=str, default=MODEL_DIR)
    parser.add_argument("--embedding_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--early_stopping_criteria", type=int, default=5)
    args = parser.parse_args()

    # 验证路径
    validate_file(DEFAULT_CSV_PATH)
    main(args)