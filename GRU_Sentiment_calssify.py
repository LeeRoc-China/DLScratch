import re
import time
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.vocab import build_vocab_from_iterator


class Imdb_Datasets(Dataset):
    """
    when u want to init this instance,
    the file path should be father/train(test)/datafile
    """

    def __init__(self, data_path: str, train=True):
        super(Imdb_Datasets, self).__init__()
        self._train_data_path = os.path.join(data_path, "train")
        self._test_data_path = os.path.join(data_path, "test")
        self._temp_data_path = self._train_data_path if train else self._test_data_path

        self.temp_data_path = [os.path.join(self._temp_data_path, 'pos'), os.path.join(self._temp_data_path, 'neg')]
        self.total_data_path_list = []
        for path in self.temp_data_path:
            self.total_data_path_list.extend([os.path.join(path, j) for j in os.listdir(path) if j.endswith('.txt')])

    def __len__(self):
        return self.total_data_path_list.__len__()

    def __getitem__(self, index):
        path = self.total_data_path_list[index]
        label_str = path.split('\\')[-2]
        label = [1, 0] if label_str == 'neg' else [0, 1]
        content = pd.read_csv(path, sep='\t')

        return content.columns[0], label


# todo: define a NLP network to address sentiment classify problem
class Imdb_Sentiment_classify(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Imdb_Sentiment_classify, self).__init__()
        self.hidden_size = 64
        self.dropout = 0.5
        self.num_layer = 2

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.gru = nn.GRU(input_size=embed_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layer,
                          dropout=self.dropout)
        self.fc = nn.Sequential(nn.Linear(self.hidden_size, 128),
                                nn.ReLU(),
                                nn.Linear(128, 2),
                                nn.Softmax(dim=1)
                                )
        self.init_weight()

    def init_weight(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, text, offsets):
        """
        注意：在embedding后，数据的维度是[batch_size, embed_size]，
        需要变成[batch_size, sequence_length, input_size]，以此来增加以满足训练的要求
        参考： https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU，
              https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        forward 这样写在Debug时，可以更加直观的看到每一层的输出
        """
        x = self.embedding(text, offsets)
        x = torch.unsqueeze(x, dim=1)
        out_, H_n = self.gru(x, None)
        output_ = self.fc(out_)
        output = torch.squeeze(output_, dim=1)
        return output


def tokenize(text):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    text = re.sub("<.*?>", " ", text)
    text = re.sub("|".join(fileters), " ", text)
    return [i.strip().lower() for i in text.split()]


def yield_tokens(data_iter):
    """
    To processing texts2tokens
    """
    for text, label in data_iter:
        yield tokenize(text)


def collate_batch(batch):
    """
    This will be use by DataLoader,
    which used to processing a batch size of datas
    :rtype: object
    """
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def train(dataloader, epo):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label.to(torch.float32))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label.argmax(1)).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epo, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label.to(torch.float32))
            total_acc += (predicted_label.argmax(1) == label.argmax(1)).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter = iter(Imdb_Datasets(r"imdb"))
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenize(x))
label_pipeline = lambda x: x

PATH = "imdb"

# todo: Hyper-parameter to net
vb_size = len(vocab)
emsize = 128
LR = 5
EPOCH = 30
BATCH_SIZE = 64

train_imdb_Dataset = Imdb_Datasets(PATH, train=True)
test_imdb_Dataset = Imdb_Datasets(PATH, train=False)

num_train = int(len(test_imdb_Dataset) * 0.95)
split_train_, split_valid_ = random_split(test_imdb_Dataset, [num_train, len(test_imdb_Dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_dataloader = DataLoader(test_imdb_Dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
val_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

model = Imdb_Sentiment_classify(vocab_size=vb_size, embed_dim=emsize).to(device)
print(model)
for i, j, k in train_dataloader:
    print(f"label:{i.shape}\ntext:{j.shape}\noffsets:{k.shape}")
    output = model(j, k)
    print(f"output shape: {output.shape}")
    print("-" * 10 + "show some detail" + "-" * 10)
    break

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

total_accu = None
for epoch in range(1, EPOCH + 1):
    epoch_start_time = time.time()
    train(train_dataloader, epo=epoch)
    accu_val = evaluate(val_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

torch.save(model.state_dict(), os.path.join(PATH, "Imdb_model.pth"))
