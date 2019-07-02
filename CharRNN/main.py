# coding:utf8
import sys, os
import torch as t
from torch.utils.data import DataLoader
from model import PoetryModel
from utils import Visualizer
from torch import nn
from tqdm import tqdm
import numpy as np

class Config(object):
    pickle_path = 'tang.npz'
    lr = 1e-3
    weight_decay = 1e-4
    epoch = 20
    batch_size = 128
    maxlen = 125
    plot_every = 20
    env = 'poetry'
    max_gen_len = 200
    model_path = r'E:/_fuxian/CharRNN/checkpoints/tang_2.pth'
    prefix_words = '细雨鱼儿出，微风燕子斜'
    start_words = '床前明月光'
    acrostic = False
    model_prefix = 'checkpoints/tang'

opt = Config()

def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    result = list(start_words)
    start_words_len = len(start_words)
    input = t.Tensor([word2ix['<START>']]).view(1,1).long()
    input.to(opt.device)
    # 输入</START>
    output, hidden = model(input)
    # 输入prefix_words
    if prefix_words:
        prefix_words_ix = [word2ix[word] for word in prefix_words]
        input = t.Tensor(prefix_words_ix).unsqueeze(0).long()
        output, hidden = model(input, hidden)

    start_words_ix = [word2ix[word] for word in start_words]
    input = t.Tensor(start_words_ix).unsqueeze(0).long()
    output, hidden = model(input, hidden)

    for i in range(opt.max_gen_len-start_words_len):
        input_ix = output.data[0].topk(1)[1][0].item()
        w = ix2word[input_ix]
        if w == '<EOP>':
            break
        result.append(w)
        input = t.Tensor([input_ix]).unsqueeze(0).long()
        output, hidden = model(input, hidden)
    return result

def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    results = []
    start_words_len = len(start_words)
    input = t.Tensor([word2ix['<START>']]).view(1,1).long()
    input = input.to(opt.device)

    index = 0
    pre_word = '<START>'

    # 输入</START>
    output, hidden = model(input)
    # 输入prefix_words
    if prefix_words:
        prefix_words_ix = [word2ix[word] for word in prefix_words]
        input = t.Tensor(prefix_words_ix).unsqueeze(0).long()
        output, hidden = model(input, hidden)
    output_index = output.data[0].topk(1)[1][0].item()

    for i in range(opt.max_gen_len):
        if pre_word in {'。','！','<START>'}:
            if index == start_words_len:
                break
            else:
                w = start_words[index]
                index += 1
                output_index = word2ix[w]
        else:
            input = t.Tensor([word2ix[output_index]]).view(1,1).long()
            output, hidden = model(input,hidden)
            output_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[output_index]

        results.append(w)
        pre_word = w
    return results


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)

    opt.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    device = opt.device
    vis = Visualizer(env=opt.env)

    # 获取数据
    data_all = np.load(opt.pickle_path)
    data = data_all['data']
    word2ix = data_all['word2ix'].item()
    ix2word = data_all['ix2word'].item()
    data = t.from_numpy(data)
    dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1)

    # 模型定义
    model = PoetryModel(len(word2ix), 128, 256)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    loss_func = nn.CrossEntropyLoss()
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location=t.device('cpu')))
    model.to(device)

    loss_avg = 0
    for epoch in range(opt.epoch):
        for ii, data_ in tqdm(enumerate(dataloader)):
            data_ = data_.long()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:,:-1], data_[:,1:]
            output, _ =model(input_)
            loss = loss_func(output, target.reshape(-1))
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()

            # 可视化
            if (ii+1) % opt.plot_every == 0:
                vis.plot('loss', loss_avg/opt.plot_every)
                loss_avg = 0
                poetrys = [[ix2word[_word] for _word in data_[i].tolist()] for i in range(data_.shape[0])][:16]
                vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]),win='origin_poem')

                gen_poetries = []
                for word in list('春江花月夜凉如水'):
                    gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                    gen_poetries.append(gen_poetry)
                vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]), win='gen_poem')

        t.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))

def gen(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)
    data_all = np.load(opt.pickle_path)
    data = data_all['data']
    word2ix = data_all['word2ix'].item()
    ix2word = data_all['ix2word'].item()
    model = PoetryModel(len(word2ix), 128, 256)
    model.load_state_dict(t.load(opt.model_path, map_location=t.device('cpu')))
    opt.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    model.to(opt.device)
    if opt.start_words.isprintable():
        start_words = opt.start_words
        prefix_words = opt.prefix_words if opt.prefix_words else None
    else:
        start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
        prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode(
            'utf8') if opt.prefix_words else None

    start_words = start_words.replace(',', '，') .replace('.', '。').replace('?', '？')

    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result))

if __name__ == '__main__':
    import fire
    fire.Fire()




