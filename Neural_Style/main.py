import torch as t
import torchvision as tv
import PIL

from torch.utils import data
from model import TransformerNet, Vgg16
import utils
from torch.nn import functional as F
from tqdm import tqdm

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Config(object):
    image_size = 256
    batch_size = 2
    data_root = r'E:/pytorch-book/chapter8-Neural Style/data/'
    num_workers = 4

    style_path = r'style.jpg'
    lr = 1e-3

    env = 'neural-style'
    plot_every = 10
    save_every = 40

    epoches = 2

    content_weight = 1e5
    style_weight =1e10

    model_path = None
    content_path = 'input.png'
    result_path = 'output.png'

def train(**kwargs):
    opt = Config()
    for _k,_v in kwargs.items():
        setattr(opt, _k, _v)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    vis = utils.Visualizer(opt.env)

    # 数据加载
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x:x*255)
    ])
    dataset = tv.datasets.ImageFolder(opt.data_root, transforms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    # 风格转换网络
    transformer = TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path, map_location=t.device('cpu')))
    transformer.to(device)

    # 损失网络 Vgg16
    vgg = Vgg16().eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    # 优化器
    optimizer = t.optim.Adam(transformer.parameters() ,opt.lr)

    # 获取风格图片的数据
    style = utils.get_style_data(opt.style_path)
    vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    style = style.to(device)

    # 风格图片的gramj矩阵
    with t.no_grad():
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]

    # 损失统计
    style_loss_avg = 0
    content_loss_avg = 0

    for epoch in range(opt.epoches):
        for ii, (x, _) in tqdm(enumerate(dataloader)):

            # 训练
            optimizer.zero_grad()
            x = x.to(device)
            y = transformer(x)
            # print(y.size())
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            features_x = vgg(x)
            features_y = vgg(y)

            # content loss
            content_loss = opt.content_weight * F.mse_loss(features_y.relu3_3, features_x.relu3_3)

            # style loss
            style_loss = 0
            for ft_y, gm_s in zip(features_y, gram_style):
                with t.no_grad():
                    gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            content_loss_avg += content_loss.item()
            style_loss_avg += style_loss.item()

            if (ii+1) % opt.plot_every == 0:
                vis.plot('content_loss', content_loss_avg/opt.plot_every)
                vis.plot('style_loss', style_loss_avg/opt.plot_every)
                content_loss_avg = 0
                style_loss_avg = 0
                vis.img('output', (y.data.cpu()[0] * 0.225 +0.45).clamp(min=0,max=1))
                vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

            if (ii+1) % opt.save_every == 0:
                vis.save([opt.env])
                t.save(transformer.state_dict(), 'checkpoints/%s_style.pth' % (ii+1))

@t.no_grad()
def stylize(**kwargs):
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    # 图片处理
    content_image = PIL.Image.open(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x*255)
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    # 模型
    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=t.device('cpu')))
    style_model.to(device)

    # 风格迁移和保存
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image((output_data/255).clamp(min=0, max=1), opt.result_path)


if __name__ == "__main__":
    import fire
    fire.Fire()