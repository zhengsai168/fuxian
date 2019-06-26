# coding:utf8
import os
import ipdb
import torch as t
import torchvision as tv
from tqdm import tqdm
from model import NetG, NetD
from torchnet.meter import AverageValueMeter


class Config(object):
    data_path = 'E:/pytorch-book/chapter7-GAN/data'  # 数据集存放路径
    num_workers = 4  # 多进程加载数据所用的进程数
    image_size = 96  # 图片尺寸
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4  # 生成器的学习率
    lr2 = 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    gpu = False  # 是否使用GPU
    nz = 100  # 噪声维度
    ngf = 64  # 生成器feature map数
    ndf = 64  # 判别器feature map数

    save_path = 'imgs/'  # 生成图片保存路径

    vis = True  # 是否使用visdom可视化
    env = 'GAN'  # visdom的env
    plot_every = 20  # 每间隔20 batch，visdom画图一次

    debug_file = '/tmp/debuggan'  # 存在该文件则进入debug模式
    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    save_every = 2  # 2个epoch保存一次模型
    netd_path = None  # 'checkpoints/netd_.pth' #预训练模型
    netg_path = None  # 'checkpoints/netg_211.pth'

    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差


opt = Config()

def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device('cuda') if opt.gpu else t.device('cpu')
    # 可视化
    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)

    # 数据
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True)

    netg, netd = NetG(opt), NetD(opt)
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=t.device('cpu')))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=t.device('cpu')))
    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)

    # 真图片label为1，假图片label为0
    # noise为生成网络的输入
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    errord = 0
    errorg = 0

    epochs = range(opt.max_epoch)
    for epoch in epochs:
        for ii, (img, _) in tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            if (ii+1) % opt.d_every == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                # 把真图片判断为正确
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                # 把假图片(netg通过噪声生成的图片)判断为错误
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                errord += (error_d_fake + error_d_real).item()

            if (ii+1) % opt.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg += error_g.item()

            if opt.vis and (ii+1) % opt.plot_every == 0:
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('errord', errord/(opt.plot_every))
                vis.plot('errorg', errorg/(opt.plot_every))
                errord = 0
                errorg = 0

            if (epoch+1) % opt.save_every == 0:
                tv.utils.save_image(fix_fake_imgs.data[:64], '%s%s.png' % (opt.save_path, epoch), normalize=True,
                                    range=(-1,1))
                t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % (epoch+1))
                t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % (epoch+1))

if __name__ == '__main__':
    import fire
    fire.Fire()









