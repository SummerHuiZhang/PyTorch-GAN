Pytorch-GAN：
2018.10.10
2018年9月10日
Paul Newman课题组的future work之一： 把离散的图像转换视觉定位变成连续的，用同一个生成器Generator，这不就是StarGAN在做的吗？StarGAN用论文中的数据集测试一下。

https://github.com/SummerHuiZhang/PyTorch-GAN 将CycleGAN的几个程序copy到本地Mac 然后scp到了二楼服务器，跑通了 Q1：没有保存models Q2: 怎么test

A1： 没读取图像没进入图像处理的命令原因： 图像路径需要写绝对路径 sudo chmod 777 -R images/ 是更改文件夹的权限

没保存images和models的原因: 保存也需要写绝对路径

A2：已经写了一半的不知道甩哪里去了的

2018.10.11上午 学习一遍Pytorch—GAN中cyclegan.py简单明了好改 参照博客https://blog.csdn.net/weixin_42445501/article/details/81234281

(1) parser.add_argument 读入多个命令行参数 print(opt)按照argument的字母表顺序，打印出来

(2) torch的多个loss函数 https://blog.csdn.net/qq_16305985/article/details/79101039

(3) ReplayBuffer：untils中定义的 生成器生成的 fake 图片还要经过另一生成器，生成 cycle 图片，所以通过该buffer函数寄存 fake 图片，用于判别器更新 line55: patch = (1, opt.img_height // 24, opt.img_width // 24)得到的patch大小为（1，16，16）

(4) line91 optimizer_G = torch.optim.Adam torch.optim是一个实现了多种优化算法的包，大多数通用的方法都已支持，提供了丰富的接口调用，未来更多精炼的优化算法也将整合进来 为了使用torch.optim，需先构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数 你必须给它一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表。 然后，您可以指定程序优化特定的选项，例如学习速率，权重衰减等。

(5) PatchGAN的思想是，既然GAN只负责处理低频成分，那么判别器就没必要以一整张图作为输入，只需要对NxN的一个图像patch去进行判别就可以了。这也是为什么叫Markovian discriminator，因为在patch以外的部分认为和本patch互相独立。

(6)utils.py中的assert assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!" assert相当于断句，如果不符合>0时就报后半句的错误，这是个好习惯。 https://www.cnblogs.com/liuchunxiao83/p/5298016.html

(7) import random; random.uniform(x,y)随机生成介于[x,y)之间的数 (8) 三种损失函数 criterion_GAN(MSE): criterion_cycle(L1): criterion_identity(L1):

(8) line153: for i, batch in enumerate(dataloader): enumerate的用法如： seasons = ['Spring', 'Summer', 'Fall', 'Winter']; list(enumerate(seasons, start=1)) #下标从 1 开始; [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]

test阶段
运行命令

sh test.sh test.sh内容如下：

cd /home/timing/Git_Repos_Summer/PyTorch-GAN/implementations/cyclegan python3 predict_image.py --epoch 3 --dataset_name Alderley_Night2Day --img_height 640 --img_width 260 --checkpoint_interval 10 --epoch 42 --result_path test_results/Alderley_Night2Day

predict_image.py是王写的test程序，和training类似，区别是直接load某一个model

test时候要改的参数：

test.sh中的“--”都要注意，尤其是result_path
predict.py里面的 line119：“val_dataloader” line134：“save_image(fake_B,opt.result_path+” Bug：
“Error(s) in loading state_dict for DataParallel: Missing key(s) in state_dict:”

Debug： 详见帖子 解决方法：

把predict_image.py中的multi_gpu=true改成了False
把test.sh也改成了docker运行 同样使用CUDA_VISIBLE_DEVICES=0，1，3
名称G_AB_0.pth无需更改为0.pth
小实验
1.名称G_AB_0.pth需要更改为0.pth吗？ 通过实验对比load改名前后的models发现名称不影响 2.直接把test.py中的图像尺寸再写反一次会怎么样？

Los Angeles回来 what should I continue with my research work?

To Do
如何把提取特征的深度网络如OverFeat和VGG放到CycleGAN？
如何放入深度网络后设计loss？ 3.怎么根据realA realB文件夹读入的图像名称得到图像对应的OverFeat特征，两个文件之间的检索？
