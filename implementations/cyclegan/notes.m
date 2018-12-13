笔记翻译，fork from https://gist.github.com/brannondorsey/fb075aac4d5423a75f57fbf7ccc12124
1.Euclidean distance between predicted and ground truth pixels is not a good method of judging similarity because it yields blurry images.
欧式距离不是判断两个图像相似的一个好方法，因为predicted 图像有模糊。
2. GANs learn a loss function rather than using an existing one.
GANs网络是自己学习一个损失函数而不是用已存的。
3. GANs learn a loss that tries to classify if the output image is real or fake, while simultaneously training a generative 
model to minimize this loss.
GANs学习一个loss来判别输出图像的真假同时训练一个G使这个损失函数最小。
4.The generator G is trained to produce outputs that cannot be distinguished from "real" images by an adversarially trained 
discrimintor, D which is trained to do as well as possible at detecting the generator's "fakes".The discriminator D, learns 
to classify between real and synthesized pairs. The generator learns to fool the discriminator.
G的训练目的是产生使D无法判别真伪的图像，D的目的是尽可能鉴别哪些图像是G产生的”fake”图。即：G骗D，D辨别G。
5. Asks G to not only fool the discriminator but also to be near the ground truth output in an L2 sense.
让G不仅能够骗过D也能使产生的图像按照L2 loss更接近真图。
6. L1 distance between an output of G is used over L2 because it encourages less blurring.
L1比L2带来的图像blurring更少
7. Without z, the net could still learn a mapping from x to y but would produce deterministic outputs (and therefore fail to 
match any distribution other than a delta function. Past conditional GANs have acknowledged this and provided Gaussian noise 
z as an input to the generator, in addition to x)

Either vanilla encoder-decoder or Unet can be selected as the model for G in this implementation.
Both generator and discriminator use modules of the form convolution-BatchNorm-ReLu.
A defining feature of image-to-image translation problems is that they map a high resolution input grid to a high resolution 
output grid.

Input and output images differ in surface appearance, but both are renderings of the same underlying structure. Therefore, 
structure in the input is roughly aligned with structure in the output.

L1 loss does very well at low frequencies (I think this means general tonal-distribution/contrast, color-blotches, etc) but 
fails at high frequencies (crispness/edge/detail) (thus you get blurry images). This motivates restricting the GAN 
discriminator to only model high frequency structure, relying on an L1 term to force low frequency correctness. 
L1 loss对低频信息处理(色调分布、色斑)较好，但是不利于高频（琐碎、边缘、细节），所以G的输出会模糊。这就促使GAN的D对高频信息建模，L1来校正低频信息。
In order to model high frequencies, it is sufficient to restrict our attention to the structure in local image patches. 
Therefore, we design a discriminator architecture – which we term a PatchGAN – that only penalizes structure at the scale of 
patches. This discriminator tries to classify if each NxN patch in an image is real or fake. 
为了应对图像bulrring问题，即高频模型，GANs加入PatchGAN，把G输出的图像分成N×N个patch，判断每个patch的真假。
We run this discriminator convolutationally across the image, averaging all responses to provide the ultimate output of D.
我们用patchGAN对整副图像卷积一样来判断每个patch，最终取均值来判断真假作为D的输出。
Because PatchGAN assumes independence between pixels seperated by more than a patch diameter (N) it can be thought of as a 
form of texture/style loss.
因为PatchGAN假设：被大于patch直径（N）所分开的像素是互不影响的，这被视为纹理/风格损失函数。？？？？？？？？
To optimize our networks we alternate between one gradient descent step on D, then one step on G (using minibatch SGD applying
the Adam solver)。为了优化我们的网络，D和G的梯度下降交替进行（）
In our experiments, we use batch size 1 for certain experiments and 4 for others, noting little difference between these two conditions.
在我们的实验中，某些实验batch_size设为1,其他设为4,这两个条件下产生的实验结果几乎没区别。
To explore the generality of conditional GANs, we test the method on a variety of tasks and datasets, including both graphics 
tasks, like photo generation, and vision tasks, like semantic segmentation.
为了测试条件GAN网络的一般性，我们在多个任务和数据集上进行了测试：图像任务（图像生成），视觉任务（语义分割）。
Evaluating the quality of synthesized images is an open and difficult problem. Traditional metrics such as per-pixel mean-
squared error do not assess joint statistics of the result, and therefore do not measure the very structure that structured 
losses aim to capture.
对生成的图像进行质量测评是一个open且difficult的问题。传统标准是用每个像素的均方差，但是它不能衡量联合分布，因此不能判断loss的目的是否达到。
FCN-Score: while quantitative evaluation of generative models is known to be challenging, recent works have tried using 
pre-trained semantic classifiers to measure the discriminability of the generated images as a pseudo-metric. 
FCN-Score：尽管模型的定量评估比较难，最近一些工作尝试了预训练的语义分类器作为“伪-度量”衡量生成图像的可辨别性。
The intuition is that if the generated images are realistic, classifiers trained on real images will be able to classify the 
synthesized image correctly as well.
判断依据是：如果生成的图像是真实的，那么在real images上训练的分类器也能辨别其真伪。
cGANs seems to work much better than GANs for this type of image-to-image transformation, as it seems that with a GAN, the 
generator collapses into producing nearly the exact same output regardless of the input photograph.
cGANs（这类conditional GAN）
16x16 PatchGAN produces sharp outputs but causes tiling artifacts, 70x70 PatchGAN alleviates these artifacts. 256x256 ImageGAN
doesn't appear to improve the tiling artifacts and yields a lower FCN-score.

An advantage of the PatchGAN is that a fixed-size patch discriminator can be applied to arbitrarily large images. This allows 
us to train on, say, 256x256 images and test/sample/generate on 512x512.
PatchGAN的好处是：固定大小的PatchGAN可以应用于任意大的图像，我们可以用256×256的训练而用512×512的图像测试。
cGANs appear to be effective on problems where the output is highly detailed or photographic, as is common in image processing
and graphics tasks.
cGANs对于图像处理和任务中的高细节/逼真图像有一定作用。
When semantic segmentation is required (i.e. going from image to label) L1 performs better than cGAN. We argue that for vision
problems, the goal (i.e. predicting output close to ground truth) may be less ambiguous than graphics tasks, and reconstruction
losses like L1 are mostly sufficient.
L1的语义分割效果比cGAN要好。视觉问题的目标（比如预测接近ground truth的输出）比图形任务模糊，重构损失函数例如L1大多是有效的。
Conclusion
The results in this paper suggest that conditional adversarial networks are a promising approach for many image-to-image 
translation tasks, especially those involving highly structured graphical outputs. These networks learn a loss adapted to the 
task and data at hand, which makes them applicable in a wide variety of settings.

Misc
Least absolute deviations (L1) and Least square errors (L2) are the two standard loss functions, that decides what function
should be minimized while learning from a dataset. (source)
How, using pix2pix, do you specify a loss of L1, L1+GAN, and L1+cGAN?
