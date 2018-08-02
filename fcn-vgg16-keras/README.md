## Fully Convolutional Networks Based on VGG16

Please refer to the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038v2)
The following content assumes you have read the paper.

### Fully Connected Layers and Their Convolutional Equivalents
It may not be obvious but take VGG16 as an example, the fully connected layers can be viewed as convolutional layers. There are 3 such layers in VGG16. In the original design the input of the first such layer is of the size 7x7x512 and its output size 4096. The equivalent conv2D layer is a layer with 4096 kernels of size 7x7x512, a kernel covering the entire input feature map.

The 2nd FC takes input of size 4096 and output tensor of size 4096. In that case, the converted conv2D layer has 4096 kernels of size 1x1x4096

For the last FC, also called classifier layer, it requires a conv2D layers with n kernels of size 1x1x4096 where n is the number of classes.

### The Benefits of the Conv2D Equivalents
The converted conv2D layers can take inputs of any size. If input size is larger, it is like doing classification on patches. Note the input patches are overlapped, if you are to use the original network to do the classification for each of the patches the computation cost will be higher. This was mentioned in section 3.1 in the original paper.

### Upsampling by Deconvolutional Layer
In section 4.1 of the original paper, it is said after the modification metioned above a deconvolutional layer should be appended to the architecture to upsample the pixel-wise classified feature map by a factor of 32. This is called FCN-32S.

### Skip Architecture
Such upsampled feature map is coarse. To get a finer-grained output, we take the already predicted output of Pool5 and upsampled it by a factor of 2. We take output of Pool4 and predict it using 1x1 conv2D. Add these two. If we append a upsampling layer with a factor of 16, the netwrok is known as FCN-16S.

Optionally, you can do the similar to the output of Pool3 and the result above. Upsample the final result as the output of the architecture requires with a factor of 8. This is called FCN-8S.

### Shape of Deconvolutional Layer Output

One thing to note is when you upsample (deconv) an input which is a result of prediction over some output of stride 2 pooling layer, the result of upsampling has undefined height and width even if you have the shape of the input (output width and height of pooling layer)

The conv2D_transpose can be viewed as a reverse process of conv2D. When conv2D has padding='same', `output = (input + stride - 1) // stride`. If `stride = 2`, input size of 33 and 34 both give output size of 16, so the process is not invertible. As a sidenote, you can see that for `stride = 1, padding='same'` output is of the same size of input.

For `tf.nn.conv2d_transpose` we can specify `output_size`. For `tf.layers.conv2d_transpose` there is not such a parameter defined. As suggested by [this Github issue](https://github.com/tensorflow/tensorflow/issues/19236) it is not addressed by Tensorflow team. The suggested workaround is using `tf.set_shape`. Another option is `tf.reshape`. This is [another related Github issue](https://github.com/tensorflow/tensorflow/issues/833#issuecomment-278016198) with solutions provided.
