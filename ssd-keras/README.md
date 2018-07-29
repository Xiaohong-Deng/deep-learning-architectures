## SSD: Single-Shot MultiBox Detector implementation in Keras
---

1. [Anchor Boxes and Data Flow](./docs/anchor_boxes.md) (Under Construction)

For detailed information about this architecture please refer to the original paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

[ssd_keras by pierluigiferrari](https://github.com/pierluigiferrari/ssd_keras) provides detailed explanation on SSD which complements the paper.


## Training Details
### How to load pretrained base network weights
To use a base network in SSD, we need to convert the based network to its FCN version, i.e., replace Dense layers with their Conv2D versions.

We can first load a model with its pretrained weights. Keras provides VGG16 or MobileNet by default. Next we build a new model, iterate over the old model layers and convert layers if necessary. Code to accomplish this is as follows
```python
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.engine import InputLayer
import keras

def to_fully_conv(model):

    new_model = Sequential()

    input_layer = InputLayer(input_shape=(None, None, 3), name="input_new")

    new_model.add(input_layer)

    for layer in model.layers:

        if "Flatten" in str(layer):
            flattened_ipt = True
            f_dim = layer.input_shape

        elif "Dense" in str(layer):

            input_shape = layer.input_shape
            output_dim =  layer.get_weights()[1].shape[0]
            W,b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1],f_dim[2],f_dim[3],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (f_dim[1],f_dim[2]),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])
                flattened_ipt = False

            else:
                shape = (1,1,input_shape[1],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (1,1),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])


        else:
            new_layer = layer

        new_model.add(new_layer)

    return new_model
```

Then we can use `model.save_weights` to save weights from the new model. After we build our SSD model we can load those weights to the SSD model and start training.
