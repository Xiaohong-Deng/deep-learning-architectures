### Anchor Boxes and Data Flow

Notice the following paragraph in `README.md` at [ssd_keras by pierluigiferrari](https://github.com/pierluigiferrari/ssd_keras)
> This may or may not be obvious to you, but it is important to understand that it is not possible for the model to predict absolute coordinates for the predicted bounding boxes. ..., In order to be able to predict absolute box coordinates, the convolutional layers responsible for localization would need to produce different output values for the same object instance at different locations within the input image. This isn't possible of course: For a given input to the filter of a convolutional layer, the filter will produce the same output regardless of the spatial position within the image because of the shared weights. This is the reason why the model predicts offsets to anchor boxes instead of absolute coordinates, and why during training, absolute ground truth coordinates are converted to anchor box offsets in the encoding process.

Somebody also confused about this and asked the author. The author answered it beautifully. Here is [the ticket](https://github.com/pierluigiferrari/ssd_keras/issues/127)

Basically if two identical objects take up two spatially different cells, a filter should be able to identify them and produce the anchor box coordinates for them. Because they are identical the dot products of the two cells and the filter are the same. But spatial locations for them are different. So producing absolute coordinates is not going to work. That is why we need offsets. It should be intuitive that two cells containing the same pixels should produce the same offsets.

### Encoding and Decoding Boxes
In the paper it was not elaborated that how you can mapping anchor boxes and ground truth boxes back and forth between the input images and feature maps of different sizes. How you can apply scaling factors to the processes? Why larger feature maps have smaller scaling factors, e.g. 0.2 and smaller feature maps have larger ones, e.g., 0.9? I call them "box problem". They are closely connected to the processes so called encoding and decoding boxes.

The definition of **encoding boxes** is as follows
> Encoding boxes means converting ground truth labels into the target format that the loss function needs during training

The definition of **decoding boxes** is as follows
> Decoding boxes means converting raw model output back to the input label format, which entails various conversion and filtering processes such as non-maximum suppression (NMS). ..., In 'training' mode, the model outputs the raw prediction tensor that still needs to be post-processed with coordinate conversion, confidence thresholding, non-maximum suppression, etc.

In most of the following explanation I have to walk through the data flow of the keras implementation to understand the box problem, please bear with me.

### Decoding Boxes （from feature maps to input images）
`SSDEncoder#generate_anchor_boxes_for_layer` does the following

To get anchor box size in the original input image, given `size = min(image_width, image_height)`, do `box_height = size * scale / np.sqrt(aspect_ratio)`. So for the larger feature maps you get smaller anchor boxes and for the smaller you get larger.

The pixel distance between two adjacent anchor boxe centers is defined as `step_height = image_height / feature_map_height` and `step_width = image_width / feature_map_height`

Then center point of an anchor box in the feature map is defined as `cx = (i + 0.5) / f_k` and `cy = (j + 0.5) / f_k` where i and j stand for the pixel indices in the k-th feature map or the cell indices in the input image, `f_k` stands for the filter size of the k-th feature map. 0.5 is the offset of the center relative to the topleft corner of the anchor box.

To retrieve center point coordinates in the input image from `cx` and `cy`, `cx_decoded = cx * image_width` and `cy_decoded = cy * image_height`

You can easily compute `(xmin, ymin, xmax, ymax)` for a bounding box using `step_height`, `step_width` and center points.

### model constructure in keras_ssd7.py

There are predictor layers `classes4`, `classes5`, `classes6` and `classes7` for predicting classes for feature maps of different sizes. Output shape is `(batch, height, width, n_boxes * n_classes)`. Note height and width here are from the feature maps, not the input images. The predition is per class per box per pixel in the feature maps.

There are predictor layers `boxes4`, `boxes5`, `boxes6` and `boxes7` for predicting anchor boxes offsets. Output shape is `(batch, height, width, n_boxes * 4)`. Note height and width are from the feature maps, not the input images. The prediction is per offset in `(cx, cy, w, h)` per box per pixel in the feature maps.

There are anchor box layers `anchors4`, `anchors5`, `anchors6` and `anchors7` producing absolute values of `(cx, cy, w, h)` for anchor boxes in the input images and their corresponding variances. Output shape is `(batch, height, width, n_boxes, 8)`.

### Encoding boxes

#### data formats in data flow
The raw data is of the form of raw images and a label file in which each row contains information for one bounding box in the form of `(image_name, xmin, xmax, ymin, ymax, class_id)`. Because image name is included in the label file, training, validation and test data can be put together in one folder. Note the class_id is integer.

By default, `parse_csv` generate a list in which each element is of the form `(image_name, 'class_id', 'xmin', 'ymin', 'xmax', 'ymax')` and the list is sorted on image names. It iterates the list and generate 4 lists named images, filenames, labels and image_ids where k-th element in each of the lists corresponds to k-th image, image file filename, all the class and coordinates of the bounding box info associated with the image and image id.

Note the loop that generates the 4 lists can handle unsorted list, but that would lead to images and filenames lists containing duplicates. If sorted, an image should be able to get all labels associated with it once and for all. In that case, not duplicates.

>             label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given object
                detection model needs as its training targets.

>             `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vecotrs in
            the last axis are the box coordinates, the next four elements after that are just dummy elements, and
            the last four elements are the variances.

`generate` takes self.labels (inherited from labels list in parse_csv). Each element in self.labels is of the form [class_id, 'xmin', 'ymin', 'xmax', 'ymax'], representing the class and coordinates of a bounding box.

`ssd_encoder.__call__` is called in `generate`. `self.labels` are divided to batches and fed to `ssd_encoder.__call__` as `ground_truth_labels`. The ground truth data is transformed to the form of `(batch_size, #boxes, #classes + 12)` where `#boxes` is the total number of boxes over all feature maps. Call it `y_encoded`.
* Find positive match. For each batch item, we find the anchor boxes that have enough overlaps with positive boxes in the ground truth data, find the corresponding indices in `y_encoded`, set the class vector and coordinates of the positive boxes. Note that elements reserved for offsets are dummy in `y_encoded`.
* Exclude non-negative boxes. For anchor boxes that are too close to a positive ground truth box but have fewer overlaps than the threshold, set the class one-hot vector to all zeros, indicating they are neither positive boxes nor negative boxes.
* All other boxes are preset as negative boxes with coordiantes set.

Then transform it to offsets relative to anchor boxes. So in each index corresponding to a certain anchor box, `y_true` has [class_vector, 4 offsets, 4 dummies, 4 variances]. `y_pred` has [class_vector, 4 offsets, 4 anchor box coords, 4 variances]

But there are much more negative boxes than positive ones in `y_true`, we shouldn't use all of them.       

```python
  # First, compute the classification loss for all negative boxes.
  neg_class_loss_all = classification_loss * negatives  # Tensor of shape (batch_size, n_boxes)
  n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32)  # The number of non-zero loss entries in `neg_class_loss_all`
  # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
  # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
  # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
  # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
  # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
  # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
  # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
  # is at most the number of negative boxes for which there is a positive classification loss.

  # Compute the number of negative examples we want to account for in the loss.
  # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller).
  n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)
```

So only chosen negative losses go into loss function, other negatives contribute zeros to loss and that leads to no update on weights corresponds to those excluded negative boxes.

#### Loss Function
We start computing loss from `y_true` and `y_pred`. They should be of the same shape such that we can compare them. `y_pred` is of the shape `(batch, n_boxes_total, classes + 12)` where `n_boxes_total` stands for the number of boxes across all the feature maps connected to a predictor layer.

#### y_true and y_pred

For ground truth each image contains the same number of boxes the model needs to predict.

Let's walk through the data processing procedures.
