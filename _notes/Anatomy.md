# Anatomy of a Caffe Model

Network is defined layer by layer from bottom to top, from data to loss. The bottom most layer is the data layer or the input layer which takes in data as a blob. A **blob** is an Unified Memory Interface for storage and communication among the layers of the network. Just like *Mat* in OpenCV. 

The 3 components that make up the anatomy of a Caffe Model are:
1. Blob
2. Layer
3. Net

## Blob

A blob holds images, parameters and derivatives or gradients. Dimensions of a 4D blob are 
1. Number *N*
2. Channel *K*
3. Height *H*
4. Width *W*

To access an element of the blob, say (n,k,h,w), we use the index :
> ( (n x K + k) x H + h) x W + w

N is the batch size. The network is trained by using one batch of images (examples) at a time. This batch size is typically 256. K is the feature dimension. Consider a data blob of RGB images. K represents each channel of the image. In another case of convolutional layer, K represents the depth index of the feature map (or activation map). 

> Note : The use of lower dimensional (say 2D) blobs for non-image oriented application is possible.

Blobs for a convolutional layer with 96 feature maps, each of spatial size 11x11, has a shape of 96 x 3 x 11 x 11

> For training the model with custom data, hacking the data layer is necessary. 

Blobs are of two types. 1. data and 2. diff (gradient). 

```c++
const Dtype* cpu_data const;
Dtype* mutable_cpu_data;
```

## Layer

A layer is the fundamental unit of computation in a Caffe Network. There are 3 functions for the Layer. 

1. SetUp : initialize layer and connections
2. Forward : Data flow from bottom to top
3. Backward : Gradient (gradient of output w.r.t input) flow from top to bottom; also involves internal storage of gradients w.r.t weights (parameters)

> In order to define a custom layer, one needs to define 1,2 and 3 functions manually.

## Net

Net is the holistic unit of a Caffe Model. It performs composition of functions, layer by layer, backward propagation of gradient calculated by auto-differentiation. The network is generally represented by a DAG of layers. DAG of logistic regression classifier is given below

![DAG of layers](/img/logreg.jpg)

### GPU/CPU switch

```c++
Caffe::mode( )
Caffe::set_mode( )

### Net::Init( )

*Net::Init( )* takes care of scaffolding DAG, initializing blobs and layers. Each layer's setUp( ) function is called.


## Model Format

Models are defined in Plaintext Protocol Buffer Schema (protobuf), while trained models are serialized in Binary Protocol Buffer Schema (binaryproto) as .caffemodel files.

> Read [caffe.proto](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)