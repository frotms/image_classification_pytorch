# image-classification-pytorch
This repo is designed for those who want to start their projects of image classification.
It provides fast experiment setup and attempts to maximize the number of projects killed within the given time.
It includes a few Convolutional Neural Network modules.You can build your own dnn easily.

## Requirements
Python3 support only. Tested on CUDA9.0, cudnn7.

* albumentations==0.1.1
* easydict==1.8
* imgaug==0.2.6
* opencv-python==3.4.3.18
* protobuf==3.6.1
* scikit-image==0.14.0
* tensorboardX==1.4
* torch==0.4.1
* torchvision==0.2.1

## model
| net                     | inputsize |
|-------------------------|-----------|
| vggnet                  | 224       |
| alexnet                 | 224       |
| resnet                  | 224       |
| inceptionV3             | 299       |
| inceptionV4             | 299       |
| squeezenet              | 224       |
| densenet                | 224       |
| dpnnet                  | 224       |
| inception-resnet-v2     | 299       |
| mobilenetV2             | 224       |
| nasnet-a-large          | 331       |
| nasnet-mobile           | 224       |
| polynet                 | 331       |
| resnext                 | 224       |
| senet                   | 224       |
| squeezenet              | 224       |
| pnasnet                 | 331       |
| shufflenetV2            | 224       |
| ...                     | ...       |

### pre-trained model
you can download pretrain model with url in ($net-module.py)

#### From [torchvision](https://github.com/pytorch/vision/) package:

- ResNet (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`)
- DenseNet (`densenet121`, `densenet169`, `densenet201`, `densenet161`)
- Inception v3 (`inception_v3`)
- VGG (`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`)
- SqueezeNet (`squeezenet1_0`, `squeezenet1_1`)
- AlexNet (`alexnet`)

#### From [Pretrained models for PyTorch](https://github.com/Cadene/pretrained-models.pytorch) package:
- ResNeXt (`resnext101_32x4d`, `resnext101_64x4d`)
- NASNet-A Large (`nasnet_a_large`)
- NASNet-A Mobile (`nasnet_a_mobile`)
- Inception-ResNet v2 (`inception_resnet_v2`)
- Dual Path Networks (`dpn68`, `dpn68b`, `dpn92`, `dpn98`, `dpn131`, `dpn107`)
- Inception v4 (`inception_v4`)
- Xception (`xception`)
- Squeeze-and-Excitation Networks (`senet154`, `se_resnet50`, `se_resnet101`, `se_resnet152`, `se_resnext50_32x4d`, `se_resnext101_32x4d`)
- PNASNet-5-Large (`pnasnet_5_large`)
- PolyNet (`polynet`)

#### From [mobilenetV2](https://github.com/ericsun99/MobileNet-V2-Pytorch) package:
- Mobilenet V2 (`mobilenet_v2`)

#### From [shufflenetV2](https://github.com/ericsun99/Shufflenet-v2-Pytorch) package:
- Shufflenet V2 (`shufflenet_v2`)

## usage

### configuration
| configure                       | description                                                               |
|---------------------------------|---------------------------------------------------------------------------|
| model_module_name               | eg: vgg_module                                                             |
| model_net_name                  | net function name in module, eg:vgg16                                     |
| gpu_id                          | eg: single GPU: "0", multi-GPUs:"0,1,3,4,7"                                                           |
| async_loading                   | make an asynchronous copy to the GPU                                      |
| is_tensorboard                  | if use tensorboard for visualization                                      |
| evaluate_before_train           | evaluate accuracy before training                                         |
| shuffle                         | shuffle your training data                                                |
| data_aug                        | augment your training data                                                |
| img_height                      | input height                                                              |
| img_width                       | input width                                                               |
| num_channels                    | input channel                                                             |
| num_classes                     | output number of classes                                                  |
| batch_size                      | train batch size                                                          |
| dataloader_workers              | number of workers when loading data                                       |
| learning_rate                   | learning rate                                                             |
| learning_rate_decay             | learning rate decat rate                                                  |
| learning_rate_decay_epoch       | learning rate decay per n-epoch                                           |
| train_mode                      | eg:  "fromscratch","finetune","update"                                    |
| file_label_separator            | separator between data-name and label. eg:"----"                          |
| pretrained_path                 | pretrain model path                                                       |
| pretrained_file                 | pretrain model name. eg:"alexnet-owt-4df8aa71.pth"                        |
| pretrained_model_num_classes    | output number of classes when pretrain model trained. eg:1000 in imagenet |
| save_path                       | model path when saving                                                    |
| save_name                       | model name when saving                                                    |
| train_data_root_dir             | training data root dir                                                    |
| val_data_root_dir               | testing data root dir                                                     |
| train_data_file                 | a txt filename which has training data and label list                     |
| val_data_file                   | a txt filename which has testing data and label list                      |

### Training
1.make your training &. testing data and label list with txt file:

txt file with single label index eg:

	apple.jpg----0
	k.jpg----3
	30.jpg----0
	data/2.jpg----1
	abc.jpg----1
2.configuration

3.train

	python3 train.py

### Inference
eg: trained by inception_resnet_v2, vgg/data/flowers/102:

	python3 inference.py --image test.jpg --module inception_resnet_v2_module --net inception_resnet_v2 --model model.pth --size 299 --cls 102

### tensorboardX

	tensorboard --logdir='./logs/' runs

logdir is log dir in your project dir 

## References
1.[https://github.com/pytorch](https://github.com/pytorch)  
2.[https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)  
3.[https://pytorch.org](https://pytorch.org)  
5.[https://github.com/yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)  
4.[https://www.tensorflow.org](https://www.tensorflow.org)  
5.[https://github.com/Cadene/pretrained-models.pytorch/tree/master/pretrainedmodels/models](https://github.com/Cadene/pretrained-models.pytorch/tree/master/pretrainedmodels/models)  
6.[https://github.com/ericsun99/MobileNet-V2-Pytorch](https://github.com/ericsun99/MobileNet-V2-Pytorch)  
7.[http://www.robots.ox.ac.uk/~vgg/data/flowers/102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102)  
8.[https://github.com/ericsun99/Shufflenet-v2-Pytorch](https://github.com/ericsun99/Shufflenet-v2-Pytorch)  
