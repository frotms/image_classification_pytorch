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
| mnasnet                 | 224       |
| mobilenetV3             | 224       |
| oct-resnet              | 224/256   |
| ...                     | ...       |

### pre-trained model
you can download pretrain model with url in ($net-module.py)

#### From [torchvision](https://github.com/pytorch/vision/) package:

- ResNet ([resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth), [resnet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth), [resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth), [resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth), [resnet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth))
- DenseNet ([densenet121](https://download.pytorch.org/models/densenet121-a639ec97.pth'), [densenet169](https://download.pytorch.org/models/densenet169-b2777c0a.pth), [densenet201](https://download.pytorch.org/models/densenet201-c1103571.pth), [densenet161](https://download.pytorch.org/models/densenet161-8d451a50.pth))
- Inception v3 ([inception_v3](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth))
- VGG ([vgg11](https://download.pytorch.org/models/vgg11-bbd30ac9.pth), [vgg11_bn](https://download.pytorch.org/models/vgg11_bn-6002323d.pth), [vgg13](https://download.pytorch.org/models/vgg13-c768596a.pth), [vgg13_bn](https://download.pytorch.org/models/vgg13_bn-abd245e5.pth), [vgg16](https://download.pytorch.org/models/vgg16-397923af.pth), [vgg16_bn](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth), [vgg19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth), [vgg19_bn](https://download.pytorch.org/models/vgg19_bn-c79401a0.pth))
- SqueezeNet ([squeezenet1_0](https://download.pytorch.org/models/squeezenet1_0-a815701f.pth), [squeezenet1_1](https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth))
- AlexNet ([alexnet](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth))

#### From [Pretrained models for PyTorch](https://github.com/Cadene/pretrained-models.pytorch) package:
- ResNeXt ([resnext101_32x4d](http://data.lip6.fr/cadene/pretrainedmodels/resnext101_32x4d-29e315fa.pth), [resnext101_64x4d](http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth))
- NASNet-A Large (`nasnet_a_large`: [imagenet](http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth), [imagenet+background](http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth))
- NASNet-A Mobile (`nasnet_a_mobile`: [imagenet](http://data.lip6.fr/cadene/pretrainedmodels/nasnetamobile-7e03cead.pth))
- Inception-ResNet v2 (`inception_resnet_v2`: [imagenet](http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth), [imagenet+background](http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth))
- Dual Path Networks ([dpn68](http://data.lip6.fr/cadene/pretrainedmodels/dpn68-66bebafa7.pth), [dpn68b](http://data.lip6.fr/cadene/pretrainedmodels/dpn68b_extra-84854c156.pth), `dpn92`: [imagenet](http://data.lip6.fr/cadene/pretrainedmodels/dpn68-66bebafa7.pth), [imagenet+5k](http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth), [dpn98](http://data.lip6.fr/cadene/pretrainedmodels/dpn98-5b90dec4d.pth), [dpn131](http://data.lip6.fr/cadene/pretrainedmodels/dpn131-71dfe43e0.pth), [dpn107](http://data.lip6.fr/cadene/pretrainedmodels/dpn107_extra-1ac7121e2.pth))
- Inception v4 (`inception_v4`: [imagenet](http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth), [imagenet+background](http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth))
- Xception ([xception](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth))
- Squeeze-and-Excitation Networks ([senet154](http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth), [se_resnet50](http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth), [se_resnet101](http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth), [se_resnet152](http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth), [se_resnext50_32x4d](http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth), [se_resnext101_32x4d](http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth))
- PNASNet-5-Large (`pnasnet_5_large`: [imagenet](http://data.lip6.fr/cadene/pretrainedmodels/pnasnet5large-bf079911.pth), [imagenet+background](http://data.lip6.fr/cadene/pretrainedmodels/pnasnet5large-bf079911.pth))
- PolyNet ([polynet](http://data.lip6.fr/cadene/pretrainedmodels/polynet-f71d82a5.pth))

#### From [mobilenetV2](https://github.com/ericsun99/MobileNet-V2-Pytorch) package:
- Mobilenet V2 ([mobilenet_v2](https://github.com/ericsun99/MobileNet-V2-Pytorch))

#### From [shufflenetV2](https://github.com/ericsun99/Shufflenet-v2-Pytorch) package:
- Shufflenet V2 ([shufflenet_v2](https://github.com/ericsun99/Shufflenet-v2-Pytorch))

#### From [MnasNet](https://github.com/billhhh/MnasNet-pytorch-pretrained) package:
- Mnasnet ([MnasNet](https://github.com/billhhh/MnasNet-pytorch-pretrained))  

#### From [mobilenetV3](https://github.com/kuan-wang/pytorch-mobilenet-v3) package:
- Mobilenet V3 ([mobilenet_v3_large](https://github.com/kuan-wang/pytorch-mobilenet-v3), [mobilenet_v3_small](https://github.com/kuan-wang/pytorch-mobilenet-v3))  

#### From [OctaveResnet](https://github.com/d-li14/octconv.pytorch) package:
- Octave Resnet ([oct_resnet26](https://github.com/d-li14/octconv.pytorch), [oct_resnet50](https://github.com/d-li14/octconv.pytorch), [oct_resnet101](https://github.com/d-li14/octconv.pytorch), [oct_resnet152](https://github.com/d-li14/octconv.pytorch), [oct_resnet200](https://github.com/d-li14/octconv.pytorch))  

## usage

### configuration
| configure                       | description                                                               |
|---------------------------------|---------------------------------------------------------------------------|
| model_module_name               | eg: vgg_module                                                            |
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
9.[https://github.com/billhhh/MnasNet-pytorch-pretrained](https://github.com/billhhh/MnasNet-pytorch-pretrained)  
10.[https://github.com/d-li14/octconv.pytorch](https://github.com/d-li14/octconv.pytorch)  
11.[https://github.com/kuan-wang/pytorch-mobilenet-v3](https://github.com/kuan-wang/pytorch-mobilenet-v3)  
