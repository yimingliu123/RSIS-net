inception骨架:
初始图片大小 torch.Size([1, 3, 256, 256])
enc0 torch.Size([1, 32, 127, 127])
enc1 torch.Size([1, 64, 62, 62])
enc2 torch.Size([1, 192, 29, 29])
enc3 torch.Size([1, 1088, 14, 14])
enc4 torch.Size([1, 2080, 6, 6])

restnet50骨架:
base所提供的5个大小尺度
初始图片大小 torch.Size([1, 3, 267, 400])
eco0大小 torch.Size([1, 64, 134, 200])
eco1大小 torch.Size([1, 256, 68, 101])
eco2大小 torch.Size([1, 512, 34, 51])
eco3大小 torch.Size([1, 1024, 17, 26])
eco4大小 torch.Size([1, 2048, 17, 26])


--mode test  --model ./models/final.pth  --test_fold ./test
--arch resnet --train_root ./data/DUTS/DUTS-TR --train_list ./data/DUTS/DUTS-TR/train_pair.lst
""
{
小伙伴,开始网络结构之旅啦:~~~~~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
x.size:     torch.Size([1, 3, 273, 400])
只经历了 base  包含backboone!!!
conv2merge.size0:     torch.Size([1, 64, 137, 200])
conv2merge.size1:     torch.Size([1, 256, 69, 101])
conv2merge.size2:     torch.Size([1, 512, 35, 51])
conv2merge.size3:     torch.Size([1, 1024, 18, 26])
conv2merge.size4:     torch.Size([1, 2048, 18, 26])
###########################################################
小伙伴 infos 构成PPM(Payramid Pooling Module)哦,含有丰富语义信息!!! 
infos0:     torch.Size([1, 512, 18, 26])
infos1:     torch.Size([1, 256, 35, 51])
infos2:     torch.Size([1, 256, 69, 101])
infos3:     torch.Size([1, 128, 137, 200])
###########################################################
conv2merge 进行 convert, 经过翻转后就是restnet传给FAM的那路特征图
conv2merge.size0:     torch.Size([1, 128, 137, 200])
conv2merge.size1:     torch.Size([1, 256, 69, 101])
conv2merge.size2:     torch.Size([1, 256, 35, 51])
conv2merge.size3:     torch.Size([1, 512, 18, 26])
conv2merge.size4:     torch.Size([1, 512, 18, 26])
#######################################################################
将conv2merge 进行上下反转~~~~
conv2merge.size0:     torch.Size([1, 512, 18, 26])
conv2merge.size1:     torch.Size([1, 512, 18, 26])
conv2merge.size2:     torch.Size([1, 256, 35, 51])
conv2merge.size3:     torch.Size([1, 256, 69, 101])
conv2merge.size4:     torch.Size([1, 128, 137, 200])
第一次FAM(Feature Aggregation Module) 
merge.size:     torch.Size([1, 512, 18, 26])
第 1 次FAM(Feature Aggregation Module) ; 一共两次 ,三个层合一
merge.size:     torch.Size([1, 256, 35, 51])
第 2 次FAM(Feature Aggregation Module) ; 一共两次 ,三个层合一
merge.size:     torch.Size([1, 256, 69, 101])
第 3 次FAM(Feature Aggregation Module) ; 一共两次 ,三个层合一
merge.size:     torch.Size([1, 128, 137, 200])
第四个FAM 空跑一次
merge.size:     torch.Size([1, 128, 137, 200])
经过 score_layer
merge.size:     torch.Size([1, 1, 273, 400])
}"


config.train_root = {str} './data/DUTS/DUTS-TR'
config.wd = {float} 0.0005
config = {Namespace} Namespace(arch='resnet', batch_size=1, cuda=True, epoch=24, epoch_save=3, iter_size=10, load='', lr=5e-05, mode='train', model=None, n_color=3, num_thread=1, pretrained_model='./dataset/pretrained/resnet50_caffe.pth', sal_mode='e', save_folder='./results',
 arch = {str} 'resnet'
 batch_size = {int} 1
 cuda = {bool} True
 epoch = {int} 24
 epoch_save = {int} 3
 iter_size = {int} 10
 load = {str} ''
 lr = {float} 5e-05
 mode = {str} 'train'
 model = {NoneType} None
 n_color = {int} 3
 num_thread = {int} 1
 pretrained_model = {str} './dataset/pretrained/resnet50_caffe.pth'
 sal_mode = {str} 'e'
 save_folder = {str} './results'
 show_every = {int} 50
 test_fold = {NoneType} None
 test_list = {str} './data/ECSSD/test.lst'
 test_root = {str} './data/ECSSD/Imgs/'
 train_list = {str} './data/DUTS/DUTS-TR/train_pair.lst'
 train_root = {str} './data/DUTS/DUTS-TR'
 wd = {float} 0.0005
 
 
 #########################################
 #class ResNet_locate(nn.Module)
    def __init__(self, bottlneck, [3,4,6,3])
    self.resnet = ResNet(bottlneck, [3,4,6,3])
 
self.resnet 结构 如下所示: 64-256-512-1024-2048
{ 
 'conv1' (139626045442848) = {Conv2d} Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
 'bn1' (139626045443352) = {BatchNorm2d} BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
 'relu' (139627509348312) = {ReLU} ReLU(inplace)
 'maxpool' (139626045443800) = {MaxPool2d} MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=True)
 
 'layer1'  :
        '0' (139625951508440) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n  (downsample): Sequential(\n    (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  )\n)
        '1' (139625951509000) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
        '2' (139625951509952) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
   
  'layer2'  :
        '0' (139625952904616) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)\n  (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n  (downsample): Sequential(\n    (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)\n    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  )\n)
        '1' (139625952904672) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
        '2' (139625952904728) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
        '3' (139625952904784) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
  
  layer3
        '0' (139625951506984) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)\n  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n  (downsample): Sequential(\n    (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)\n    (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  )\n)
        '1' (139625951507600) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
        '2' (139625951507936) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
        '3' (139625951510064) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
        '4' (139625951510008) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
        '5' (139625951509392) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
  
'layer4' 
      '0' (139625952905568) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)\n  (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n  (downsample): Sequential(\n    (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)\n    (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  )\n)
      '1' (139625952905624) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)\n  (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
      '2' (139625952903328) = {Bottleneck} Bottleneck(\n  (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)\n  (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (relu): ReLU(inplace)\n)
  };
  
  
  
self.ppms_pre:  Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

self.ppms = {OrderedDict} 
 '0' (139625952044736) = {Sequential} Sequential(\n  (0): AdaptiveAvgPool2d(output_size=1)\n  (1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (2): ReLU(inplace)\n)
 '1' (139625952045408) = {Sequential} Sequential(\n  (0): AdaptiveAvgPool2d(output_size=3)\n  (1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (2): ReLU(inplace)\n)
 '2' (139625952045912) = {Sequential} Sequential(\n  (0): AdaptiveAvgPool2d(output_size=5)\n  (1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (2): ReLU(inplace)\n)
 
self.ppm_cat :Sequential((0): Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  (1): ReLU(inplace))

self.infos
 '0' (139625951807560) = {Sequential} Sequential(\n  (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (1): ReLU(inplace)\n)
 '1' (139625951807616) = {Sequential} Sequential(\n  (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (1): ReLU(inplace)\n)
 '2' (139625951807672) = {Sequential} Sequential(\n  (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (1): ReLU(inplace)\n)
 '3' (139625951807728) = {Sequential} Sequential(\n  (0): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (1): ReLU(inplace)\n)
 ############################################################################################
 
 ############################################################################################
 
 def extra_layer(base_model_cfg, resnet50_locate()):
    convert_layers = ConvertLayer(config['convert'])
    就是上面类ConvertLayer() 传进来的
        '0' (139837833267272) = {Sequential} Sequential(\n  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (1): ReLU(inplace)\n)
        '1' (139837833266656) = {Sequential} Sequential(\n  (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (1): ReLU(inplace)\n)
        '2' (139837833202912) = {Sequential} Sequential(\n  (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (1): ReLU(inplace)\n)
        '3' (139837833200336) = {Sequential} Sequential(\n  (0): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (1): ReLU(inplace)\n)
        '4' (139837833319960) = {Sequential} Sequential(\n  (0): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (1): ReLU(inplace)\n)
#############  deep_pool 是 FAM    
   deep_pool_layers :
        0=
            'pools' (140538574066272) = {ModuleList} ModuleList(\n  (0): AvgPool2d(kernel_size=2, stride=2, padding=0)\n  (1): AvgPool2d(kernel_size=4, stride=4, padding=0)\n  (2): AvgPool2d(kernel_size=8, stride=8, padding=0)\n)
            'convs' (140538576907376) = {ModuleList} ModuleList(\n  (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n)
            'relu' (140540054595880) = {ReLU} ReLU()
            'conv_sum' (140538574040176) = {Conv2d} Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            'conv_sum_c' (140538574040240) = {Conv2d} Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        1 =       
            'pools' (140538574066272) = {ModuleList} ModuleList(\n  (0): AvgPool2d(kernel_size=2, stride=2, padding=0)\n  (1): AvgPool2d(kernel_size=4, stride=4, padding=0)\n  (2): AvgPool2d(kernel_size=8, stride=8, padding=0)\n)
            'convs' (140538576907376) = {ModuleList} ModuleList(\n  (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n)
            'relu' (140540054595880) = {ReLU} ReLU()
            'conv_sum' (140538574040176) = {Conv2d} Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            'conv_sum_c' (140538574040240) = {Conv2d} Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        2 = 
            'pools' (140538574066272) = {ModuleList} ModuleList(\n  (0): AvgPool2d(kernel_size=2, stride=2, padding=0)\n  (1): AvgPool2d(kernel_size=4, stride=4, padding=0)\n  (2): AvgPool2d(kernel_size=8, stride=8, padding=0)\n)
            'convs' (140538576907376) = {ModuleList} ModuleList(\n  (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n)
            'relu' (140540054595880) = {ReLU} ReLU()
            'conv_sum' (140538574040176) = {Conv2d} Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            'conv_sum_c' (140538574040240) = {Conv2d} Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        3 =   
            'pools' (140538574066272) = {ModuleList} ModuleList(\n  (0): AvgPool2d(kernel_size=2, stride=2, padding=0)\n  (1): AvgPool2d(kernel_size=4, stride=4, padding=0)\n  (2): AvgPool2d(kernel_size=8, stride=8, padding=0)\n)
            'convs' (140538576907376) = {ModuleList} ModuleList(\n  (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n)
            'relu' (140540054595880) = {ReLU} ReLU()
            'conv_sum' (140538574040176) = {Conv2d} Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            'conv_sum_c' (140538574040240) = {Conv2d} Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
        4 = 
            'pools' (140538574066272) = {ModuleList} ModuleList(\n  (0): AvgPool2d(kernel_size=2, stride=2, padding=0)\n  (1): AvgPool2d(kernel_size=4, stride=4, padding=0)\n  (2): AvgPool2d(kernel_size=8, stride=8, padding=0)\n)
            'convs' (140538576907376) = {ModuleList} ModuleList(\n  (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n  (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n)
            'relu' (140540054595880) = {ReLU} ReLU()
            'conv_sum' (140538574040176) = {Conv2d} Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

   ScoreLayer((score): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1)))
   
#################################################################################################
class poolnet:  ()    
        self.base = base       [resnet50_locate()]在class resnet50_locate中
        #剩下的三个都在  def extra_layer    
        self.deep_pool = nn.ModuleList(deep_pool_layers)   
        self.score = score_layers
        self.convert = convert_layers
