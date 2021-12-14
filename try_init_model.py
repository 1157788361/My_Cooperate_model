import sys
sys.path.append('/home/lvjinkai/Cooperative_Training/')
sys.path.append('/home/Cooperative_Training/') # 在docker 里面时，以 docker 里面的目录结构为主。
# 远程断了重连 ，sys.path.append 需要再运行一次。

from medseg.models.ebm.encoder_decoder import MyEncoder, MyDecoder
import torch.nn as nn
import torch
from os.path import join
import torch.optim as optim
#
# shape_inc_ch=4
# reduce_factor=4
# encoder_dropout=None
# num_classes = 4
# decoder_dropout=None
#
# shape_encoder = MyEncoder(input_channel=shape_inc_ch, output_channel=512 // reduce_factor, feature_reduce=reduce_factor,
#                           if_SN=False, encoder_dropout=encoder_dropout, norm=nn.BatchNorm2d, act=nn.ReLU())
# shape_decoder = MyDecoder(input_channel=512 // reduce_factor, up_type='NN', output_channel=num_classes,
#                           feature_reduce=reduce_factor, if_SN=False, decoder_dropout=decoder_dropout,
#                           norm=nn.BatchNorm2d)
# segmentation_decoder = MyDecoder(input_channel=512 // reduce_factor, up_type='NN', output_channel=num_classes,
#                                              feature_reduce=reduce_factor, if_SN=False, decoder_dropout=decoder_dropout, norm=nn.BatchNorm2d)
#
# model_list =[shape_encoder,shape_decoder,segmentation_decoder]
# checkpoint_dir = '/data/zzu_student/ljk633/Cooperative_Training/saved/train_skull_dataset_standard_n_cls_4/cooperative_training/0/model/best/checkpoints'
#
# shape_decoder_path = join(checkpoint_dir, 'shape_decoder.pth')
# shape_encoder_path = join(checkpoint_dir, 'shape_encoder.pth')
# segmentation_decoder_path = join(checkpoint_dir, 'segmentation_decoder.pth')
# path_list = [ shape_encoder_path,shape_decoder_path,segmentation_decoder_path]
#
#
# # model.load_state_dict(torch.load(resume_path))
# # model.load_state_dict(torch.load(resume_path)['model_state'], strict=False)
#
# # 初始化 shape_encoder
# shape_encoder.load_state_dict(torch.load(shape_encoder_path))
#
# # 初始化 shape_encoder
# shape_decoder.load_state_dict(torch.load(shape_decoder_path))
#
# # 初始化 shape_encoder
# segmentation_decoder.load_state_dict(torch.load(segmentation_decoder_path))

class try_model(nn.Module):
    def __init__(self):
        super(try_model, self).__init__()
        self.shape_inc_ch=4
        self.reduce_factor=4
        self.encoder_dropout=None
        self.num_classes = 4
        self.decoder_dropout=None
        self.model = self.get_model()
        self.learning_rate = 0.0001
    def get_model(self):
        shape_encoder = MyEncoder(input_channel=self.shape_inc_ch, output_channel=512 // self.reduce_factor, feature_reduce=self.reduce_factor,
                                  if_SN=False, encoder_dropout=self.encoder_dropout, norm=nn.BatchNorm2d, act=nn.ReLU())
        shape_decoder = MyDecoder(input_channel=512 // self.reduce_factor, up_type='NN', output_channel=self.num_classes,
                                  feature_reduce=self.reduce_factor, if_SN=False, decoder_dropout=self.decoder_dropout,
                                  norm=nn.BatchNorm2d)
        segmentation_decoder = MyDecoder(input_channel=512 // self.reduce_factor, up_type='NN', output_channel=self.num_classes,
                                                     feature_reduce=self.reduce_factor, if_SN=False, decoder_dropout=self.decoder_dropout, norm=nn.BatchNorm2d)

        model_list =[shape_encoder,shape_decoder,segmentation_decoder]
        checkpoint_dir = '/data/zzu_student/ljk633/Cooperative_Training/saved/train_skull_dataset_standard_n_cls_4/cooperative_training/0/model/best/checkpoints'

        shape_decoder_path = join(checkpoint_dir, 'shape_decoder.pth')
        shape_encoder_path = join(checkpoint_dir, 'shape_encoder.pth')
        segmentation_decoder_path = join(checkpoint_dir, 'segmentation_decoder.pth')
        path_list = [ shape_encoder_path,shape_decoder_path,segmentation_decoder_path]


        # model.load_state_dict(torch.load(resume_path))
        # model.load_state_dict(torch.load(resume_path)['model_state'], strict=False)

        # 初始化 shape_encoder
        shape_encoder.load_state_dict(torch.load(shape_encoder_path))

        # 初始化 shape_encoder
        shape_decoder.load_state_dict(torch.load(shape_decoder_path))

        # 初始化 shape_encoder
        segmentation_decoder.load_state_dict(torch.load(segmentation_decoder_path))
        model = {
                 'segmentation_decoder': segmentation_decoder,
                 'shape_encoder': shape_encoder,
                 'shape_decoder': shape_decoder,
                 }
        return model
    def print_model_items(self):
        pass

A = try_model()

# A.train()
# 默认为model.train()
print('A.training:',A.training)  # True

A.eval()
print('A.training:',A.training)  # False


optimizers_dict = {}
for model_name, model in A.model.items():
    print('model_name:',model_name)
    print('model :',model)
    #print('model_parameters:',model.parameters())
    print('set optimizer for:', model_name)
    optimizer = optim.Adam(model.parameters(), lr=A.learning_rate)
    optimizer.zero_grad()
    optimizers_dict[model_name] = optimizer
    # self.optimizers = optimizers_dict

b = A.model
for model in b.values():  # .values 针对于 字典。
    for p in model.parameters():
        if p.grad is not None:
            # 当我们再训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；
            # 或者只 训练部分分支网络，并不让其梯度对主网络的梯度造成影响，
            # 这时候我们就需要使用detach()函数来切断一些分支的反向传播
            p.grad.detach_()
            p.grad.zero_()
for k, v in optimizers_dict.items():
        v.zero_grad() # 梯度置零 见下面函数。
"""model :
MyDecoder(
  (up1): res_up_family(
    (up): Sequential(
      (0): UpsamplingNearest2d(scale_factor=2.0, mode=nearest)
    )
    (conv): Sequential(
      (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.2)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (conv_input): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
    (last_act): LeakyReLU(negative_slope=0.2)
  )
  (up2): res_up_family(
    (up): Sequential(
      (0): UpsamplingNearest2d(scale_factor=2.0, mode=nearest)
    )
    (conv): Sequential(
      (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.2)
      (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (conv_input): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
    (last_act): LeakyReLU(negative_slope=0.2)
  )
  (up3): res_up_family(
    (up): Sequential(
      (0): UpsamplingNearest2d(scale_factor=2.0, mode=nearest)
    )
    (conv): Sequential(
      (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.2)
      (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (conv_input): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
    (last_act): LeakyReLU(negative_slope=0.2)
  )
  (up4): res_up_family(
    (up): Sequential(
      (0): UpsamplingNearest2d(scale_factor=2.0, mode=nearest)
    )
    (conv): Sequential(
      (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.2)
      (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (conv_input): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    (last_act): LeakyReLU(negative_slope=0.2)
  )
  (final_conv): Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1))
)"""