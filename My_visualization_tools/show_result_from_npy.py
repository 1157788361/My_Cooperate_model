import numpy as np
from glob import glob
import os
import copy
import torch
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as mpatches
class npy_skull_show():
    def __init__(self,path,classes=2):
        self.data_path = path + '/' + 'pred_npy'
        self.save_path = path
        self.name_list = glob(os.path.join(self.data_path, '*.npy'), recursive=True)
        self.name_id = list(set(x.split('/')[-1].split('_')[0] for x in self.name_list))
        self.classes = classes
        self.color_map = np.random.randint(0, 255, (self.classes, 3)) # (classes, 3) 表示生成的 数据形状 为  classes X 3 # 随机颜色
        self.IOU = {}

        "sorted IOU 只写了 label数量 为 2 个"
        # IOU 计算，交集比上并集。
        # 第一次，统计self.IOU{id:region}
        for i in self.name_id:
            self.show_npy(i)
        sorted_IOU = sorted(self.IOU.items(), key=lambda x: x[1])

        for j in sorted_IOU:
            self.show_npy(j[0],sorted_flag=True)



    def show_npy(self,id,sorted_flag=False):
        self.id = id

        self.gt_path = os.path.join(self.data_path, '{}_gt.npy'.format(id))
        self.image_path = os.path.join(self.data_path, '{}_image.npy'.format(id))
        self.soft_pred_path = os.path.join(self.data_path, '{}_soft_pred.npy'.format(id))
        self.pred_path = os.path.join(self.data_path, '{}_pred.npy'.format(id))

        #TODO : 弄清楚 输出的 class  是否 含有 背景。 含有背景  （0,1,2,3）
        gt = np.load(self.gt_path) # 1X 512 X 512
        image=np.load(self.image_path) # 1X 512 X 512
        soft_pred = np.load(self.soft_pred_path)# 1 X 2 X 512 X 512
        pred = np.load(self.pred_path) # 1X512X512

        soft_pred_2D = np.squeeze(soft_pred)

        soft_pred_block_2D = [soft_pred_2D[x,:,:] for x in range(soft_pred_2D.shape[0])]

        gt_2D=np.squeeze(gt)
        image_2D = np.squeeze(image)
        pred_npy_2D = np.squeeze(pred)


        self.draw_image(image=image_2D,gt=gt_2D,pred=pred_npy_2D,soft_pred=soft_pred_block_2D,sorted_flag=sorted_flag)
        self.draw_feature(soft_pred_block_2D)
        print('finish')

    def draw_feature(self, feature):
        save_path_feature_show = os.path.join(self.save_path, 'show_feature')
        if not os.path.exists(save_path_feature_show):
            os.makedirs(save_path_feature_show)

        plt.figure(figsize=(5 * self.classes, 5))
        for i in range(self.classes):
            # soft_predict channel数 应与 classes 数 相同
            plt.subplot(1, self.classes, i+1)
            plt.imshow(feature[i], cmap='jet')
        plt.savefig(save_path_feature_show + '/' + 'predict_feature_{}.png'.format(self.id), bbox_inches='tight')
    def draw_image(self,image,gt,pred,soft_pred,sorted_flag=False):
        current_IOU=0
        save_path_show = os.path.join(self.save_path, 'show_image')
        if not os.path.exists(save_path_show):
            os.makedirs(save_path_show)

        save_path_feature_show = os.path.join(self.save_path, 'show_feature')
        if not os.path.exists(save_path_feature_show ):
            os.makedirs(save_path_feature_show )


        image = np.expand_dims(image, -1)
        image = np.repeat(image, 3, -1)
        image*=255

        shape = image.shape

        gt_only = np.zeros(shape)
        image_pred = np.array(image) # 类似于 copy 一次。
        image_soft_pred = np.zeros(shape)
        soft_pred_sum = soft_pred[0] + soft_pred[1]
        soft_pred_sum= (soft_pred_sum+abs(soft_pred_sum.min())) * 255/(soft_pred_sum.max() - soft_pred_sum.min())
        soft_pred_sum = np.expand_dims(soft_pred_sum, -1)
        soft_pred_sum = np.repeat(soft_pred_sum, 3, -1)



        for i in range(self.classes-1): # 除了背景，其他都上色
            for j in range(3):
                # 为原图上色，将label范围内的像素转为指定颜色
                image_pred[ :, :, j][pred == i+1] = self.color_map[i][j]
                gt_only[ :, :, j][gt == i] = self.color_map[i][j]
        res = np.concatenate([image, image_pred, gt_only,soft_pred_sum], axis=1)
        # save_path+'/'+'contrast_predict_{}.png'.format(self.id)
        # save_path_show = os.path.join(self.save_path, 'show_image') 上移
        # if not os.path.exists(save_path_show):
        #     os.makedirs(save_path_show)
        cv2.imwrite(os.path.join(save_path_show, 'show_predict_{}.png'.format(self.id)), res)

        "可视化 带有 混淆矩阵的 label "
        plt.figure(figsize=(20,5))
        gt_only_1 = np.array(image)
        image_pred_wrong = np.array(image)
        image_predict_true = np.array(image)
        " 图像对比部分，查看 未预测到的，与 预测错的 " # 未预测到，实际就是 预测错误，预测为label_0。
        # 利用 混淆矩阵 即   pred*len(self.classes)+gt   如 self.classes =3  label = 0(背景),1,2
        #   gt_0 gt_1 gt_2
        #    0    1    2  # 预测为 label 0 (预测错误为label_0:1,2，原 label 为 1,2 ，预测为 0 即为错误。)
        #    3    4    5  # 预测为 label 1 (预测错误为label_1:3,5，原 label 为 0,2 ，预测为 1 即为错误。)
        #    6    7    8  # 预测为 lbael 2 (预测错误为label_1:6,7，原 label 为 0,1 ，预测为 2 即为错误。)
        # 其IOU计算为 如 label_2 = 8 / (2+5( gt为 2 ) +8 +6+7 ( 预测为 2 ))

        res_contrast = pred*self.classes+gt

        current_color = ['#FFFF00', '#FFE4B5', '#FFB6C1', '#FF69B4', '#FF00FF',
                 '#FF0000', '#ADFF2F', '#98FB98', '#87CEFA']

        color_contrast_map = self.Hex_to_RGB(current_color)
        # 'label_1->label_1' '#FF00FF' 品红。
        if self.classes == 3 :
            current_label = ['bg->bg','label_1->bg','label_2->bg',
                             'bg->label_1','label_1->label_1','label_2->label_1',
                             'bg->label_2', 'label_1->label_2', 'label_2->label_2']
            current_num_sum = [i for i in range((self.classes-1)*self.classes+self.classes-1)]
            current_label_num = [i for i in range(self.classes)] # 0,1,2

        elif self.classes ==2 :
            current_label = ['bg->bg', 'label_1->bg',
                             'bg->label_1', 'label_1->label_1']
            current_num_sum = [i for i in range((self.classes - 1) * self.classes + self.classes - 1)]
            current_label_num = [i for i in range(self.classes)] # 0,1
        if self.classes == 3:
            for i in current_label_num: # 除了背景，其他都上色
                # c_list = current_label_num  浅拷贝，会改变原值
                c_list = copy.deepcopy(current_label_num)
                c_list = list(set(c_list).difference(set([i])))

                for j in range(3):
                    # 为原图上色，将label范围内的像素转为指定颜色
                    image_predict_true[ :, :, j][res_contrast == i+i*self.classes] = color_contrast_map[i+i*self.classes][j] # 预测正确的情况。 # 表示在
                    # gt 为 i , 但 预测错误的情况 (0,3,6) (1,4,7) (2,5,8)
                    image_pred_wrong[:, :, j][res_contrast == i + c_list[0] * self.classes] = color_contrast_map[i + c_list[0] * self.classes][j]
                    image_pred_wrong[:, :, j][res_contrast == i + c_list[1] * self.classes] = color_contrast_map[i + c_list[1] * self.classes][j]
                    gt_only_1[:, :, j][gt == i] = color_contrast_map[i+i*self.classes][j]

                    # IOU计算：

        if self.classes == 2:
            for i in current_label_num: # 除了背景，其他都上色
                c_list = copy.deepcopy(current_label_num)
                c_list = list(set(c_list).difference(set([i])))
                for j in range(3):
                    # 为原图上色，将label范围内的像素转为指定颜色
                    image_predict_true[ :, :, j][res_contrast == i+i*self.classes] = color_contrast_map[i+i*self.classes][j] # 预测正确的情况。
                    image_pred_wrong[:, :, j][res_contrast == i + c_list[0] * self.classes] = color_contrast_map[i + c_list[0] * self.classes][j]

                    # IOU计算：
            current_IOU = (res_contrast==3).sum()/( (res_contrast==1).sum()+(res_contrast==2).sum()+(res_contrast==3).sum() +1e-3)
            self.IOU[self.id] = current_IOU

        res_ = np.concatenate([image,gt_only_1,image_predict_true,image_pred_wrong], axis=1)
        plt.imshow(res_.astype(np.uint8))

        # 分别为： 原图像，带有预测的图像，只有gt，soft 预测的图像。
        save_path = os.path.join(self.save_path ,'contrast_result')
        save_path_sorted = os.path.join(self.save_path, 'sorted_contrast_result')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_sorted):
            os.makedirs(save_path_sorted)

        if self.classes==3 :
            current_num_sum_copy=copy.deepcopy(current_num_sum)
            b = [0,4,8]
            current_num_sum_copy = list(set(current_num_sum_copy).difference(set(b)))
            # list(set(a).difference(set(b))) # a - b
            patches = [mpatches.Patch(color=current_color[i], label="{:s}".format(current_label[i])) for i in current_num_sum_copy]

        if self.classes==2 :
            current_num_sum_copy = copy.deepcopy(current_num_sum)
            b_1 = [0,3]
            current_num_sum_copy = list(set(current_num_sum_copy).difference(set(b_1)))
            patches = [mpatches.Patch(color=current_color[i], label="{:s}".format(current_label[i])) for i in current_num_sum_copy]

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
        # 下面一行中bbox_to_anchor指定了legend的位置
        ax.legend(handles=patches, bbox_to_anchor=(0.95, 1.12), ncol=len(current_num_sum))  # 生成legend
        if sorted_flag ==False:
            plt.savefig(save_path+'/'+'contrast_predict_{}.png'.format(self.id), bbox_inches='tight')
        if sorted_flag ==True:
            plt.savefig(save_path_sorted + '/' + 'contrast_predict_{}.png'.format(self.id), bbox_inches='tight')

    def Hex_to_RGB(self,hex_list):
        color_map = []
        for hex in hex_list:
            r = int(hex[1:3], 16)
            g = int(hex[3:5], 16)
            b = int(hex[5:7], 16)
            #     rgb = str(r)+','+str(g)+','+str(b)
            rgb = [r, g, b]
            #     print(rgb)
            color_map.append(rgb)
        return color_map




if __name__ == '__main__':
    a = npy_skull_show('/home/My_Cooperative_model/illness_test_result',classes=3)






