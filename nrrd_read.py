import nrrd
from PIL import Image
import numpy as np

skull_path = 'argmax_2_1516.npy'
skull_data = np.load(skull_path)
path_label = "002_seg.nrrd"

path_image = "002_img.nrrd"

# path_label = "ED_img.nrrd"
# path_image = "ED_label.nrrd"

data_im,options_im = nrrd.read(path_image)
# data：保存图片的多维矩阵;
# nrrd_options：保存图片的相关信息
data_label,options_label = nrrd.read(path_label )




for i in range(data_label.shape[-1]):
    image_1 = data_im[:,:,i]
    label_1 = data_label[:,:,i]

    image_1 = np.expand_dims(image_1, -1)
    image_1 = np.repeat(image_1, 3, -1)*255
    image_1 = image_1.astype(int)

    label_1= np.expand_dims(label_1, -1)
    label_1= np.repeat(label_1, 3, -1)*255.
    label_1 = label_1.astype(int)

    res = np.concatenate([image_1 ,label_1], axis=1)
    print(res.dtype)

    img=Image.fromarray(np.uint8(res))
    img.show()
    # 将图片存储到指定文件夹
    # cv2.imwrite(os.path.join(despath, image_id), res)








# print('data.shape:',data_im.shape) # [w, h, n] 第三维度保存的是图像的序号；