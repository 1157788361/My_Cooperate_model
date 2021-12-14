import SimpleITK as sitk
import numpy as np
new_spacing=[1.36719, 1.36719, -1]

label_array = np.random.randint(0,4,size=(9,256,216))
# label_array[0,:,:]=0
my_label_array_1 = np.zeros_like(label_array)
my_label_array_2 = np.zeros_like(label_array)

my_label_array_1 = np.uint8(my_label_array_1)
my_label_array_2 = np.uint8(my_label_array_2)




my_sitkImage_from_my_label = sitk.GetImageFromArray(label_array)
my_sitkImage_from_pid_label = sitk.ReadImage('patient001_frame01_gt.nii.gz')

my_sitkImage_from_pid_label_array = sitk.GetArrayFromImage(my_sitkImage_from_pid_label)

my_sitkImage_from_pid_label_array[0,1,1]=2
my_sitkImage_from_pid_label_contain_tiny_number = sitk.GetImageFromArray(my_sitkImage_from_pid_label_array)


my_sitkImage_from_pid_label_array = my_sitkImage_from_pid_label_array[1:]

my_label_array_1[my_sitkImage_from_pid_label_array==1]=1
my_label_array_1[my_sitkImage_from_pid_label_array==2]=2
my_label_array_1[my_sitkImage_from_pid_label_array==3]=3

my_copy_sitkImage_from_pid_label_array_int32 = np.int32(my_label_array_1)
my_copy_sitkImage_from_pid_label_array = np.uint8(my_label_array_1)

my_copy_sitkImage_from_pid_label = sitk.GetImageFromArray(my_copy_sitkImage_from_pid_label_array)
my_copy_sitkImage_from_pid_label_int32 = sitk.GetImageFromArray(my_copy_sitkImage_from_pid_label_array_int32)

my_sitkImage_from_my_label.SetSpacing(spacing=(1.5625, 1.5625, 10.0))
my_sitkImage_from_pid_label.SetSpacing(spacing=(1.5625, 1.5625, 10.0))
my_copy_sitkImage_from_pid_label.SetSpacing(spacing=(1.5625, 1.5625, 10.0))
my_copy_sitkImage_from_pid_label_int32.SetSpacing(spacing=(1.5625, 1.5625, 10.0))
my_sitkImage_from_pid_label_contain_tiny_number.SetSpacing(spacing=(1.5625, 1.5625, 10.0))

transform = sitk.AffineTransform(3)
transform.SetCenter((0.0,0.0,0.0))


label_resample_my_sitkImage_from_my_label = sitk.Resample(my_sitkImage_from_my_label, [247,293,10], transform, 1, (0.0,0.0,0.0), [1.36719, 1.36719, -1], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
label_resample_my_sitkImage_from_pid_label = sitk.Resample(my_sitkImage_from_pid_label, [247,293,10], transform, 1, (0.0,0.0,0.0), [1.36719, 1.36719, -1], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
label_resample_my_copy_sitkImage_from_pid_label = sitk.Resample(my_copy_sitkImage_from_pid_label, [247,293,10], transform, 1, (0.0,0.0,0.0), [1.36719, 1.36719, -1], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
label_resample_my_copy_sitkImage_from_pid_label_int32 = sitk.Resample(my_copy_sitkImage_from_pid_label_int32, [247,293,10], transform, 1, (0.0,0.0,0.0), [1.36719, 1.36719, -1], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
label_resample_my_sitkImage_from_pid_label_contain_tiny_number = sitk.Resample(my_sitkImage_from_pid_label_contain_tiny_number, [247,293,10], transform, 1, (0.0,0.0,0.0), [1.36719, 1.36719, -1], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

# sitk.Resample(im, new_size, transform, interpolator, im.GetOrigin(), new_spacing, im.GetDirection())



label_resample_my_sitkImage_from_my_label_Array = sitk.GetArrayFromImage(label_resample_my_sitkImage_from_my_label)
label_resample_my_sitkImage_from_pid_label_Array = sitk.GetArrayFromImage(label_resample_my_sitkImage_from_pid_label)
label_resample_my_copy_sitkImage_from_pid_label_Array = sitk.GetArrayFromImage(label_resample_my_copy_sitkImage_from_pid_label)
label_resample_my_copy_sitkImage_from_pid_label_Array_int32 = sitk.GetArrayFromImage(label_resample_my_copy_sitkImage_from_pid_label_int32)
label_resample_my_sitkImage_from_pid_label_contain_tiny_number_array = sitk.GetArrayFromImage(label_resample_my_sitkImage_from_pid_label_contain_tiny_number)

resample_sum_label_1 = np.sum(label_resample_my_sitkImage_from_my_label_Array)
resample_sum_label_2 = np.sum(label_resample_my_sitkImage_from_pid_label_Array)
resample_sum_label_3 = np.sum(label_resample_my_copy_sitkImage_from_pid_label_Array)
resample_sum_label_4 = np.sum(label_resample_my_copy_sitkImage_from_pid_label_Array_int32)
resample_sum_label_5 = np.sum(label_resample_my_sitkImage_from_pid_label_contain_tiny_number_array)

print('sssssss')