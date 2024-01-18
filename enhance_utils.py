import cv2
import numpy as np
import os
import random
import time


# 模拟生成模糊图像
def generate_blur(input_image, kernel_min=3, kernel_max=7):
    if kernel_min < kernel_max:
        kernel_min = kernel_max

    # 指定高斯滤波器的内核大小，以控制模糊程度
    r = random.choice([kernel_min, kernel_max])
    kernel_size = (r, r)

    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(input_image, kernel_size, 0)

    return blurred_image


# # 模拟生成运动模糊图像
# def generate_blur_motion(input_image, kernel_min=3, kernel_max=5):
#     if kernel_min < kernel_max:
#         kernel_min = kernel_max
#     # 创建运动模糊内核
#     r = random.randint(kernel_min, kernel_max)
#     if r % 2 == 0:
#         r += 1
#     kernel_size = r  # 内核大小，控制模糊程度

#     angle = random.randint(1, 24) * 15   # 运动方向的角度
#     motion_blur_kernel = np.zeros((kernel_size, kernel_size))

#     # 计算内核的中心点
#     center = (kernel_size - 1) / 2

#     # 计算运动方向的起点和终点
#     start = int(center - kernel_size / 2)
#     end = int(center + kernel_size / 2)

#     # 在内核中设置运动方向
#     for i in range(start, end + 1):
#         x = int(center + np.cos(np.deg2rad(angle)) * (i - center))
#         y = int(center + np.sin(np.deg2rad(angle)) * (i - center))
#         motion_blur_kernel[y, x] = 1

#     # 归一化内核，确保内核的总和为1
#     motion_blur_kernel = motion_blur_kernel / kernel_size

#     # 应用运动模糊
#     blurred_image = cv2.filter2D(input_image, -1, motion_blur_kernel)

#     return blurred_image
def generate_blur_motion(input_image, kernel_min, kernel_max):
    if kernel_min < kernel_max:
        kernel_min = kernel_max

    # 随机选择内核大小
    kernel_size = np.random.choice(np.arange(kernel_min, kernel_max + 1, 2))
    
    # 随机选择运动方向的角度
    angle = np.random.randint(1, 25) * 15
    
    # 创建运动模糊内核
    motion_blur_kernel = np.zeros((kernel_size, kernel_size))
    center = (kernel_size - 1) / 2
    indices = np.arange(kernel_size)
    
    # 计算运动方向的坐标
    x = np.round(center + np.cos(np.deg2rad(angle)) * (indices - center)).astype(int)
    y = np.round(center + np.sin(np.deg2rad(angle)) * (indices - center)).astype(int)
    
    # 设置运动方向
    motion_blur_kernel[y, x] = 1
    
    # 归一化内核
    motion_blur_kernel /= kernel_size

    # 应用运动模糊
    blurred_image = cv2.filter2D(input_image, -1, motion_blur_kernel)

    return blurred_image

# 模拟叠加随机噪点

def gaussian_noise(image, sigma=20):
    row, col, ch = image.shape
    mean = 0
    #sigma = 15
    # if not degree:
    #     var = np.random.uniform(0.004, 0.01)
    # else:
    #     var = degree
    # sigma = var ** 0.5 * 25
    #gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = np.random.normal(mean, sigma, (row, col, 1))

    #gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    noisy = np.clip(noisy,a_min=0,a_max=255)
    #noisy = np.array(noisy, dtype=np.uint8)
    
    return noisy



def generate_noisy(img, noise_level=0.01, max_noise_size=2):
    noisy_img = img.copy()

    # Calculate the total number of noise pixels
    total_noise_pixels = int(noise_level * img.size / img.shape[2])

    # Counter for number of noise pixels added
    noise_pixels_added = 0

    while noise_pixels_added < total_noise_pixels:
        # Randomly pick a pixel
        x = np.random.randint(0, img.shape[1])
        y = np.random.randint(0, img.shape[0])

        # Randomly choose the size of the noise (1x1, 2x2, 3x3)
        size = np.random.randint(1, max_noise_size + 1)

        # Ensure the noise block stays within image bounds
        if y + size > img.shape[0]:
            size = img.shape[0] - y
        if x + size > img.shape[1]:
            size = img.shape[1] - x

        # Set the chosen area to black
        noisy_img[y:y + size, x:x + size] = (np.random.randint(0, 80),
                                             np.random.randint(0, 80),
                                             np.random.randint(0, 80))

        # Update the counter
        noise_pixels_added += size * size

    return noisy_img


# 模拟手机拍照叠加光照效果
def generate_photo_light(input_image, direction=-1, darkness=0.7):
    #direction: -1 random, 0 up, 1 down, 2 left, 3 right
    #ratio = 4 / 60
    #print(input_image.shape)
    if direction == -1:
        direction = np.random.randint(0, 4)

    #print(direction)
    #if input_image.shape == 3:
    #    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    #flattened_image = input_image.flatten()

    # 计算排序后的百分位数
    # low_value = np.percentile(flattened_image, percentile_low)
    # high_value = np.percentile(flattened_image, percentile_high)

    low_value = np.min(input_image) + 20
    high_value = np.max(input_image) - 20

    ratio = min((high_value - low_value) * darkness, 80)
    #ratio = (high_value - low_value) 

    #ratio = min((high_value - low_value) * 0.8, low_value *1.2)
    ratio = int(ratio) 
    #print("ratio",low_value/255,(high_value - low_value) /255)
    #ratio = ratio * 0.35
    print("ratio",ratio)

    # Create a mask of the same size as the image, filled with horizontal gradient
    mask = np.zeros_like(input_image, dtype=np.uint8)
    
    if direction in [2,3]:
        max_intensity = ratio #* input_image.shape[1]

        # Calculate gradient
        for x in range(input_image.shape[1]):
            intensity = int(x / input_image.shape[1] * max_intensity)
            if direction == 2:
                mask[:, x] = max_intensity - intensity
            else:
                mask[:, x] = intensity

        # Apply the light effect mask to the original image
        light_effect_image = cv2.subtract(input_image, mask,dtype=cv2.CV_8U)
    
    elif direction in [0,1]:
        max_intensity = ratio #* input_image.shape[0]

        # Calculate gradient
        for x in range(input_image.shape[0]):
            intensity = int(x / input_image.shape[0] * max_intensity)
            if direction == 0:
                mask[x, :] = max_intensity - intensity
            else:
                mask[x, :] = intensity

        # Apply the light effect mask to the original image
        light_effect_image = cv2.subtract(input_image, mask,dtype=cv2.CV_8U)

    return light_effect_image


# def generate_photo_shadow(input_image,percentile_low=1,percentile_high=99):
#         # 将图像转换为一维数组

#     if len(input_image.shape) == 3:
#         input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

#     #flattened_image = input_image.flatten()

#     # 计算排序后的百分位数
#     # low_value = np.percentile(flattened_image, percentile_low)
#     # high_value = np.percentile(flattened_image, percentile_high)
#     low_value = np.min(input_image) +10
#     high_value= np.max(input_image) -10
#     #print(low_value,high_value)

#     # 生成多边形的凸包
#     mask = np.zeros_like(input_image, dtype=np.uint8)

#     num_points = np.random.randint(3, 20)
#     points = np.random.randint(0, min(input_image.shape[0], input_image.shape[1]), size=(num_points, 2))
#     hull = cv2.convexHull(points)
#     # 在初始掩码上填充凸包的区域
#     cv2.fillPoly(mask, [hull], 255)
    
#     # 用灰度值填充凸包区域（在原本像素低于阴影像素时进行填充）
#     depth = np.random.randint((5*low_value + 5*high_value)/10,(1 * low_value + 9 * high_value)/10)  # 随机生成灰度值
#     #print(depth)
#     #depth = (4*low_value + 6*high_value)/10

#     mask_shadow = np.ones_like(input_image, dtype=np.uint8)
#     mask_shadow = mask_shadow  *  np.random.uniform(0.3, 0.7) * (high_value - low_value)
#     # 剪切原图中的阴影区域
#     shadow_cut_image =  np.where(mask > 0, input_image, 0)
#     mask_shadow = np.where(mask > 0, mask_shadow, 0)
#     #shadow_cut_image =  cv2.subtract(shadow_cut_image, mask_shadow)
#     shadow_cut_image = cv2.subtract(shadow_cut_image.astype(np.float32), mask_shadow.astype(np.float32), dtype=cv2.CV_8U)

#     #np.where(shadow_cut_image > depth, depth , shadow_cut_image)

#     shadow_effect_image = np.where(mask > 0, shadow_cut_image, input_image)
#     return shadow_effect_image


def generate_photo_shadow(input_image, darkness=0.7):
    # 将图像转换为一维数组
    #if input_image.shape[-1] == 3:
    #    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # 计算排序后的百分位数
    low_value = np.min(input_image) + 20
    high_value = np.max(input_image) - 20

    # 生成多边形的凸包
    mask = np.zeros_like(input_image, dtype=np.uint8)

    num_points = np.random.randint(3, 20)
    points = np.random.randint(0, min(input_image.shape[0], input_image.shape[1]), size=(num_points, 2))
    hull = cv2.convexHull(points)
    # 在初始掩码上填充凸包的区域
    cv2.fillPoly(mask, [hull], 255)

    # 用灰度值填充凸包区域（在原本像素低于阴影像素时进行填充）
    depth = np.random.randint((5 * low_value + 5 * high_value) / 10, (1 * low_value + 9 * high_value) / 10)

    mask_shadow = darkness * (high_value - low_value)
    mask_shadow = (mask_shadow * np.ones_like(input_image)).astype(np.uint8)

    # 剪切原图中的阴影区域
    shadow_cut_image = np.where(mask > 0, input_image, 0)
    mask_indices = mask > 0
    shadow_cut_image = cv2.subtract(shadow_cut_image.astype(np.float32), mask_shadow.astype(np.float32), dtype=cv2.CV_8U)

    # np.where(shadow_cut_image > depth, depth, shadow_cut_image)
    shadow_cut_image = np.clip(shadow_cut_image, None, depth)

    shadow_effect_image = np.where(mask_indices, shadow_cut_image, input_image)
    return shadow_effect_image



def check_image_type(image):
    # 确保图像是灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 获取图像的宽度和高度
    h, w = gray.shape
    gray = gray[int(0.2*h):int(0.8*h), int(0.2*w):int(0.8*w)]
    h, w = gray.shape

    # 随机选择100个像素点
    random_pixels = [gray[np.random.randint(h), np.random.randint(w)] for _ in range(100)]

    # 找到这100个点中的最大值
    max_value = max(random_pixels)

    # 判断图像类型
    if max_value > 235:
        return True
    else:
        return False


if __name__ == '__main__':
    #file_path_name = './tools/annotation/test3.jpg'
    file_path_name = '/root/autodl-tmp/Code/image_tools-master/test3.jpg'


    file_path = file_path_name[:file_path_name.rfind('/')]
    file_name = file_path_name[file_path_name.rfind('/') + 1:]
    file_name_no_ext = file_name[:file_name.rfind('.')]
    print(file_path_name)
    im = cv2.imread(file_path_name)


    hello = check_image_type(im)
    time_start = time.time()
    im1 = generate_blur(im, 3, 7)
    cv2.imwrite(os.path.join(file_path, file_name_no_ext + '_1.jpg'), im1)
    
    print(time.time()-time_start)
    time_start = time.time()

    im2 = generate_blur_motion(im, 3, 9)
    cv2.imwrite(os.path.join(file_path, file_name_no_ext + '_2.jpg'), im2)

    print("time",time.time()-time_start)
    time_start = time.time()

    im3 = generate_noisy(im)
    cv2.imwrite(os.path.join(file_path, file_name_no_ext + '_3.jpg'), im3)

    print("time",time.time()-time_start)

    time_start = time.time()

    im4 = generate_photo_light(im,direction=1,darkness=0.3)
    cv2.imwrite(os.path.join(file_path, file_name_no_ext + '_4.jpg'), im4)

    print("time",time.time()-time_start)
    time_start = time.time()

    im5 = generate_photo_light(im,direction=3,darkness=0.3)
    cv2.imwrite(os.path.join(file_path, file_name_no_ext + '_5.jpg'), im5)

    print("time",time.time()-time_start)
    time_start = time.time()

    im6 = generate_photo_shadow(im)
    cv2.imwrite(os.path.join(file_path, file_name_no_ext + '_6.jpg'), im6)

    print("time",time.time()-time_start)
    time_start = time.time()
    
    im7 =gaussian_noise(im)
    cv2.imwrite(os.path.join(file_path, file_name_no_ext + '_7.jpg'), im7)

    print("time",time.time()-time_start)
