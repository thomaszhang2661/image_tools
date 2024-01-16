import cv2
import os
import sys
import shutil
from enhance_utils import generate_blur, generate_blur_motion, generate_noisy, generate_photo_light, generate_photo_shadow, check_image_type
from itertools import combinations
import random
from concurrent.futures import ProcessPoolExecutor
import psutil
import time

def set_affinity(process_id, cpu_core):
    try:
        p = psutil.Process(process_id)
        p.cpu_affinity([cpu_core])
    except Exception as e:
        print(f"Error setting affinity for process {process_id}: {e}")

#input_image_path = '/root/autodl-tmp/training_data_gray/'

# enhance_mark 可以是 B M N L R 5个字母的任意组合
# 总共可以生成的组合数：C(1/5) + C(2/5) + C(3/5) + C(4/5) + C(5/5) = 5 + 10 + 10 + 5 = 30
# B： blur
# M： blur motion
# N： noisy
# L： photo light, 从左到右，越来越白
# R： photo light, 从右到左，越来越白
enhance_mark = ['B', 'M', 'N', 'L', 'S']
# 预先生成所有可能的组合
all_combinations = []
for r in range(1, len(enhance_mark) + 1):
    all_combinations.extend(combinations(enhance_mark, r))

# 屏蔽B和M同时出现的组合
all_combinations = [combo for combo in all_combinations if not ('B' in combo and 'M' in combo)]

# 屏蔽L和S同时出现的组合
all_combinations = [combo for combo in all_combinations if not ('L' in combo and 'S' in combo)]

# if len(sys.argv) >= 2:
#     input_image_path = sys.argv[1]

# if not os.path.exists(input_image_path):
#     print(f'{input_image_path} does not exist')
#     exit(-1)

#label_path = os.path.join(input_image_path, 'labels')

# if len(sys.argv) >= 3:
#     enhance_mark = sys.argv[2]

# for c in enhance_mark:
#     if c not in ['B', 'M', 'N', 'L', 'S']:
#         print(f'{enhance_mark} must be either B,M,N,L,S or their combo')
#         exit(-1)

def process_image(file):
    time_start = time.time()
    input_image_path = '/root/autodl-tmp/training_data_gray/'
    label_path = os.path.join(input_image_path, 'labels')
    # 随机选择图像增强类型
    output_image_path = './images_enhanced/'
    print(file)
    random.seed(os.getpid() + int(time.time()))

    enhance_mark = "".join(random.choice(all_combinations))
    print(enhance_mark)



    file_name_no_ext = file[:file.rfind('.')]
    file_name_ext = file[file.rfind('.'):]

    label_path_output = os.path.join(output_image_path, 'labels')

    # 拷贝label
    file_path = os.path.join(label_path, file_name_no_ext + ".json")
    #print(file_path)
    file_path2 = os.path.join(label_path_output, file_name_no_ext + '_' + enhance_mark + '.json')
    try:
        shutil.copy(file_path, file_path2)
    except:
        return

    if file_name_ext == '.jpg' or file_name_ext == '.png':
        file_path_name = os.path.join(input_image_path, file)
        output_image_filename = os.path.join(output_image_path, file_name_no_ext + "_" + enhance_mark + file_name_ext)

        im = cv2.imread(file_path_name)
        is_scan_image = check_image_type(im)
        if 'B' in enhance_mark:
            if is_scan_image:
                im = generate_blur(im, 3, 7)
            else:
                im = generate_blur(im, 3, 5)
        if 'M' in enhance_mark:
            if is_scan_image:
                im = generate_blur_motion(im, 3, 7)
            else:
                im = generate_blur_motion(im, 3, 5)
        if 'N' in enhance_mark:
            im = generate_noisy(im, noise_level=0.01, max_noise_size=3)
        if 'L' in enhance_mark:
            im = generate_photo_light(im)
        if 'S' in enhance_mark:
            im = generate_photo_shadow(im)
        print(output_image_filename)

        cv2.imwrite(output_image_filename, im)
        print("time:",time.time()-time_start)

if __name__ == '__main__':
    output_image_path = './images_enhanced/'
    input_image_path = '/root/autodl-tmp/training_data_gray/'

    items = os.listdir(input_image_path)
    # 过滤出子目录
    image_files = [entry for entry in items if os.path.isfile(os.path.join(input_image_path, entry))]

    print(len(image_files))
    #output_files =[os.path.join(output_image_path, entry) for entry in items if os.path.isfile(os.path.join(input_image_path, entry))]
    
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
    
    label_path_output = os.path.join(output_image_path, 'labels')
    if not os.path.exists(label_path_output):
        os.makedirs(label_path_output)


    with ProcessPoolExecutor(max_workers=12) as executor:
        # Get the current process ID
        current_process_id = os.getpid()
        
        # Set affinity for the current process (optional)
        set_affinity(current_process_id, 0)  # You can choose a specific CPU core
        
        # Use executor.submit to set affinity for each submitted task
        futures = [executor.submit(set_affinity, os.getpid(), i % psutil.cpu_count()) for i in range(12)]

        # Process the images
        for file in image_files:
            executor.submit(process_image, file).result()
