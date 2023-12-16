from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def find_white_bbox(image_path,new_path):
    # 打开图片
    img = Image.open(image_path)
    img_array = np.array(img)
    #print(img_array)
    non_zero_coords = list(zip(*np.nonzero(img_array)))
    #print(img_array[non_zero_coords])
    img_gray = img.convert('L')
    
    threshold = 120
    img_binary = img_gray.point(lambda p: p > threshold)
    # print(img_binary)
    img_array = np.array(img_binary)
    # print(img_array)
    non_zero_coords = list(zip(*np.nonzero(img_array)))
    #print(non_zero_coords)
    if not non_zero_coords:
        #print(image_path,"没有白色像素点")
        return

    min_x = min([coord[0] for coord in non_zero_coords])
    min_y = min([coord[1] for coord in non_zero_coords])
    max_x = max([coord[0] for coord in non_zero_coords])
    max_y = max([coord[1] for coord in non_zero_coords])
    
    min_x = max(0 , min_x - 3)
    min_y = max(0 , min_y - 3)
    max_x = min(127 , max_x + 3)
    max_y = min(127 , max_y + 3)
    #print(max_x)
    #print(max_y)
    bbox = (min_y, min_x, max_y, max_x)
    cropped_img = img.crop(bbox)
    
    # 保存切割后的图片
    cropped_img.save(new_path)

def process_images(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的子文件夹
    for subfolder in tqdm(os.listdir(input_folder), desc="Processing folders"):
        subfolder_path = os.path.join(input_folder, subfolder)

        # 确保是文件夹
        if os.path.isdir(subfolder_path):
            # 创建子文件夹在输出文件夹中
            output_subfolder_path = os.path.join(output_folder, subfolder)
            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path)

            files_to_process = [filename for filename in os.listdir(subfolder_path) if filename.lower().endswith(".png")]

            for filename in tqdm(files_to_process, desc=f"Processing {subfolder}", leave=False):
                file_path = os.path.join(subfolder_path, filename)

                new_filename = f"{filename.split('.')[0]}.png"
                new_file_path = os.path.join(output_subfolder_path, new_filename)

                find_white_bbox(file_path, new_file_path)

if __name__ == "__main__":
    input_folder = "data"  
    output_folder = "cutData"  
    process_images(input_folder, output_folder)
