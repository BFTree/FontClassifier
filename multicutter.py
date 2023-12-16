from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
min_thresh = 5 #字符上最少的像素点
min_range = 5 #字符最小的宽度
 
def vertical(img_arr):
    h,w = img_arr.shape
    ver_list = []
    for x in range(w):
        ver_list.append(h - np.count_nonzero(img_arr[:, x]))
    return ver_list
 
def horizon(img_arr):
    h,w = img_arr.shape
    hor_list = []
    for x in range(h):
        hor_list.append(w - np.count_nonzero(img_arr[x, :]))
    return hor_list
 
def OTSU_enhance(img_gray, th_begin=0, th_end=256, th_step=1):  
    max_g = 0  
    suitable_th = 0  
    for threshold in range(th_begin, th_end, th_step):  
        bin_img = img_gray > threshold  
        bin_img_inv = img_gray <= threshold  
        fore_pix = np.sum(bin_img)  
        back_pix = np.sum(bin_img_inv)  
        if 0 == fore_pix:  
            break  
        if 0 == back_pix:  
            continue  
 
        w0 = float(fore_pix) / img_gray.size  
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix  
        w1 = float(back_pix) / img_gray.size  
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix  
        # intra-class variance  
        g = w0 * w1 * (u0 - u1) * (u0 - u1)  
        if g > max_g:  
            max_g = g  
            suitable_th = threshold  
    return suitable_th 
 
def cut_line(horz, pic):
    begin, end = 0, 0
    w, h = pic.size
    cuts=[]
    for i,count in enumerate(horz):
       if count >= min_thresh and begin == 0:
            begin = i
       elif count >= min_thresh and begin != 0:
            continue
       elif count <= min_thresh and begin != 0:
            end = i
            #print (begin, end), count
            if end - begin >= 2:
                cuts.append((end - begin, begin, end))
                begin = 0
                end = 0
                continue
       elif count <= min_thresh or begin == 0:
            continue
    cuts = sorted(cuts, reverse=True)
    print("Line",cuts)
    img_arr_list = []
    if len(cuts) == 0:
        return 0, False
    else:
        for i in range(len(cuts)):
            crop_ax = (0, cuts[i][1], w, cuts[i][2])
            img_arr = np.array(pic.crop(crop_ax))
            img_arr_list.append(img_arr)

    #print(img_arr_list)
    return img_arr_list, True
 
def simple_cut(vert):
    begin, end = -1,-1
    cuts = []
    for i,count in enumerate(vert):
        # print(count,begin,end)
        if count >= min_thresh:
           if begin == -1:
               begin = i
        if count < min_thresh:
            if begin == -1:
                continue
            else:
                end = i
                if end-begin >= min_range:
                    cuts.append((begin,end))
                    begin = -1
                    end = -1
    print("Column",cuts)
    return cuts
 
def get_max_gap(cut):
    max_gap_width = 0
    max_gap_end = 0

    for i in range(len(cut) - 1):
        gap_width = cut[i + 1][0] - cut[i][1]
        if gap_width > max_gap_width:
            max_gap_width = gap_width
            max_gap_end = i

    return max_gap_width, max_gap_end


 
def isnormal_width(w_judge, w_normal_min, w_normal_max):
    if w_judge < w_normal_min:
        return -1
    elif w_judge > w_normal_max:
        return 1
    else:
        return 0

 
def cut_by_kmeans(pic_path, save_path, save_size):
    src_pic = Image.open(pic_path).convert('L') #先把图片转化为灰度
    src_arr = np.array(src_pic)
    threshold = OTSU_enhance(src_arr) * 0.9 #用大津阈值
    bin_arr = np.where(src_arr < threshold, 0, 255) #二值化图片
    
    horz = horizon(bin_arr) #获取到该行的 水平 方向的投影
    line_arrs, flag = cut_line(horz, src_pic) #把文字（行）所在的位置切割下来
    if flag == False:
        return flag
    for xi in range(len(line_arrs)):
        line_arr = line_arrs[xi]
        line_arr = np.where(line_arr < threshold, 0, 255)
        line_img = Image.fromarray((255 - line_arr).astype("uint8"))
        width, height = line_img.size
        
        vert = vertical(line_arr) #获取到该行的 垂直 方向的投影
        cut = simple_cut(vert) #先进行简单的文字分割（即有空隙就切割）
        
        #cv.line(img,(x1,y1), (x2,y2), (0,0,255),2)    
        width_data = []
        width_data_TooBig = []
        width_data_withoutTooBig = []
        for i in range(len(cut)):
            tmp = (cut[i][1] - cut[i][0], 0) 
            if tmp[0] > height * 1.8:     #比这一行的高度大两倍的肯定是连在一起的
                temp = (tmp[0], i)
                width_data_TooBig.append(temp)
            else:
                width_data_withoutTooBig.append(tmp)
            width_data.append(tmp)
        kmeans = KMeans(n_clusters=2).fit(width_data_withoutTooBig)
    
        label_tmp = kmeans.labels_
        label = []
        j = 0
        k = 0
        for i in range(len(width_data)):       
            if j != len(width_data_TooBig) and k != len(label_tmp):
                if i == width_data_TooBig[j][1]:
                    label.append(2)
                    j = j + 1
                else:
                    label.append(label_tmp[k])
                    k = k + 1
            elif j == len(width_data_TooBig) and k != len(label_tmp):
                label.append(label_tmp[k])
                k = k + 1
            elif j != len(width_data_TooBig) and k == len(label_tmp):
                label.append(2)
                j = j + 1
                
        label0_example = 0
        label1_example = 0
        for i in range(len(width_data)):
            if label[i] == 0:
                label0_example = width_data[i][0]
            elif label[i] == 1:
                label1_example = width_data[i][0]
        if label0_example > label1_example:  
            label_width_normal = 0
        else:
            label_width_normal = 1
        label_width_small = 1 - label_width_normal
            
        cluster_center = []
        cluster_center.append(kmeans.cluster_centers_[0][0])
        cluster_center.append(kmeans.cluster_centers_[1][0])
        for i in range(len(width_data)):
            if label[i] == label_width_normal and width_data[i][0] > cluster_center[label_width_normal] * 4 / 3: 
                label[i] = 2
                temp = (width_data[i][0], i)
                width_data_TooBig.append(temp)        
        max_gap = get_max_gap(cut) 
        for i in range(len(label)):
            if i == max_gap[1]:
                label[i] = 3
        
        width_normal_data = []     
        width_data_TooSmall = []
        for i in range(len(width_data)):
            if label[i] == label_width_normal:
                width_normal_data.append(width_data[i][0])
            elif label[i] != label_width_normal and label[i] != 2 and label[i] != 3: 
                tmp = (width_data[i][0], i)
                width_data_TooSmall.append(tmp)
        width_normal_max = max(width_normal_data)
        width_normal_min = min(width_normal_data)    
        
        if len(width_data_TooBig) != 0:   
            for i in range(len(width_data_TooBig)):
                index = width_data_TooBig[i][1]
                mid = (cut[index][0] + cut[index][1]) / 2
                tmp1 = (cut[index][0], int(mid))
                tmp2 = (int(mid)+1, cut[index][1])
                del cut[index]
                cut.insert(index, tmp2)
                cut.insert(index, tmp1)
                del width_data[index]
                tmp1 = (tmp1[1] - tmp1[0], index)
                tmp2 = (tmp2[1] - tmp2[0], index+1)
                width_data.insert(index, tmp2)
                width_data.insert(index, tmp1)
                label[index] = label_width_normal
                label.insert(index, label_width_normal)
                
                
        if len(width_data_TooSmall) != 0:               #除':'以外有小字符,先找'('、')'label = 4                             
            for i in range(len(width_data_TooSmall)):
                index = width_data_TooSmall[i][1]
                border_left = cut[index][0] + 1
                border_right = cut[index][1]
                RoI_data = line_arr[:,border_left:border_right]
                
                horz = horizon(RoI_data)
            
                up_down = np.sum(np.abs(RoI_data - RoI_data[::-1]))
                left_right = np.sum(np.abs(RoI_data - RoI_data[:,::-1]))
                vert = vertical(RoI_data)
            
                if up_down <= left_right * 0.6 and np.array(vert).var() < len(vert) * 2:    
                    label[index] = 4
        
            index_delete = [] 
            cut_final = []
            width_untilnow = 0
            for i in range(len(width_data)):
                if label[i] == label_width_small and width_untilnow == 0:
                    index_delete.append(i)
                    cut_left = cut[i][0]
                    width_untilnow = cut[i][1] - cut[i][0]
                elif label[i] != 3 and label[i] != 4 and width_untilnow != 0:
                    width_untilnow = cut[i][1] - cut_left
                    if isnormal_width(width_untilnow, width_normal_min, width_normal_max) == -1: 
                        index_delete.append(i)
                    elif isnormal_width(width_untilnow, width_normal_min, width_normal_max) == 0:
                        width_untilnow = 0
                        cut_right = cut[i][1]
                        tmp = (cut_left, cut_right)
                        cut_final.append(tmp)
                    elif isnormal_width(width_untilnow, width_normal_min, width_normal_max) == 1:   
                        width_untilnow = 0
                        cut_right = cut[i-1][1]
                        tmp = (cut_left, cut_right)
                        cut_final.append(tmp)
                        index_delete.append(i)
                        cut_left = cut[i][0]
                        width_untilnow = cut[i][1] - cut[i][0]
                        if i == len(width_data):
                            tmp = (cut[i][0], cut[i][1])
                            cut_final.append(tmp)
                else:
                    tmp = (cut[i][0], cut[i][1])
                    cut_final.append(tmp)
            i1 = len(cut_final) - 1
            i2 = len(cut) - 1
            if cut_final[i1][1] != cut[i2][1]:
                tmp = (cut[i2][0], cut[i2][1])
                cut_final.append(tmp)
                    
        else:
            cut_final = cut
                                
        for x in range(len(cut_final)):
            ax = (cut_final[x][0] - 1, 0, cut_final[x][1] + 1, height)
            temp = line_img.crop(ax)
            temp = temp.resize(save_size,Image.ANTIALIAS)
            temp.save('{}/{}_{}.png'.format(save_path, xi,x))
    return flag


pic_path = 'ori2.png'
save_path = 'split'
 
cut_by_kmeans(pic_path,save_path,(128,128))