from pathlib import Path

import numpy as np

from src.checkbox_fixings.main import ImageProcessor

if __name__ == '__main__':
    m = np.array([[0,0,1,1,1,0,0,1,1,0,0,1,1,1,1],[0,0,1,1,1,0,0,1,1,0,0,1,1,1,1]])
    dm = np.diff(m)
    sx,sy = np.where(dm==1)
    sy+=1

    ex,ey = np.where(dm==-1)
    ey+=1
    print((sx,sy))
    print((ex,ey))
    print(ex)
    ex_points =np.where(np.diff(ex)==1)[0]+1
    print(ex_points)
    exit()




    # points = np.concatenate((np_start_indices.reshape(-1,1),np_end_indices.reshape(-1,1)),axis=1)
    # print(points)
    # print(dm)
    # print(m)
    # print(m[2:5])
    exit()

    # processor = ImageProcessor()
    # image_path = "/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg"
    # binary_arr = processor.get_binary_image(image_path=image_path)
    # row = bin
    # ary_arr[500]
    # print(np.sum(row == 1))
    # print(np.sum(row == 0))
    # print(np.sum(row))
