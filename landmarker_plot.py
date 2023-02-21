import os
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt

def overlay_land_on_mouth(file_index, boundary, file_path='./test/video_proc_0', save_path=None):

    land = np.load(file_path + '/landmarkers/' + file_index + '.npy')
    plt.figure()
    im = plt.imread(file_path + '/images/' + file_index + '.png')
    plt.imshow(im[int(min(land[49:69, 1]))-boundary:int(max(land[49:69, 1]))+boundary, 
                        int(min(land[49:69, 0]))-boundary:int(max(land[49:69, 0]))+boundary])
    plt.scatter([int(x) for x in land[49:69, 0]-int(min(land[49:69, 0]))+boundary], 
                [int(x) for x in land[49:69, 1]-int(min(land[49:69, 1]))+boundary], 
                c='r', s=30)
    plt.axis('off')
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + '/' + file_index + '.png', bbox_inches = 'tight', pad_inches = 0)

    plt.close()

if __name__ == '__main__':
    for idx in tqdm.tqdm(glob.glob('./test/video_proc_0/landmarkers/*.npy')):
        overlay_land_on_mouth(file_index=idx[-8:-4], boundary=30, file_path='./test/video_proc_0', save_path='./test/video_proc_0/mouth_with_land')