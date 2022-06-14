#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tkinter as tk
from tkinter import filedialog, dialog
from PIL import ImageTk, Image, ImageDraw
import os
import cv2


# In[2]:


filename = './demo_file'    # 相對路徑
set_img_size = 230
pca_rate = 100


# In[3]:


def zoom_img(img, size) :
    w = img.width
    h = img.height
    if w > h :
        img_zoom = img.resize( ( size, int((h/w)*size) ) )
    else :
        img_zoom = img.resize( ( int((w/h)*size), size ) )
    return img_zoom


# In[4]:


def nearest_compression() :
    img_ary = np.array(img)
    image_result = cv2.resize( img_ary, (img_ary.shape[1],img_ary.shape[0]), interpolation = cv2.INTER_NEAREST)
    return image_result

def linear_compression() :
    img_ary = np.array(img)
    image_result = cv2.resize( img_ary, (img_ary.shape[1],img_ary.shape[0]), interpolation = cv2.INTER_LINEAR)
    return image_result


# In[5]:


def pca_conti(image_pca, rate) :
    h, w = image_pca.shape[:2]
    
    # calculate the mean
    mean = np.mean(image_pca, axis = 1) #均質化
    mean = mean[:,np.newaxis]
    mean = np.tile(mean, w)
    
    normal = image_pca.astype(np.float64) - mean
    eig_val, eig_vec = np.linalg.eigh( np.cov(normal) ) #計算其特徵值、特徵向量
    p = np.size(eig_vec, axis = 1)
    index = np.argsort( eig_val ) #排列特徵值 小->大
    index = index[::-1] #取特徵向量
    
    feature = eig_vec[:,index]
    
    if rate < p or rate > 0 :
        feature = feature[:, range(rate)]
    
    score = np.dot(feature.T, normal)
    recon = np.dot(feature, score) + mean #重建數據
    result = np.uint8(np. absolute(recon))
    return result

def pca_compression(rate) :
    img_ary = np.array(img)
    r = pca_conti( img_ary[:,:,0], rate)
    b = pca_conti( img_ary[:,:,1], rate)
    g = pca_conti( img_ary[:,:,2], rate)
    image_result = cv2.merge( [r, b, g] )

    return image_result


# In[6]:


def get_jpg_button():
    global img_tk, get_path, img, img_compress

    get_path = filedialog.askopenfilename(title = '請選擇要壓縮的.jpg檔', initialdir = filename, filetypes = [('JPG','jpg')])
    img = Image.open(get_path)
    
    img_zoom = zoom_img(img, set_img_size)
    img_tk = ImageTk.PhotoImage(img_zoom)
    img_label = tk.Label(img_frame, width = 40, height = 230, image = img_tk)
    img_label.grid(row = 1, column = 0, padx = 0, pady = 0, sticky = 'nwes')
    
    sign_label = tk.Label(save_frame, width = 38, text = '已選擇檔案，請在右側選擇檔案壓縮方式...', font = ('Arial', 12), fg = 'gray')
    sign_label.grid(row = 0, column = 0, padx = 0, pady = 5, columnspan = 2 , sticky = 'n')

def save_jpg_button():
    global img_compress
    
    save_path = filedialog.asksaveasfilename(title = '請選擇儲存.jpg檔的位置', initialdir = filename, filetypes = [('JPG','jpg')])
    img_compress.save(str(save_path) + '.jpg')
    
    sign_label = tk.Label(save_frame, width = 38, text = '已儲存為.jpg檔, 請至資料夾查看', font = ('Arial', 12), fg = 'gray')
    sign_label.grid(row = 0, column = 0, padx = 5, pady = 5, columnspan = 2 , sticky = 'n')

def save_pdf_button():
    global img_compress
    
    file_path = filedialog.asksaveasfilename(title = '請選擇儲存.pdf檔的位置',
                                            initialdir = filename,
                                            filetypes = [('PDF','pdf')])
    img_compress.save(str(file_path) + '.pdf')
    
    sign_save_label = tk.Label(save_frame, width = 38, text = '已儲存為.pdf檔, 請至資料夾查看', font = ('Arial', 12), fg = 'gray')
    sign_save_label.grid(row = 0, column = 0, padx = 5, pady = 5, columnspan = 2 , sticky = 'n')


# In[7]:


def nearest_func() :
    global img_compress, img_compress_tk
    
    sign_label = tk.Label(save_frame, width = 38, text = '最近鄰差值法壓縮壓縮中，請稍後...', font = ('Arial', 12), fg = 'gray')
    sign_label.grid(row = 0, column = 0, padx = 5, pady = 5, columnspan = 2 , sticky = 'n')
    
    img_compress = Image.fromarray( nearest_compression() )

    sign_label = tk.Label(save_frame, width = 38, text = '最近鄰差值法壓縮完成，請選擇儲存檔案格式...', font = ('Arial', 12), fg = 'gray')
    sign_label.grid(row = 0, column = 0, padx = 0, pady = 5, columnspan = 2 , sticky = 'n')
    
def linear_func() :
    global img_compress, img_compress_tk
    
    sign_label = tk.Label(save_frame, width = 38, text = '雙線性差值法壓縮壓縮中，請稍後...', font = ('Arial', 12), fg = 'gray')
    sign_label.grid(row = 0, column = 0, padx = 5, pady = 5, columnspan = 2 , sticky = 'n')
    
    img_compress = Image.fromarray( linear_compression() )

    sign_label = tk.Label(save_frame, width = 38, text = '雙線性差值法壓縮完成，請選擇儲存檔案格式...', font = ('Arial', 12), fg = 'gray')
    sign_label.grid(row = 0, column = 0, padx = 0, pady = 5, columnspan = 2 , sticky = 'n')

def pca_func() : # pca 按鈕
    global img_compress, img_compress_tk
    
    sign_label = tk.Label(save_frame, width = 38, text = 'PCA壓縮中，請稍後...', font = ('Arial', 12), fg = 'gray')
    sign_label.grid(row = 0, column = 0, padx = 5, pady = 5, columnspan = 2 , sticky = 'n')
    
    img_compress = Image.fromarray( pca_compression(pca_rate) )

    sign_label = tk.Label(save_frame, width = 38, text = 'PCA壓縮完成，請選擇儲存檔案格式...', font = ('Arial', 12), fg = 'gray')
    sign_label.grid(row = 0, column = 0, padx = 0, pady = 5, columnspan = 2 , sticky = 'n')


# In[8]:


win = tk.Tk()
win.title('圖片壓縮與轉檔')
win.geometry('580x530')
win.resizable(False,False)

# get frame
get_frame = tk.Frame(win, width = 100, height = 20)
get_frame.grid(row = 0, column = 0, padx = 0, pady = 0, columnspan = 2, sticky = 'nw')
# the label notice user to choose the .jpg @get_frame
sign_choose_label = tk.Label(get_frame, width = 38, text = '請選擇要壓縮的.jpg檔案', font = ('Arial', 12), fg = 'gray')
sign_choose_label.grid(row = 0, column = 0, padx = 0, pady = 10, columnspan = 2 , sticky = 'nwe')
# the button press to command get_jpg_button for open a file to choose .jpg @get_frame
get_button = tk.Button(get_frame, width = 60, height = 1, text = '選擇.jpg檔', font = ('Arial', 12), command = get_jpg_button)
get_button.grid(row = 1, column = 0, padx = 15, pady = 0, sticky = 'n')

# img frame
img_frame = tk.Frame(win, width = 40, height = 380)
img_frame.grid(row = 1, column = 0, padx = 0, pady = 0, sticky = 'nw')

img_label = tk.Label(img_frame, width = 25, height = 2, text = '您選擇的圖片', font = ('Arial', 12), bg = 'steelblue', fg = 'lightgray')
img_label.grid(row = 0, column = 0, padx = 15, pady = 30, sticky = 'nw')

# compress choice frame
compress_frame = tk.Frame(win, width = 60, height = 320)
compress_frame.grid(row = 1, column = 1, padx = 0, pady = 0, sticky = 'nw')
blank_label = tk.Label(compress_frame, width = 38, text = '     ', font = ('Arial', 12), fg = 'gray')
blank_label.grid(row = 1, column = 0, padx = 0, pady = 40, columnspan = 2 , sticky = 'nw')
# the button press to command nearest_button @compress_frame
nearest_button = tk.Button(compress_frame, width = 20, text = '最近鄰差值法壓縮', font = ('Arial', 12), command = nearest_func)
nearest_button.grid(row = 2, column = 0, padx = 60, pady = 35, sticky = 'nw')
# the button press to command linear_button @compress_frame
linear_button = tk.Button(compress_frame, width = 20, text = '雙線性差法值壓縮', font = ('Arial', 12), command = linear_func)
linear_button.grid(row = 3, column = 0, padx = 60, pady = 0, sticky = 'nw')
# the button press to command pca_button @compress_frame
pca_button = tk.Button(compress_frame, width = 20, text = 'PCA 壓縮', font = ('Arial', 12), command = pca_func)
pca_button.grid(row = 4, column = 0, padx = 60, pady = 35, sticky = 'nw')

# save choice frame
save_frame = tk.Frame(win, width = 100, height = 30)
save_frame.grid(row = 2, column = 0, padx = 0, pady = 10, columnspan = 2, sticky = 'nw')
# the label notice user to compress finish or not and the next step @save_frame
sign_label = tk.Label(save_frame, width = 38, text = '         ', font = ('Arial', 12), fg = 'gray')
sign_label.grid(row = 0, column = 0, padx = 0, pady = 5, columnspan = 2 , sticky = 'nw')
# the button press to command save_jpg_button save the .jpg after compree as a .jpg @save_frame
jpg_button = tk.Button(save_frame, width = 28, text = '儲存為.jpg檔', font = ('Arial', 12), command = save_jpg_button)
jpg_button.grid(row = 1, column = 0, padx = 15, pady = 10, sticky = 'nw')
# the button press to command save_pdf_button save the .jpg after compree as a .pdf @save_frame
pdf_button = tk.Button(save_frame, width = 28, text = '儲存為.pdf檔', font = ('Arial', 12), command = save_pdf_button)
pdf_button.grid(row = 1, column = 1, padx = 5, pady = 10, sticky = 'nw')

win.mainloop()

