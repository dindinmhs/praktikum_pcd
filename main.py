import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None
        self.PreviousImage = None
        self.button_loadCitra.clicked.connect(self.fungsi)
        self.button_prosesCitra.clicked.connect(self.grayscale)
        self.button_clear.clicked.connect(self.reset)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBinary.triggered.connect(self.binary)
        self.actionGrayscale.triggered.connect(self.histGray)
        self.actionRGB.triggered.connect(self.histRGB)
        self.actionEqualization.triggered.connect(self.histEqual)
        self.actionTranslation.triggered.connect(self.translation)
        self.action45_deg.triggered.connect(lambda : self.rotation(45))
        self.actionMin45_deg.triggered.connect(lambda : self.rotation(-45))
        self.action90_deg.triggered.connect(lambda : self.rotation(90))
        self.actionMin90_deg.triggered.connect(lambda : self.rotation(-90))
        self.action180_deg.triggered.connect(lambda : self.rotation(180))
        self.action2x.triggered.connect(lambda : self.scaling(2))
        self.action3x.triggered.connect(lambda : self.scaling(3))
        self.action4x.triggered.connect(lambda : self.scaling(4))
        self.action075x.triggered.connect(lambda : self.scaling(0.75))
        self.action05x.triggered.connect(lambda : self.scaling(0.5))
        self.action025x.triggered.connect(lambda : self.scaling(0.25))
        self.actionCrop.triggered.connect(lambda : self.cropImage(50, 50, 200, 200))
        self.actionAdd.triggered.connect(lambda : self.arithmeticOperation('ADD'))
        self.actionSub.triggered.connect(lambda : self.arithmeticOperation('SUB'))
        self.actionMul.triggered.connect(lambda : self.arithmeticOperation('MUL'))
        self.actionDiv.triggered.connect(lambda : self.arithmeticOperation('DIV'))
        self.actionAND.triggered.connect(lambda : self.boolean('AND'))
        self.actionOR.triggered.connect(lambda : self.boolean('OR'))
        self.actionXOR.triggered.connect(lambda : self.boolean('XOR'))
        self.actionTranpose.triggered.connect(self.transposeImage)
        self.actionkernel5x5.triggered.connect(self.kernel5)
        self.actionkernel3x3.triggered.connect(self.kernel3)
        self.actionMean_2.triggered.connect(self.mean)
        self.actionGaussian.triggered.connect(self.gaussian)
        self.actionsharpLaplace.triggered.connect(self.sharpeningLaplace)
        self.actionsharp1.triggered.connect(self.sharpeningI)
        self.actionsharp2.triggered.connect(self.sharpeningII)
        self.actionsharp3.triggered.connect(self.sharpeningIII)
        self.actionsharp4.triggered.connect(self.sharpeningIV)
        self.actionsharp5.triggered.connect(self.sharpeningV)
        self.actionsharp6.triggered.connect(self.sharpeningVI)
        self.actionMedian.triggered.connect(self.median)
        self.actionMax_Filter.triggered.connect(self.maxFilter)
        self.actionMin_Filter.triggered.connect(self.minFilter)
        self.actionFourier.triggered.connect(self.fourier)
        self.actionDeteksi_Tepi.triggered.connect(self.deteksitepi)
        self.actionSobel.triggered.connect(self.sobel)
        self.actionPrewitt.triggered.connect(self.prewitt)
        self.actionRoberts.triggered.connect(self.robert)
        self.actionCanny.triggered.connect(self.canny)

    def fungsi(self):
        self.Image = cv2.imread('naruto.jpg')
        self.PreviousImage = cv2.imread('naruto.jpg')
        self.displayImage()
        
    def reset(self):
        self.Image = self.PreviousImage
        self.displayImage()

    def grayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H,W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        
        self.Image = gray
        self.displayImage(2)

    def brightness(self):
       try:
           self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
       except:
           pass
       
       H, W = self.Image.shape[:2]
       brightness = 70
       for i in range(H):
           for j in range(W):
               a = self.Image.item(i, j)
               b = np.clip(a + brightness, 0, 255)
               self.Image[i, j] = b
        
       self.displayImage() 
    
    def contrast(self):
       try:
           self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
       except:
           pass
       
       H, W = self.Image.shape[:2]
       contrast = 1.5
       for i in range(H):
           for j in range(W):
               a = self.Image.item(i, j)
               b = np.clip(a * contrast, 0, 255)
               self.Image[i, j] = b
        
       self.displayImage()
    
    def contrastStretching(self):
        try:
           self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
           pass
       
        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a-minV) / (maxV - minV) * 255
                self.Image[i, j] = b
            
        self.displayImage()
    
    def negative(self):
       try:
           self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
       except:
           pass
       
       H, W = self.Image.shape[:2]

       for i in range(H):
           for j in range(W):
               a = self.Image.item(i, j)
               b = math.ceil(255 - a)
               self.Image[i, j] = b
        
       self.displayImage()

    def binary(self):
       try:
           self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
       except:
           pass
       
       H, W = self.Image.shape[:2]
       threshold = 180

       for i in range(H):
           for j in range(W):
               a = self.Image.item(i, j)
               b = 255 if a >= threshold else 0
               self.Image[i, j] = b
        
       self.displayImage()
    
    def displayImage(self,label=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                    self.Image.strides[0], qformat)

        img = img.rgbSwapped()
        if label == 1 :
            self.label.setPixmap(QPixmap.fromImage(img))
        else:
            self.label_2.setPixmap(QPixmap.fromImage(img))
   
    # Histogram
    def histGray(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H,W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255).astype(np.uint8)
        
        self.Image = gray
        self.displayImage(1)
        plt.hist(self.Image.ravel(), 256, [0,255])
        plt.title('Histogram grayscale')
        plt.show()
        
    def histRGB(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.Image], [i], None, [256], [0,255])
            plt.plot(histo, color=col)
            plt.xlim([0,255])
            
        plt.title('Histogram RGB')
        plt.show()
        
    def histEqual(self):
        hist, bins = np.histogram(self.Image.ravel(), 256, [0,255])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.Image = cdf[self.Image]
        self.displayImage(1)
        
        plt.plot(cdf_normalized, color='b')
        plt.hist(self.Image.ravel(), 256, [0,256], color='r')
        plt.xlim([0,256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()         

    def translation(self):
        h, w = self.Image.shape[:2]
        quarter_h, quarter_w = h/4, w/4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w,h))
        self.Image = img
        self.displayImage(1)
    
    def rotation(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h,
        w))
        self.Image=rot_image
        self.displayImage(1)
    
    def transposeImage(self):
        transposed_image = cv2.transpose(self.Image)

        self.Image = transposed_image
        self.displayImage(1)

    def scaling(self, scale):
        resize_image = cv2.resize(self.Image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Result', resize_image)
        cv2.imshow('Original', self.Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def cropImage(self, start_row, start_col, end_row, end_col):  
        h, w = self.Image.shape[:2]
        if not (0 <= start_row < h and 0 <= end_row <= h and
                0 <= start_col < w and 0 <= end_col <= w):
            print("Error: Coordinates are out of image boundaries.")
            return

        cropped_image = self.Image[start_row:end_row, start_col:end_col]

        cv2.imshow('Cropped Image', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def arithmeticOperation(self, operation):

        first_image = cv2.imread('baju.jpg', 0)
        second_image = cv2.imread('baju2.jpeg', 0)

        h1, w1 = self.Image.shape[:2]
        h2, w2 = second_image.shape[:2]
        
        if (h1, w1) != (h2, w2):
            second_image = cv2.resize(second_image, (w1, h1))
        
        result=None

        if operation == "ADD":
            result = np.clip(first_image + second_image, 0, 255).astype(np.uint8)
        elif operation == "SUB":
            result = np.clip(first_image - second_image, 0, 255).astype(np.uint8)
        elif operation == "MUL":
            result = np.clip(first_image * second_image, 0, 255).astype(np.uint8)
        elif operation == "DIV":
            result = np.clip(first_image / second_image, 0, 255).astype(np.uint8)

        cv2.imshow('Added Image', result)
        cv2.imshow('Image 1', first_image)
        cv2.imshow('Image 2', second_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def boolean(self, operation):

        second_image = cv2.imread('baju2.jpeg', cv2.IMREAD_COLOR_RGB)
        first_image = cv2.imread('baju.jpg', cv2.IMREAD_COLOR_RGB)
        
        h1, w1 = first_image.shape[:2]
        h2, w2 = second_image.shape[:2]
        
        if (h1, w1) != (h2, w2):
            second_image = cv2.resize(second_image, (w1, h1))
            
        result = None
        
        if operation == 'AND':
            result = cv2.bitwise_and(first_image, second_image)
        elif operation == 'OR':
            result = cv2.bitwise_or(first_image, second_image)
        elif operation == 'XOR':
            result = cv2.bitwise_xor(first_image, second_image)
            
        cv2.imshow('Result', result)
        cv2.imshow('Image 1', first_image)
        cv2.imshow('Image 2', second_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Operasi Spasial
    def conv(self, X, F):
        X_height, X_width = X.shape
        F_height, F_width = F.shape
        H, W = F_height // 2, F_width // 2
        
        out = np.zeros((X_height, X_width))
        
        for i in range(H, X_height - H):
            for j in range(W, X_width - W):
                s = 0
                for k in range(-H, H + 1):
                    for l in range(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        s += w * a
                out[i, j] = s
        return out
    
    def conv2(self, img, kernel):
        k_h, k_w = kernel.shape

        # Ukuran gambar
        i_h, i_w = img.shape

        output = np.zeros((i_h - k_h + 1, i_w - k_w + 1))

        # Proses konvolusi
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                region = img[i:i + k_h, j:j + k_w]
                output[i, j] = np.sum(region * kernel)
        
        return output


    def kernel3(self):
        img = self.Image
        kernel = np.array(
            [[1000, 1000, 1000],
             [1000, 1000, 1000],
             [1000, 1000, 1000]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Kernel 3x3')
        plt.show()

    def kernel5(self):
        img = self.Image
        kernel = np.array(
            [[-1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1],
             [-1, -1, -1, -1, -1]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Kernel 5x5')
        plt.show()

    def mean(self):
        img = self.Image
        kernel = (1.0 / 9) * np.ones((3, 3))
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Mean Kernel 3x3')
        plt.show()

    def gaussian(self):
        img = self.Image
        kernel = np.array([
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ], dtype=np.float32)
        kernel = kernel / kernel.sum() 
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil gaussian')
        plt.show()

    def sharpeningLaplace(self):
        img = self.Image
        kernel = (1.0 / 16) * np.array(
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 16, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Sharpening Laplace')
        plt.show()

    def sharpeningI(self):
        img = self.Image
        kernel = (1.0 / 16) * np.array(
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Sharpening I')
        plt.show()

    def sharpeningII(self):
        img = self.Image
        kernel = (1.0 / 16) * np.array(
            [[-1, -1, -1],
             [-1, 9, -1],
             [-1, -1, -1]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Sharpening II')
        plt.show()

    def sharpeningIII(self):
        img = self.Image
        kernel = (1.0 / 16) * np.array(
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Sharpening III')
        plt.show()

    def sharpeningIV(self):
        img = self.Image
        kernel = (1.0 / 16) * np.array(
            [[1, -2, 1],
             [-2, 5, -2],
             [1, -2, 1]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Sharpening IV')
        plt.show()

    def sharpeningV(self):
        img = self.Image
        kernel = (1.0 / 16) * np.array(
            [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Sharpening V')
        plt.show()

    def sharpeningVI(self):
        img = self.Image
        kernel = (1.0 / 16) * np.array(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]])
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Sharpening VI')
        plt.show() 

    def median(self):
        img = self.Image
        hasil = img.copy()
        h, w = img.shape
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                neighbors = []
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        neighbors.append(a)
                neighbors.sort()
                median_val = neighbors[24]
                hasil[i, j] = median_val
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Median')
        plt.show()      
        
    def maxFilter(self):
        img = self.Image
        hasil = img.copy()
        h, w = img.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a > max:
                            max = a
                b = max
                hasil[i, j] = b
        plt.imshow(hasil, cmap='gray', interpolation='bicubic'), plt.title('Hasil Max Filter')
        plt.show()

    def minFilter(self):
        img = self.Image
        img_out = img.copy()
        h, w = img.shape[:2]

        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                min = 255
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a < min:
                            min = a
                    b = min
                    img_out[i, j] = b

        plt.imshow(img_out, cmap='gray', interpolation='bicubic'), plt.title('Hasil Min Filter')
        plt.show()

    def fourier(self):
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)
        y += max(y)

        Img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)

        plt.imshow(Img, cmap='gray')
        plt.show()

        Img = self.Image.astype(np.float32)

        if len(Img.shape) == 2:
            dft = cv2.dft(Img, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
            rows, cols = Img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols, 2), np.uint8)
            r = 50
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
            mask[mask_area] = 1

            fshift = dft_shift * mask
            fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
            f_shift = np.fft.ifftshift(fshift)

            Img_back = cv2.idft(f_shift)
            Img_back = cv2.magnitude(Img_back[:, :, 0], Img_back[:, :, 1])

            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(Img, cmap='gray')
            ax1.set_title('Input Image')
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(magnitude_spectrum, cmap='gray')
            ax2.set_title('FFT of Image')
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.imshow(fshift_mask_mag, cmap='gray')
            ax3.set_title('FFT + Mask')
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.imshow(Img_back, cmap='gray')
            ax4.set_title('Inverse Fourier')
            plt.show()
        else:
            print("Input image should be a single-channel grayscale image.")
    
    def deteksitepi(self):
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)

        y += max(y)

        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)

        img = self.Image

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 80
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()

    def sobel(self):
        img = self.Image

        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        img_x = self.conv(img, Sx)
        img_y = self.conv(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def prewitt(self):
        img = self.Image

        Sx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
        img_x = self.conv(img, Sx)
        img_y = self.conv(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255
        
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    def robert(self):
        
        img = self.Image

        Sx = np.array([[1, 0],
                       [0, -1]])
        Sy = np.array([[0, 1],
                       [-1, 0]])
        img_x = self.conv2(img, Sx)
        img_y = self.conv2(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out))* 255
        
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()
    
    def canny(self):
        try :
            img = self.Image
            
            conv = (1 / 345) * np.array(
                [[1, 5, 7, 5, 1],
                [5, 20, 33, 20, 5],
                [7, 33, 55, 33, 7],
                [5, 20, 33, 20, 5],
                [1, 5, 7, 5, 1]])

            out_img = self.conv(img, conv)
            out_img = out_img.astype("uint8")
            cv2.imshow("Noise reduction", out_img)

            # finding gradien
            Sx = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
            Sy = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
            img_x = self.conv(out_img, Sx)
            img_y = self.conv(out_img, Sy)
            img_out = np.sqrt(img_x * img_x + img_y * img_y)
            img_out = (img_out / np.max(img_out)) * 255
            cv2.imshow("finding Gradien", img_out)
            theta = np.arctan2(img_y, img_x)

            angle = theta * 180. / np.pi
            angle[angle < 0] += 180
            H, W = img.shape[:2]
            Z = np.zeros((H, W), dtype=np.int32)
            for i in range(1, H - 1):
                for j in range(1, W - 1):

                    try:
                        q = 255
                        r = 255

                        # angle 0
                        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                            q = img_out[i, j + 1]
                            r = img_out[i, j - 1]
                            # angle 45
                        elif (22.5 <= angle[i, j] < 67.5):
                            q = img_out[i + 1, j - 1]
                            r = img_out[i - 1, j + 1]
                            # angle 90
                        elif (67.5 <= angle[i, j] < 112.5):
                            q = img_out[i + 1, j]
                            r = img_out[i - 1, j]
                            # angle 135
                        elif (112.5 <= angle[i, j] < 157.5):
                            q = img_out[i - 1, j - 1]
                            r = img_out[i + 1, j + 1]
                        if (img_out[i, j] >= q) and (img_out[i, j] >= r):

                            Z[i, j] = img_out[i, j]
                        else:
                            Z[i, j] = 0
                    except IndexError as e:
                        pass
            img_N = Z.astype("uint8")
            cv2.imshow("Non Maximum Suppression", img_N)
            
            weak = 15
            strong = 90
            for i in np.arange(H):
                for j in np.arange(W):
                    a = img_N.item(i, j)
                    if (a > weak):
                        b = weak
                        if (a > strong):
                            b = 255
                    else:
                        b = 0
                    img_N[i,j] = b
            img_H1 = img_N.astype("uint8")
            cv2.imshow("hysteresis part 1", img_H1)
            # hysteresis treshold part 2
            strong = 255
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    if (img_H1[i, j] == weak):
                        try:
                            if ((img_H1[i + 1, j - 1] == strong) or (img_H1[i + 1, j] == strong) or (
                                    img_H1[i + 1, j + 1] == strong) or (img_H1[i, j - 1] == strong) or (
                                    img_H1[i, j + 1] == strong) or (img_H1[i - 1, j - 1] == strong) or (
                                    img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                                img_H1[i, j] = strong
                            else:
                                img_H1[i, j] = 0
                        except IndexError as e:
                            pass
            img_H2 = img_H1.astype("uint8")
            cv2.imshow("hysteresis part 2", img_H2)
        except Exception as e:
            print(e)

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 1')
window.show()
sys.exit(app.exec_())