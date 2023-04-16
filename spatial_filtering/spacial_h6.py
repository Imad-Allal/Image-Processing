import cv2
import numpy as np
import matplotlib.pyplot as plt

I = np.zeros((250, 250), dtype=np.uint8)

I.fill(50)

I[100:150, 100:150] = 200

h4 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)
I1 = cv2.filter2D(I, -1, h4)

I_norm = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
I1_norm = cv2.normalize(I1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
I1_norm_abs = cv2.normalize(np.abs(I1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(I_norm, cmap='gray')
axs[0].set_title('I')
axs[1].imshow(I1_norm_abs, cmap='gray')
axs[1].set_title('mat2gray(abs(I1))')
axs[2].imshow(I1_norm, cmap='gray')
axs[2].set_title('mat2gray(I1)')
plt.show()