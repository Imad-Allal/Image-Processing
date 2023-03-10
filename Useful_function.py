

#convert image from BGR (openCV format to RGB)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)



# Generate Gaussian noise
gauss = np.random.normal(0,1,image.size)
gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')
# Add the Gaussian noise to the image
img_gauss = cv.add(image,gauss)


# Add the Speckle noise to the image
image_Sppeckle = image + image * gauss



# Add Salt & Pepper noise to the image
noise = np.random.randint(low=0, high=101, size = (image.shape[0], image.shape[1], 1))
image_SP = np.where(noise == 0, 0, image)
image_SP = np.where(noise == (100), 1, image_SP)




# Image Filtering
# ###############

# Create the Kernel 
kernel2 = np.ones((5, 5), np.float32)/25  
# Applying the filter
img = cv.filter2D(src=image, ddepth=-1, kernel=kernel2)

# we can also use the pre-built functions
#  cv2.blur(image, shapeOfTheKernel)
averageBlur = cv2.blur(image, (5, 5))

# cv2. GaussianBlur(image, shapeOfTheKernel, sigmaX )
gaussian = cv2.GaussianBlur(image, (3, 3), 0)

#cv. medianBlur(image, kernel size)
medianBlur = cv2.medianBlur(image, 9)