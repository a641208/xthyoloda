import cv2

img = cv2.imread("D:/temp/rddc2020-master\CAMdataset/5.jpg")  # Read image

# Setting parameter values
t_lower = 0.025  # Lower Threshold
t_upper = 0.065  # Upper threshold

# Applying the Canny Edge filter
edge = cv2.Canny(img, t_lower, t_upper)

#cv2.imshow('original', img)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()