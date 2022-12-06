import cv2
import matplotlib.pyplot as plt

c_img = cv2.imread("7.png")

crop_img = c_img[714:2228,341:1420]

test_images = [
    "1.png",
    "5.png",
    "3.png",
    "2.png",
    "10.png",
    "9.png",
    "6.png",
    "8.png",
    "4.png",
    "7.png"
]

correlations = []

for img in test_images:
    testImg = cv2.imread(img)
    croppedTestImg = testImg[714:2228,341:1420]
    plt.imshow(croppedTestImg)
    plt.show()
    X = croppedTestImg - crop_img
    ssd = sum(X[:]**2)
    correlations.append(ssd)
print(correlations)

cv2.imshow("correlated",c_img)
plt.imshow(crop_img)
plt.show()