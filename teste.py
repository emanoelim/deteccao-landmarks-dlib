import dlib
from skimage import io
import matplotlib.pyplot as plt


img = io.imread("IMG_20190421_145140.jpg")
predictor_path = "shape_predictor_68_face_landmarks(4).dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
shape = detector(img, 1)
shape = predictor(img, shape[0])
x = []
y = []
for i in range(0,68):
	x.append(shape.part(i).x)
	y.append(shape.part(i).y)
plt.imshow(img)
plt.scatter(x, y, c='r')
plt.show()

