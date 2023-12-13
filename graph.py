import matplotlib.pyplot as plt
from matplotlib import image
image_path = "C:/UCI/Project/facial_expressions/images/Abdoulaye_Wade_0004.jpg"
example = image.imread(image_path)
plt.imshow(example, cmap='gray')
plt.xlabel("Expression: happinese")
plt.show()