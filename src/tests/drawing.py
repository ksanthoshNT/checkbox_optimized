import cv2
import numpy as np
from PIL import Image, ImageDraw

# Convert numpy array to PIL Image
img = Image.fromarray(cv2.imread("/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg"))

# Create drawing object
draw = ImageDraw.Draw(img)

# Draw points (as small circles)
# draw.ellipse([10-2, 10-2, 10+2, 10+2], fill='red')  # point at (10,10)
# draw.ellipse([22-2,/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg 25-2, 22+2, 25+2], fill='red')  # point at (22,25)
x,y = 390,300
draw.rectangle([x,y,x+40,y+40],fill='red')

# Convert back to numpy array if needed
result_array = np.array(img)

# Save or show
img.save('marked_image.png')