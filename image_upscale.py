from super_image import PanModel, ImageLoader
from PIL import Image

img = Image.open('label.jpg')
#img = img.resize((int(img.width/2),int(img.height/2)))

model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=2)
inputs = ImageLoader.load_image(img)

preds = model(inputs)

ImageLoader.save_image(preds, './scaled_2x.png')
ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')
