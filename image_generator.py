import os
import string
from PIL import Image
from PIL import ImageFont, ImageDraw, ImageOps

width, height = 100, 100
letter_uppercase = list(string.ascii_uppercase)
digits = list(string.digits)
texts = letter_uppercase + digits

for text in texts:
    # enumerate the texts
    os.makedirs('./hologram_image/{}'.format(text), exist_ok=True)

    for font_size in range(30, 105, 5):
        # change the font size
        img = Image.new("L", (width, height), color=255)   # "L": (8-bit pixels, black and white)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", font_size)

        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(text, font=font)
        h += int(h * 0.21)
        draw.text(((width-w)/2, (height-h)/2), text=text, fill='black', font=font)

        for angle in range(0, 360, 5):
            # rotate the text every 5 degrees
            rot_im = img.rotate(angle, expand=False, fillcolor="white")
            rot_im.save('./hologram_image/{}/{}_size_{}_angle_{}_thickness.png'.format(text, text, font_size, angle))
