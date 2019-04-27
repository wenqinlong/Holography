import os
import string
from PIL import Image
from PIL import ImageFont, ImageDraw

width, height = 100, 100
digits = list(string.digits)
letter_uppercase = list(string.ascii_uppercase)

texts = digits + letter_uppercase

label = 0
for text in texts:
    # enumerate the texts
    os.makedirs('./hologram_image/{}'.format(text), exist_ok=True)
    num = 0
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
            rot_im.save('./hologram_image/{}/{num:04}_{}_{}_size_{font_size:03}_angle_{angle:03}.png'.format(text, num, label, text, font_size, angle))
            num += 1
    label += 1
