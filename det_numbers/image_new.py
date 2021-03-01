import os
import random
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype, load_default,load
import numpy as np

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
try:
    from wheezy.captcha import image as wheezy_captcha
except ImportError:
    wheezy_captcha = None

DEFAULT_FONTS = ['c:\windows\fonts\arial.ttf',
                 'c:\windows\fonts\calibri.ttf',
                 'c:\windows\fonts\cambria.ttf',
                 'c:\windows\fonts\Times New Roman.ttf']
if wheezy_captcha:
    __all__ = ['ImageCaptcha', 'WheezyCaptcha']
else:
    __all__ = ['ImageCaptcha']

table = []
for i in range(256):
    table.append(i * 1.97)


def trans(img):
    # *********** Begin **********#
    # 编写函数依次处理每个像素点
    # 要求:若r、g、b、的均值大于230 则表示该像素的颜色接近白色,属于背景中的像素点,将其背景色设置为透明。否则将该像素的颜色替换为黑色。
    # img = img.convert('RGBA')

    x, y = img.size
    print(x, y)
    for i in range(x):
        for j in range(y):
            color = img.getpixel((i, j))
            if color[3] == 0:
                color = (255, 255, 255)
            # Mean = np.mean(list(color[:-1]))
            # print(Mean)
            # if Mean > 230:
            #     color = color[:-1] + (0,)
            else:
                color = (0, 0, 0)
            img.putpixel((i, j), color)


class _Captcha(object):
    def generate(self, chars, format='png'):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out1 = BytesIO()
        out2 = BytesIO()
        # im[0].save(out1, format=format)
        im.save(out2, format=format)
        # out1.seek(0)
        out2.seek(0)
        return out2

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        im = self.generate_image(chars)
        return im.save(output, format=format)


class WheezyCaptcha(_Captcha):
    """Create an image CAPTCHA with wheezy.captcha."""

    def __init__(self, width=200, height=75, fonts=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS

    def generate_image(self, chars):
        text_drawings = [
            wheezy_captcha.warp(),
            wheezy_captcha.rotate(),
            wheezy_captcha.offset(),
        ]
        fn = wheezy_captcha.captcha(
            drawings=[
                wheezy_captcha.background(),
                wheezy_captcha.text(fonts=self._fonts, drawings=text_drawings),
                wheezy_captcha.curve(),
                wheezy_captcha.noise(),
                wheezy_captcha.smooth(),
            ],
            width=self._width,
            height=self._height,
        )
        return fn(chars)


class ImageCaptcha(_Captcha):
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """

    def __init__(self, width=160, height=60, fonts=["C:\\WINDOWS\\Fonts\\SIMYOU.TTF"], font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or [56]# list(range(53, 56))
        self._truefonts = []
        # for n in self._fonts:
        #     # print(n)

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    @staticmethod
    def create_noise_curve(image, color):
        w, h = image.size
        x1 = random.randint(0, int(w / 5))
        x2 = random.randint(w - int(w / 5), w)
        y1 = random.randint(int(h / 5), h - int(h / 5))
        y2 = random.randint(y1, h - int(h / 5))
        points = [x1, y1, x2, y2]
        end = random.randint(160, 200)
        start = random.randint(0, 20)
        Draw(image).arc(points, start, end, fill=color)
        return image

    @staticmethod
    def create_noise_dots(image, color, width=3, number=30):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image

    def create_captcha_image(self, chars, color, background1, background2):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image1 = Image.new('RGB', (self._width, self._height), background1)
        image2 = Image.new('RGB', (self._width, self._height), background2)
        draw1 = Draw(image1)
        draw1 = Draw(image2)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            # font = load_default().font
            # print(font)
            w, h = draw1.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = Image.new('RGBA', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)
            # im.show()

            # rotate
            # im = im.crop(im.getbbox())
            # im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)
            #
            # # warp
            # dx = w * random.uniform(0.1, 0.3)
            # dy = h * random.uniform(0.2, 0.3)
            # x1 = int(random.uniform(-dx, dx))
            # y1 = int(random.uniform(-dy, dy))
            # x2 = int(random.uniform(-dx, dx))
            # y2 = int(random.uniform(-dy, dy))
            # w2 = w + abs(x1) + abs(x2)
            # h2 = h + abs(y1) + abs(y2)
            # data = (
            #     x1, y1,
            #     -x1, h2 - y2,
            #     w2 + x2, h2 + y2,
            #     w2 - x2, -y1,
            # )
            # im = im.resize((w2, h2))
            # im = im.transform((w, h), Image.QUAD, data)
            return im

        images = []
        for c in chars:
            # if random.random() > 0.5:
            #     images.append(_draw_character(" "))
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image1 = image1.resize((width, self._height))
        image2 = image2.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        for im in images:
            w, h = im.size
            # im.show()
            r, g, b, a = im.split()
            # print(r, g, b, a)
            # mask = trans(im)
            mask = im.convert('L').point(table)
            image1.paste(im, (offset, int((self._height - h) / 2)), mask = a)
            image2.paste(im, (offset, int((self._height - h) / 2)), mask = a)
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            image1 = image1.resize((self._width, self._height))
            image2 = image2.resize((self._width, self._height))

        return image1, image2

    def generate_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background1 = random_color(238, 255)
        background2 = random_color(255, 255)
        color1 = random_color(10, 220, random.randint(220, 220))
        color1 = (0, 0, 0)
        im1, im2 = self.create_captcha_image(chars, color1, background1, background2)
        # 无背景im2中不添加干扰线和噪点#
        im2 = im2.filter(ImageFilter.SMOOTH)
        # self.create_noise_dots(im1, color1)
        # self.create_noise_curve(im1, color1)
        # im1 = im1.filter(ImageFilter.SMOOTH)
        return im2


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)