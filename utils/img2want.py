import os
import glob
import numpy as np
from PIL import Image


def img2want(opt):
    if opt.format == 'JPEG':
        suffix = 'jpg'
    elif opt.format == 'PNG':
        suffix = 'png'
    else:
        raise NotImplementedError('Unknown format: {}'.format(opt.format))
    savedir = opt.output_dir
    os.makedirs(savedir, exist_ok=True)
    print("The converted Images[%s] are at %s" % (opt.format, opt.output_dir))

    photo_expr = opt.photo_dir + "/*.*"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    print('The size of photo is %d' % len(photo_paths))
    for i, photo_path in enumerate(photo_paths):
        photo = Image.open(photo_path).convert('RGB')
        if opt.crop:
            if opt.add_random:
                r = (np.random.random(4)-0.5)*40
            else:
                r = [0, 0, 0, 0]
            print("crop to [{},{}][{},{}]".format(opt.x1+r[0], opt.y1+r[1], opt.x2+r[2], opt.y2+r[3]))
            photo = photo.crop((opt.x1+r[0], opt.y1+r[1], opt.x2+r[2], opt.y2+r[3]))

        if opt.resize is not None:
            w, h = photo.size
            if w < h:
                h = int(h*opt.resize/w)
                w = int(opt.resize)
            else:
                w = int(w*opt.resize/h)
                h = int(opt.resize)
            photo.resize(w, h)
        savepath = os.path.join(savedir + "/%d.%s" % (i, suffix))
        photo.save(savepath, format=opt.format, subsampling=0, quality=100)
    print('Finish convert to [%s]' % opt.format)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--photo_dir', type=str, required=True,
                        help='Path to the photo directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory the output images will be written to.')
    parser.add_argument('--format', type=str, default='JPEG', choices=['JPEG', 'PNG'],
                        help='the format converted to.')
    parser.add_argument('--resize', type=int, default=None,
                        help='to resize the h or w of the picture')
    parser.add_argument('--crop', action='store_true', help='whether to crop the picture')
    parser.add_argument('--add_random', action='store_true', help='wheter to random change the size when cropping')
    parser.add_argument('--x1', type=int, default=None,
                        help='x1')
    parser.add_argument('--y1', type=int, default=None,
                        help='y1')
    parser.add_argument('--x2', type=int, default=None,
                        help='x2')
    parser.add_argument('--y2', type=int, default=None,
                        help='y2')
    opt = parser.parse_args()

    print("Start converting Photo in %s" % opt.photo_dir)
    img2want(opt)

