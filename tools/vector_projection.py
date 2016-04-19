""" Write graphic illustrating vector projection
"""

import sys
from distutils.version import LooseVersion

import numpy as np

from pyx import canvas, path, deco, __version__
from pyx.style import linestyle as ls
from pyx.color import rgb

if LooseVersion(__version__) < '0.14':
    raise RuntimeError("Need PyX >= 0.14 + Python 3")


w = np.array((3, 4))
v = np.array((2, 1))
c = w.dot(v) / v.dot(v)
w_m_cv = w - c * v


def make_line(end, origin=(0, 0)):
    return path.line(*(tuple(origin) + tuple(end)))


def text_at(cnv, point, *args):
    cnv.text(*(tuple(point) + args))


def build_canvas():
    cnv = canvas.canvas()
    cnv.stroke(make_line(w), [deco.earrow()])
    text_at(cnv, w / 2. - (.5, 0), r'$\vec{w}$')
    cnv.stroke(make_line(v), [deco.earrow()])
    text_at(cnv, v / 2. - (0, .5), r'$\vec{v}$')
    cnv.stroke(make_line(c * v, w), [deco.earrow(), rgb.red])
    text_at(cnv, w - (w - c * v) / 2 + (.1, .1), r'$\vec{w} - c \vec{v}$')
    cnv.stroke(make_line(c * v), [deco.earrow(), ls.dashed, rgb.blue])
    text_at(cnv, c * v / 1.5 - (0, .5), r'$c \vec{v}$')
    cnv.stroke(make_line(v * -1.5, v * 3), [ls.dotted])
    return cnv


def main():
    out_root = sys.argv[1]
    cnv = build_canvas()
    cnv.writePDFfile(out_root)
    cnv.writeSVGfile(out_root)


if __name__ == '__main__':
    main()
