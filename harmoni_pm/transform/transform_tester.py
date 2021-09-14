#
# Copyright (c) 2020 Gonzalo J. Carracedo <BatchDrake@gmail.com>
# 
#
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this 
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from PIL import Image, ImageDraw, ImageFont
from harmoni_pm.common.array import FloatArray

from skyfield.api import Star, load
from skyfield.data import hipparcos
from skyfield.units import Angle
from matplotlib import cm

import numpy as np


IMAGE_WIDTH         = 1920
IMAGE_HEADER_HEIGHT = 320
IMAGE_MARGIN_WIDTH  = 24
IMAGE_POINT_RADIUS  = 4
IMAGE_COLORMAP_SIZE = 256

IMAGE_DEFAULT_VECTOR_COLOR      = "#ff0000"
IMAGE_DEFAULT_ORIGINAL_COLOR    = "black"
IMAGE_DEFAULT_DESTINATION_COLOR = "red"
IMAGE_DEFAULT_LIMIT_COLOR       = "black"
IMAGE_DEFAULT_TYPE              = "vector"
IMAGE_DEFAULT_COLORMAP          = "inferno"

class TransformTester:
    def __init__(self, transform):
        self.transform = transform
        self.planets = load('de421.bsp')
        self.earth = self.planets['earth']
        self.radlimit = None
        
        self.vector_color      = IMAGE_DEFAULT_VECTOR_COLOR
        self.original_color    = IMAGE_DEFAULT_ORIGINAL_COLOR
        self.destination_color = IMAGE_DEFAULT_DESTINATION_COLOR
        self.point_radius      = IMAGE_POINT_RADIUS
        self.colormap          = IMAGE_DEFAULT_COLORMAP
        
    def prepare_dataset(self):
        self.input       = self.point_array
        self.origin_desc = "Undistorted pattern"
        
        self.point_count = self.point_array.shape[0]
        self.result      = self.point_array
        self.result_desc = self.origin_desc
        
    def generate_stars(self, ra, dec, ra_width, dec_width, maglimit):
        self.width  = ra_width
        self.height = dec_width
        self.delta  = None
        self.J      = None
        
        # The "False" arguments below are to prevent skyfield from assuming
        # that I am not in full command of my cognitive abilities
        
        self.desc   = "Field stars in RADEC {0} / {1}".format(
            Angle(degrees = ra).hstr(warn = False), 
            Angle(degrees = dec).dstr(warn = False)) 
        
        df = hipparcos.load_dataframe(load.open(hipparcos.URL))

        df = df[(df['magnitude'] <= maglimit)
                & (df['ra_degrees']  >= ra  - .5 * ra_width)
                & (df['ra_degrees']  <  ra  + .5 * ra_width)
                & (df['dec_degrees'] >= dec - .5 * dec_width)
                & (df['dec_degrees'] <  dec + .5 * dec_width)]
        
        print("Selected: {0} stars in field".format(len(df)))
        
        stars = Star.from_dataframe(df)
        ts = load.timescale()
        t = ts.now()
        # t = ts.utc(2020, 12, 24)
        
        astrometric = self.earth.at(t).observe(stars)
        ralist, declist, distance = astrometric.radec()
        
        self.sizes = .5 * 10. ** (.5 * (2.5 -.4 * FloatArray.make(df['magnitude'])))
        
        self.point_array = np.transpose(
            FloatArray.make(
                [-ralist._degrees + ra, declist.degrees - dec]))
    
        self.prepare_dataset()
        
    def generate_points(self, width, height, delta_x, delta_y, radlimit = None):
        self.width    = width
        self.height   = height
        self.delta    = FloatArray.make([delta_x, delta_y])
        self.J        = None
        self.sizes    = None
        self.radlimit = radlimit
        
        if radlimit is not None:
            self.desc = "Rectangular grid (radius {0})".format(radlimit)
        else:
            self.desc   = "Rectangular grid of points"
         
        rows = int(np.floor(height / delta_y))
        cols = int(np.floor(width  / delta_x))
        count = 0
        self.point_array = FloatArray([rows * cols, 2])
        
        # TODO: matrix-hack this?
        for i in range(cols):
            for j in range(rows):
                x = (.5 + i - np.floor(cols) / 2) * delta_x
                y = (.5 + j - np.floor(rows) / 2) * delta_y
                if radlimit is None or np.sqrt(x * x + y * y) < radlimit:
                    self.point_array[count, 0] = x
                    self.point_array[count, 1] = y
                count += 1
        
        self.prepare_dataset()
        
    def backfeed(self):
        self.input = self.result
        self.origin_desc += u" → " + self.result_desc
        
    def sample(self, event = "manufacture"):
        self.transform.generate(event)
        
    def forward(self):
        self.J      = None
        self.result = self.transform.forward(self.input)
        self.result_desc = "Forward"
        
    def backward(self):
        self.J      = None
        self.result = self.transform.backward(self.input)
        self.result_desc = "Backward"
    
    # TODO: Verify whether there is delta information
    def forward_jacobian(self):
        dx = FloatArray.make([self.delta[0], 0])
        dy = FloatArray.make([0, self.delta[1]])
        
        self.forward()
        
        dTdx = (self.transform.forward(self.input + dx) - dx - self.result) / dx[0]
        dTdy = (self.transform.forward(self.input + dy) - dy - self.result) / dy[1]
        
        self.J = [dTdx, dTdy]
    
    def backward_jacobian(self):
        dx = FloatArray.make([self.delta[0], 0])
        dy = FloatArray.make([0, self.delta[1]])
        
        self.backward()
        
        dTdx = (self.transform.backward(self.input + dx) - self.result) / dx[0]
        dTdy = (self.transform.backward(self.input + dy) - self.result) / dy[1]
        
        self.J = [dTdx, dTdy]
    
    def distortion_rms(self):
        err =  (self.point_array - self.result)
        return np.sqrt(np.mean(err[:, 0] * err[:, 0] + err[:, 1] * err[:, 1]))
    
    def draw_vectors(self, draw, m, x0y0, scale = 1):
        for i in range(self.point_count):
            if self.sizes is None:
                radius = self.point_radius
            else:
                radius = self.sizes[i]
                
            xy1 = self.point_array[i, :] * m + x0y0
            orig = [(xy1[0] - radius, xy1[1] - radius),
                    (xy1[0] + radius, xy1[1] + radius)]
            
            xy2 = (self.point_array[i, :] + scale * (self.result[i, :] - self.point_array[i, :])) * m + x0y0
            dest = [(xy2[0] - radius, xy2[1] - radius),
                    (xy2[0] + radius, xy2[1] + radius)] 
            
            rect = [(xy1[0], xy1[1]), (xy2[0], xy2[1])]
            
            draw.ellipse(
                orig, 
                fill = self.original_color, 
                outline = self.original_color)
            draw.line(rect, fill = self.vector_color, width = 1)
            draw.ellipse(
                dest, 
                fill = self.destination_color, 
                outline = self.destination_color)
        
    @staticmethod
    def _to_rgb(c):
        return (int(255 * c[0]), int(255 * c[1]), int(255 * c[2]))
    
    def draw_colorbar(self, draw, m, x0y0, fmin, fmax, cmap):
        x0 = -.5 * m[0] * self.width  + x0y0[0]
        y0 = .5 * m[1] * self.height + x0y0[1] - 3 * IMAGE_MARGIN_WIDTH
        
        x1 = .5 * m[0] * self.width + x0y0[0]
        y1 = .5 * m[1] * self.height + x0y0[1] - IMAGE_MARGIN_WIDTH
        
        step = (x1 - x0) / IMAGE_COLORMAP_SIZE
        
        for i in range(IMAGE_COLORMAP_SIZE):
            xp = i * step + x0
            xn = xp + step
            
            v  = self._to_rgb(cmap(i / float(IMAGE_COLORMAP_SIZE)))
            
            draw.rectangle([int(xp), int(y0), int(xn), int(y1)], v)
                
        font = ImageFont.truetype('Tuffy_Bold.ttf', 30)
        
        draw.rectangle([x0, y0, x1, y1], None, "black", 2)
        
        txt = u"{0:e}".format(fmin)
        dim = font.getsize(txt)
        draw.text(
            (x0 + .5 * dim[1], (y1 + y0) * .5 - dim[1] * .5),
            txt,
            fill = self._to_rgb(cmap(1.)),
            font = font)

        txt = u"{0:e}".format(fmax)
        dim = font.getsize(txt)
        draw.text(
            (x1 - dim[0] - .5 * dim[1], (y1 + y0) * .5 - dim[1] * .5),
            txt,
            fill = self._to_rgb(cmap(0.)),
            font = font)

    def draw_scalar_field(self, draw, field, m, x0y0, f0 = None, f1 = None):
        cmap = cm.get_cmap(self.colormap, IMAGE_COLORMAP_SIZE)
        
        if f0 is None:
            f0 = np.min(field)
            
        if f1 is None:
            f1 = np.max(field)
        
        h    = 1. / (f1 - f0)
        
        for i in range(self.point_count):
            xy0 = (self.point_array[i, :] - .5 * self.delta) * m + x0y0
            xy1 = (self.point_array[i, :] + .5 * self.delta) * m + x0y0
            x0 = int(xy0[0])
            y0 = int(xy0[1])
            x1 = int(xy1[0])
            y1 = int(xy1[1])
            
            draw.rectangle(
                [x0, y0, x1, y1],
                self._to_rgb(cmap((field[i] - f0) * h)),
                width = 0)
            
        
        self.draw_colorbar(draw, m, x0y0, f0, f1, cmap)
            
    def draw_jacobian_det(self, draw, m, x0y0):
        field = (self.J[0][:, 0]) * (self.J[1][:, 1]) - self.J[0][:, 1] * self.J[1][:, 0]
        self.draw_scalar_field(draw, field, m, x0y0)
        
    def draw_divergence(self, draw, m, x0y0):
        field = self.J[0][:, 0] + self.J[1][:, 1]
        self.draw_scalar_field(draw, field, m, x0y0)
    
    def draw_curl(self, draw, m, x0y0):
        field = -self.J[1][:, 0] + self.J[0][:, 1]
        self.draw_scalar_field(draw, field, m, x0y0)
    
    def draw_ex(self, draw, m, x0y0):
        field = (self.result - self.input)[:, 0]
        self.draw_scalar_field(draw, field, m, x0y0)
        
    def draw_ey(self, draw, m, x0y0):
        field = (self.result - self.input)[:, 1]
        self.draw_scalar_field(draw, field, m, x0y0)
    
    def draw_e(self, draw, m, x0y0):
        e = self.result - self.input
        field = np.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2)
        self.draw_scalar_field(draw, field, m, x0y0)
    
    def draw_rotation(self, draw, m, x0y0):
        e = self.result - self.input
        field = (np.arctan2(e[:, 1], e[:, 0]) - np.arctan2(self.input[:, 1], self.input[:, 0]))
        self.draw_scalar_field(draw, np.abs(np.sin(field)), m, x0y0, 0, 1)
        
    def save_to_image(self, path, maptype = IMAGE_DEFAULT_TYPE, zoom = 1):
        # Calculate geometry of the resulting image
        width        = int(IMAGE_WIDTH)
        field_width  = np.ceil(width - 2 * IMAGE_MARGIN_WIDTH)
        scale        = field_width / self.width
        field_height = np.ceil(self.height * scale)
        hdr_height   = IMAGE_HEADER_HEIGHT + IMAGE_MARGIN_WIDTH
        height       = int(hdr_height + field_height + IMAGE_MARGIN_WIDTH)
        field_corner = (IMAGE_MARGIN_WIDTH, hdr_height)
        field_center = (field_corner[0] + field_width / 2, field_corner[1] + field_height / 2)
        field_box    = [field_corner, (IMAGE_MARGIN_WIDTH + field_width, hdr_height + field_height)] 
        descs        = {
            "vector"      : "displacement vector field",
            "curl"        : "transform curl",
            "divergence"  : "transform divergence",
            "determinant" : "jacobian determinant",
            "rotation"    : "rotation w.r.t scaling",
            "offset"      : "total displacement",
            "offset-x"    : "horizontal displacement",
            "offset-y"    : "vertical displacement"}
            
        m            = scale * FloatArray.make((1, -1))
        x0y0         = FloatArray.make(field_center)
        
        im = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(im)
            
        fnt_big   = ImageFont.truetype('Tuffy_Bold.ttf', 60)
        fnt_small = ImageFont.truetype('Tuffy_Bold.ttf', 36)

        draw.text(
            (IMAGE_MARGIN_WIDTH, IMAGE_MARGIN_WIDTH),
            u"Transform test pattern for \"" + type(self.transform).__name__ + "\"",
            font = fnt_big,
            fill = "#000000")
            
        draw.text(
            (IMAGE_MARGIN_WIDTH, IMAGE_MARGIN_WIDTH + 64),
            u"Transform chain: " + self.origin_desc + u" → " + self.result_desc,
            font = fnt_small,
            fill = "#000000")
        
        draw.text(
            (IMAGE_MARGIN_WIDTH, IMAGE_MARGIN_WIDTH + 100),
            u"Field geometry: {0}x{1} (error scaled by {2})".format(self.width, self.height, zoom),
            font = fnt_small,
            fill = "#000000")
        
        draw.text(
            (IMAGE_MARGIN_WIDTH, IMAGE_MARGIN_WIDTH + 136),
            u"RMS of distortion: {0}".format(self.distortion_rms()),
            font = fnt_small,
            fill = "#000000")
        
        draw.text(
            (IMAGE_MARGIN_WIDTH, IMAGE_MARGIN_WIDTH + 172),
            u"Description: {0}".format(self.desc),
            font = fnt_small,
            fill = "#000000")
                
        draw.text(
            (IMAGE_MARGIN_WIDTH, IMAGE_MARGIN_WIDTH + 208),
            u"Variable: {0}".format(descs[maptype]),
            font = fnt_small,
            fill = "#000000")
        
        # Enclose it
        if self.radlimit is not None:
            draw.ellipse((x0y0[0] - scale * self.radlimit, x0y0[1] - scale * self.radlimit, x0y0[0] +scale * self.radlimit, x0y0[1]  + scale * self.radlimit), width=(IMAGE_POINT_RADIUS * 2), outline=IMAGE_DEFAULT_LIMIT_COLOR)
        else:
            draw.rectangle(field_box, fill='white', outline="gray", width=(IMAGE_POINT_RADIUS * 2))
            
        # Draw
        if maptype == "vector":
            self.draw_vectors(draw, m, x0y0, zoom)
        elif maptype == "determinant":
            self.draw_jacobian_det(draw, m, x0y0)
        elif maptype == "divergence":
            self.draw_divergence(draw, m, x0y0)
        elif maptype == "curl":
            self.draw_curl(draw, m, x0y0)
        elif maptype == "rotation":
            self.draw_rotation(draw, m, x0y0)
        elif maptype == "offset":
            self.draw_e(draw, m, x0y0)
        elif maptype == "offset-x":
            self.draw_ex(draw, m, x0y0)
        elif maptype == "offset-y":
            self.draw_ey(draw, m, x0y0)
            
        del draw
        im.save(path)
        
