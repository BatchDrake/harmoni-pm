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

import numpy as np

IMAGE_WIDTH         = 1920
IMAGE_HEADER_HEIGHT = 320
IMAGE_MARGIN_WIDTH  = 24
IMAGE_POINT_RADIUS  = 3

class TransformTester:
    def __init__(self, transform):
        self.transform = transform
        self.planets = load('de421.bsp')
        self.earth = self.planets['earth']
        
    def prepare_dataset(self):
        self.input       = self.point_array
        self.origin_desc = "Undistorted pattern"
        
        self.point_count = self.point_array.shape[0]
        self.result      = self.point_array
        self.result_desc = self.origin_desc
        
    def generate_stars(self, ra, dec, ra_width, dec_width, maglimit):
        self.width  = ra_width
        self.height = dec_width
        
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
                [ralist._degrees - ra, declist.degrees - dec]))
    
        self.prepare_dataset()
        
    def generate_points(self, width, height, delta_x, delta_y):
        self.width  = width
        self.height = height
        self.sizes  = None
        self.desc   = "Rectangular grid of points"
         
        rows = int(np.floor(height / delta_y))
        cols = int(np.floor(width  / delta_x))
        count = 0
        self.point_array = FloatArray([rows * cols, 2])
        
        # TODO: matrix-hack this?
        for i in range(cols):
            for j in range(rows):
                self.point_array[count, 0] = (.5 + i - np.floor(cols) / 2) * delta_x
                self.point_array[count, 1] = (.5 + j - np.floor(rows) / 2) * delta_y
                count += 1
        
        self.prepare_dataset()
        
    def backfeed(self):
        self.input = self.result
        self.origin_desc += u" → " + self.result_desc
        
    def sample(self):
        self.transform.generate()
        
    def forward(self):
        self.result = self.transform.forward(self.input)
        self.result_desc = "Forward"
        
    def backward(self):
        self.result = self.transform.backward(self.input)
        self.result_desc = "Backward"
    
    def distortion_rms(self):
        err =  (self.point_array - self.result)
        return np.mean(np.sqrt(err[:, 0] * err[:, 0] + err[:, 1] * err[:, 1]))
    
    def save_to_image(self, path):
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
            u"Field geometry: {0}x{1}".format(self.width, self.height),
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
                
        # Enclose it
        draw.rectangle(field_box, fill='black', outline="gray", width=(IMAGE_POINT_RADIUS * 2))
        
        for i in range(self.point_count):
            if self.sizes is None:
                radius = IMAGE_POINT_RADIUS
            else:
                radius = self.sizes[i]
                
            xy1 = self.point_array[i, :] * m + x0y0
            orig = [(xy1[0] - radius, xy1[1] - radius),
                    (xy1[0] + radius, xy1[1] + radius)]
            
            xy2 = self.result[i, :] * m + x0y0
            dest = [(xy2[0] - radius, xy2[1] - radius),
                    (xy2[0] + radius, xy2[1] + radius)] 
            
            rect = [(xy1[0], xy1[1]), (xy2[0], xy2[1])]
            
            draw.ellipse(orig, fill = "gray", outline = "gray")
            draw.line(rect,    fill = "#007f00",  width = 1)
            draw.ellipse(dest, fill = "white", outline = "white")
        
            
        del draw
        im.save(path)
        
