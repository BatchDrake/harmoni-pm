#!/usr/bin/python3
#
# Copyright (c) 2021 Gonzalo J. Carracedo <BatchDrake@gmail.com>
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

from ..common.exceptions import InvalidFileTypeError
from ..common.array import FloatArray
from ..transform import SampledTransform

import chardet
import numpy as np

ZPL_REPORT_PARSER_CHARDET_READAHEAD_SIZE = 256
ZPL_REPORT_PARSER_DATA_HEADER = \
  "  Grid X      Grid Y      Ray X       Ray Y       Err.        ROTATION\n"
  
class ZplReportParser():
    def _locate_header(self, fp):
        line = fp.readline()
        while line:
            if line == ZPL_REPORT_PARSER_DATA_HEADER:
                return True
            
            line = fp.readline()
            
        return False
    
    def _deserialize_table(self, fp):
        entries = []
        line = fp.readline()
        while line:
            f = line.split()
            try:
                if len(f) > 4:
                    entries.append(
                        [float(f[0]), float(f[1]), float(f[2]), float(f[3])])
                
            except ValueError:
                pass
            line = fp.readline()
            
        return FloatArray.make(entries)
    
    def __init__(self, path):
        self.path = path
        
        try:
            rawdata = open(path, "rb").read(
                ZPL_REPORT_PARSER_CHARDET_READAHEAD_SIZE)
            result = chardet.detect(rawdata)
            self.encoding = result["encoding"]
            self.fp = open(path, "r", encoding = self.encoding)
        except:
            raise
        
        if not self._locate_header(self.fp):
            raise InvalidFileTypeError("Could not find report data header")
        
    def _guess_sampling_info(self, t):
        N = t.shape[0]

        # Extract sampling properties. Please note this code is highly dependent
        # on the script output format and the order of the sampling points.
        
        delta_x = 0
        delta_y = 0
        
        max_x = min_x = t[0, 0]
        max_y = min_y = t[0, 1]
        
        for i in range(1, N):
            # Extract step. We only take backward sampling steps into account
            # (positive steps are assumed to be caused by new scan lines)
            dx = t[i - 1, 0] - t[i, 0]
            dy = t[i - 1, 1] - t[i, 1]
            
            # TODO: Do not compare floats directly. Maybe use tolerances?
            if dx > 0:
                if delta_x == 0:
                    delta_x = dx
                elif delta_x != dx:
                    raise InvalidFileTypeError(
                        "Non-uniform sampling detected (dx = {0} != {1})".format(
                            dx,
                            delta_x))
            
            if dy > 0:
                if delta_y == 0:
                    delta_y = dy
                elif delta_y != dy:
                    raise InvalidFileTypeError(
                        "Non-uniform sampling detected (dy = {0} != {1})".format(
                            dy,
                            delta_y))
            
            # Detect limits
            if t[i, 0] > max_x:
                max_x = t[i, 0]
            if t[i, 1] > max_y:
                max_y = t[i, 1]
            if t[i, 0] < min_x:
                min_x = t[i, 0]
            if t[i, 1] < min_y:
                min_y = t[i, 1]

        cols = int(np.floor((max_x - min_x) / delta_x)) + 1
        rows = int(np.floor((max_y - min_y) / delta_y)) + 1
        
        x0   = min_x
        y0   = min_y
        
        return (x0, y0, delta_x, delta_y, cols, rows)
    
    def _build_maps(self, t, x0, y0, delta_x, delta_y, cols, rows):
        e_x = np.zeros([rows, cols])
        e_y = np.zeros([rows, cols])
        
        for n in range(t.shape[0]):
            x = t[n, 0]
            y = t[n, 1]
            
            i = int(np.floor((x - x0) / delta_x))
            j = int(np.floor((y - y0) / delta_y))
            
            e_x[j, i] = t[n, 2] - t[n, 0]
            e_y[j, i] = t[n, 3] - t[n, 1]
        
        return (e_x, e_y)
        
    def parse(self):
        t = self._deserialize_table(self.fp)
        if t.shape[0] < 4:
            raise InvalidFileTypeError("Failed to deserialize distortions")
                    
        info = self._guess_sampling_info(t)
        #
        # I know this is not the best representation format for the sampling info, 
        # but since this is the internal API, I did not care too much about creating
        # a dictionary or not. Feel free to use a more readable approach if this 
        # takes more than 5 seconds to understand.
        #    
        self.x0      = info[0]
        self.y0      = info[1]
        self.delta_x = info[2]
        self.delta_y = info[3]
        self.cols    = info[4]
        self.rows    = info[5]
        
        # Done. Build maps and return.
        self.e_x, self.e_y = self._build_maps(
            t, 
            self.x0, 
            self.y0, 
            self.delta_x, 
            self.delta_y, 
            self.cols, 
            self.rows)
        
    def make_transform(self):
        return SampledTransform(
            self.x0, 
            self.y0, 
            self.delta_x, 
            self.delta_y, 
            self.e_x, 
            self.e_y)
    