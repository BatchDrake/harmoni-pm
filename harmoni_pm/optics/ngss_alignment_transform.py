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

from ..common import Configuration
from ..common import FloatArray
from ..tolerance import GQ
from ..transform import Transform

HARMONI_NGSS_ALIGNMENT_X = "0 m"
HARMONI_NGSS_ALIGNMENT_Y = "0 m"

class NGSSAlignmentTransform(Transform):
    def _init_configuration(self):
        self.params = Configuration()
        self.params["ngss_alignment.x0"] = HARMONI_NGSS_ALIGNMENT_X
        self.params["ngss_alignment.y0"] = HARMONI_NGSS_ALIGNMENT_Y
        
    def _extract_params(self):
        self.x0 = GQ(self.params["ngss_alignment.x0"])
        self.y0 = GQ(self.params["ngss_alignment.y0"])
        self.generate("manufacture")
        
    def generate(self, event = "manufacture"):
        if event == "manufacture":
            x0 = self.x0.generate(1, "m")
            y0 = self.y0.generate(1, "m")
            self.offset = FloatArray.make((x0, y0)).transpose()
            self.generate("session")
            
    def set_params(self, params = None):
        if params is not None:
            self.params.copy_from(params)
            
        self._extract_params()
        
    def __init__(self, params = None):
        self._init_configuration()
        self.set_params(params)
        
    def _forward_matrix(self, p):
        return p + self.offset
    
    def _forward(self, p):
        return p + self.offset
    
    def _backward_matrix(self, p):
        return p - self.offset
    
    def _backward(self, p):
        return p - self.offset
