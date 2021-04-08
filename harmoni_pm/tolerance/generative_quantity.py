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

from ..common.quantity_type import Q, QuantityType
from ..common.exceptions import InvalidQuantityRepresentation
from .dirac_delta import DiracDelta
from .normal import Normal
from .uniform import Uniform

import re

FLOAT_POINT_RE = "(\d+([.]\d*)?([eE][+-]?\d+)?|[.]\d+([eE][+-]?\d+)?)"

def GQ(string):
    return GenerativeQuantity.make(string)

class GenerativeQuantity(QuantityType):
    def __init__(self, dist, units):
        self.dist = dist
        self.dist_type = dist.__class__
        
        value = self.dist.mu()
        error = .5 * self.dist.fwhm()
        
        super().__init__(units, value = value, error = error)
    
    def distribution(self):
        return self.dist_type.__name__.lower()
    
    def seed(self, seed):
        self.dist.seed(seed)
    
    #
    # TODO: We cannot expand this elif forever. Add a distribution registry
    # of some kind.
    #
    @staticmethod
    def name_to_distribution(dist):
        if dist == "normal" or dist == "gauss" or dist == "gaussian":
            return Normal
        elif dist == "uniform" or dist == "pp" or dist == "flat":
            return Uniform
        elif dist == "diracdelta":
            return DiracDelta
        else:
            return None
            
    @staticmethod
    def make(string):
        expr = '(?P<value>[+-]?{0}?)\s*(\+/?-\s*(?P<error>{0}?))?\s*' + \
               '(?P<units>[a-zA-Z][^(]*?)(\((?P<distribution>[^)]*?)\))?\s*$'
        
        pattern = re.compile(expr.format(FLOAT_POINT_RE), re.MULTILINE)
        
        match = pattern.match(string)
        
        if match is None:
            raise InvalidQuantityRepresentation("Invalid quantity")
        
        value = float(match.group('value'))
        error = match.group('error')
        units = match.group('units')
        dist  = match.group('distribution')

        # No distribution provided. Assume gaussian
        if dist is None:
            dist = "normal"
        else:
            dist = dist.lower()
            
        # If no error is provided, we assume a Dirac's delta    
        if error is None:
            dist = "diracdelta"
        else:
            error = float(error)
        
        D = GenerativeQuantity.name_to_distribution(dist)
        
        if D is None:
            raise InvalidQuantityRepresentation(
                "Unrecognized distribution `{0}'".format(dist))
            
        return GenerativeQuantity(D(value, error), units)
    
    def assign(self, gq):
        super().set_value(gq.value(self.units()))
        super().set_error(gq.error(self.units()))

        self.dist_type    = gq.dist_type
        self.dist         = self.dist_type(self.value(), self.error())
         
    def to_string(self):
        return "{0} +/- {1} {2} ({3})".format(
            self.value(),
            self.error(),
            self.units(),
            self.distribution())
        
    def set_string(self, string):
        self.assign(GenerativeQuantity.make(string))
        
    def set_value(self, value, units = None):
        super().set_value(value, units)
        self.dist = self.dist_type(self.value(), self.error())
    
    def set_error(self, error, units = None):
        super().set_error(error, units)
        self.dist = self.dist_type(self.value(), self.error())
            
    def generate(self, n = 1, as_units = None):
        if as_units is None:
            return self.dist.generate(n)
        else:
            return Q(self.dist.generate(n), self.units()).to(as_units)
    