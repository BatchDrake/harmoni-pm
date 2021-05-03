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

import argparse
from pint import UnitRegistry

Q = UnitRegistry().Quantity

class QuantityType(object):
    def __init__(self, base_units, value = 0, error = 0, tol = 0):
        if tol > 0:
            self._value = Q(value, base_units).plus_minus(tol, relative = True)
        else:
            self._value = Q(value, base_units).plus_minus(error)
        self._units = str(self._value.units)
        
    def units(self):
        return self._units
    
    def value(self, units = None):
        if units is None:
            return self._value.value.magnitude
        else:
            return self._value.to(units).value.magnitude
    
    def error(self, units = None):
        if units is None:
            return self._value.error.magnitude
        else:
            return self._value.to(units).error.magnitude
    
    def _set_value_error(self, value, error):
        self._value = Q(value, self._units).plus_minus(error)
        
    def set_value(self, value, units = None):
        error = self.error()
        if units is not None:
            quantity = Q(value, units)
            value    = quantity.to(self._units).magnitude
            
        self._set_value_error(value, error)
        
    def set_error(self, error, units = None):
        value = self.value()
        if units is not None:
            quantity = Q(error, units)
            error = quantity.to(self._units).magnitude
        
        self._set_value_error(value, error)
        
    def __call__(self, asstr):
        try:
            quantity = Q(asstr)
            value = quantity.to(self._units).magnitude
            self.set_value(value)
            return self
        except:
            raise argparse.ArgumentTypeError(
                "{0} is not a quantity that can be converted to {1}s".format(
                    asstr, 
                    self._units))
        
    def __getitem__(self, key):
        return self.value(key)
    
    def __setitem__(self, key, value):
        self.set_value(value, key)
    