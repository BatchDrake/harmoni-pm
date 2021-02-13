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

from builtins import zip
from .array import FloatArray
from .exceptions import InvalidPrototypeError

def is_prototype(args, types):
    if len(args) != len(types):
        return False
    
    for (a, t) in zip(args, types):
        if not isinstance(a, t):
            return False
        
    return True

def get_xy(xy = None, x = None, y = None):
    if x is not None and y is not None:
        if not isinstance(x, float) or not isinstance(y, float):
            raise InvalidPrototypeError("x and y must be floats")
        
        return FloatArray.make([x, y])
    elif xy is not None:
        if isinstance(xy, tuple):
            if len(xy) != 2:
                raise InvalidPrototypeError("xy must be a tuple of exactly 2 elements")
            elif not isinstance(xy[0], float) or not isinstance(xy[1], float):
                raise InvalidPrototypeError("xy elements must be floats")
        
            return FloatArray.make(xy)
        
        elif FloatArray.compatible_with(xy): 
            if xy.shape[len(xy.shape) - 1] != 2:
                raise InvalidPrototypeError("Last dimension of xy must be of 2 elements")
            
            return xy
        
        else:
            raise InvalidPrototypeError("Invalid compound xy coordinates")
    else:
        raise InvalidPrototypeError("No coordinates were passed to method")

        