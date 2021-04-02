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

from .error_distribution import ErrorDistribution
from ..common.exceptions import InvalidTensorShapeError
from ..common.array import FloatArray
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root

import numpy as np

#
# This limits the size of the matrices held in memory. Feel free to adjust it
# to fine-tune performance.
# 
HARMONI_PM_GEN_MAX_SLICE_SIZE = 100000

class Arbitrary(ErrorDistribution):
    def __init__(self, x, p, deg = 3):
        if x.shape != p.shape:
            raise InvalidTensorShapeError(
                "X sample point vector does not match probability density function")
        
        if len(x.shape) != 1:
            raise InvalidTensorShapeError(
                "Arbitrary tensor-shaped distributions not yet supported")
        
        self.x    = FloatArray.make(x)
        self.p    = FloatArray.make(p)
        
        self.area = np.trapz(self.p, self.x)
        self.p   /= self.area
        self.peak = np.max(self.p)
        
        self.x0         = x[0]
        self.delta_x    = x[-1] - x[0]
        self.score_prop = self.area / (self.delta_x * self.peak)
        self.p_spline   = UnivariateSpline(
            self.x, 
            self.p, 
            s = 0, 
            k = deg, 
            ext = 1)
        
        mu = np.trapz(x * self.p, x)
        sigma = np.sqrt(np.trapz((x - mu) ** 2 * self.p, x))
        
        #
        # Find FWHM of this distribution. Extreme values of x are a good starting point
        # TODO: find a more appropriate technique
        #
        roots = root(
            lambda x: self.p_spline(x) - .5 * self.peak,
            np.linspace(self.x[0], self.x[-1], 10)).x
        
        delta_min = self.x[1] - self.x[0]
        delta_max = self.x[-1] - self.x[-2]
        
        roots = roots[
            (roots >= self.x[0] - delta_min) & 
            (roots <= self.x[-1] + delta_max)]
        
        hm_lo = np.min(roots)
        hm_hi = np.max(roots)
        
        self.p_fwhm = np.abs(hm_hi - hm_lo)
        
        super().__init__(mu, sigma)
    
    def fwhm(self):
        return self.p_fwhm
    
    def _generate(self, n, up_to):        
        ux  = np.random.uniform(0, 1, n)
        uy  = np.random.uniform(0, 1, n)
        
        x   = self.delta_x * ux + self.x0
        y   = self.peak * uy 
        
        result = FloatArray.make([])
        
        while len(result) < up_to:
            result = np.append(result, x[self.p_spline(x) >= y])
    
        return result[0:up_to]
    
    def generate(self, n = 1):        
        gen_size = int(n / self.score_prop)
        
        #
        # The trick is the following. Since I know the area of the rectangle
        # in which the distribution is enclosed + the area under the
        # distribution, I can make a rough estimate of the number of the
        # accepted samples. I can adjust this number so the matrix containing
        # the uniformly-distributed (x, y) pairs is not too big. Then, _generate
        # will generate candidates until the accepted sample count is reached.
        #
        # Why? Because it is fast. Other techniques (like the inverse CDF
        # transform) proved to be unstable or extremely slow / cumbersome.
        # Also, Python and its inherent slowness.
        #
        
        if gen_size > HARMONI_PM_GEN_MAX_SLICE_SIZE:
            gen_size = HARMONI_PM_GEN_MAX_SLICE_SIZE

        return FloatArray.make(self._generate(gen_size, n))
