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

from harmoni_pm.common.singleton  import Singleton
from harmoni_pm.common.configuration import Configuration
from harmoni_pm.common.exceptions import  AlreadyRegisteredError
from harmoni_pm.calibration.calibration_strategy_factory import CalibrationStrategyFactory

class CalibrationStrategyCollection(metaclass = Singleton):
    def __init__(self):
        self.collection = {}
        
    def register(self, name, factory):
        if type(name) is not str:
            raise ValueError("Strategy name must be a string")
        
        if not issubclass(factory.__class__, CalibrationStrategyFactory):
            raise ValueError(
                "Strategy factory must be an object of type CalibrationStrategyFactory")
            
        if name in self.collection:
            raise AlreadyRegisteredError(
                "Strategy name {0} already registed".format(name))
            
        self.collection[name] = factory
        
    def make(self, gcu, name, config = Configuration()):
        if type(name) is not str:
            raise ValueError("Strategy name must be a string")
        
        if name not in self.collection:
            raise ValueError("Invalid strategy name {0}".format(name))
        
        return self.collection[name].make(gcu, config)
        
    def get_strategies(self):
        return self.collection.keys()
        
    def generate_points(self, gcu, name, config = Configuration(), scale = 1):
        strategy = self.make(gcu, name, config)
        
        return strategy.generate_points(scale)
    
