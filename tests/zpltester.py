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

from harmoni_pm.transform import TransformTester, InverseTransform
from harmoni_pm.optics.zpl_report_parser import ZplReportParser

def runTest(transf):
    transf_name = type(transf).__name__
    print("Running tester on {0}...".format(transf_name))
    tester = TransformTester(transf)
    tester.type = "determinant"
    tester.generate_points(200, 200, .5, .5)
    #tester.generate_stars(85, -1, 100, 100, 6)
    tester.sample()
    
    print("  Applying forward...")
    tester.forward_jacobian()
    print("  Saving...")
    # tester.save_to_image(transf_name + "-distorted.png")
    tester.save_to_image(transf_name + "-determinant.png", "determinant")
    tester.save_to_image(transf_name + "-off-x.png", "offset-x")
    tester.save_to_image(transf_name + "-off-y.png", "offset-y")
    tester.save_to_image(transf_name + "-offset.png", "offset")
    tester.save_to_image(transf_name + "-curl.png", "curl")
    tester.save_to_image(transf_name + "-divergence.png", "divergence")
    tester.save_to_image(transf_name + "-rotation.png", "rotation")
    print("  Applying backward..")
    tester.backfeed()
    tester.backward_jacobian()
    
    print("  Saving...")
    tester.save_to_image(transf_name + "-restored.png")
    
    print("  Distortion RMS after undo: {0}".format(tester.distortion_rms()))


report = ZplReportParser("FPRS_distortion_map.txt")
report.parse()
t = report.make_transform()
runTest(t)
runTest(InverseTransform(t))
