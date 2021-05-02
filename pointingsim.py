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

from harmoni_pm.calibration import Calibration
from harmoni_pm.zernike import ComplexZernike

from datetime import datetime

import matplotlib.pyplot as plt
import argparse, sys
import numpy as np
from harmoni_pm.common.configuration import Configuration

POINTINGSIM_DEFAULT_OUTPUT_PREFIX = datetime.now().strftime(
    "pointing_sim_%Y%m%d_%H%M%S")

class PointingSimulator:
    def _extract_config(self):
        self.path            = self.config["config.file"]
        self.N               = self.config["simulation.N"]
        self.heatmap         = self.config["simulation.heatmap"]
        self.save_statistics = self.config["artifacts.save-statistics"]
        self.save_sim        = self.config["artifacts.save-sim"]
        self.do_plot         = self.config["artifacts.plot"]
        
        # Parse config tweaks
        self.tweaks = []
        for i in self.config["config.tweaks"]:
            tweak = i[0].split("=", 1)
            if len(tweak) == 2:
                self.tweaks.append((tweak[0], tweak[1]))
                
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self._extract_config()
        
        # Initialize model
        model_config = Configuration()
        model_config.load(self.path)
        
        for tweak in self.tweaks:
            try:
                model_config.parse(tweak[0], tweak[1])
            except SyntaxError:
                model_config.set(tweak[0], tweak[1])
        
        self.calibration = Calibration(model_config)

    def print_summary(self):
        print("PointingSim: the pointing error simulator")
        print("  Model configuration file: {0}".format(self.config["config.file"]))
        
        if len(self.tweaks) > 0:
            print("  Model overrides:")
            for t in self.tweaks:
                print("    {0} = {1}".format(t[0], t[1]))
            
        print("  Number of simulations: {0}".format(self.N))
        print(
            "  Save statistics:  {0}".format(
                "yes" if self.save_statistics else "no"))
        print(
            "  Save simulations: {0}".format(
                "yes" if self.save_sim else "no"))
        print(
            "  Plot results:     {0}".format(
                "yes" if self.do_plot else "no"))
        print("  ")

    def run_calculate_prior(self):
        coefs = self.calibration.sample_pointing_model(self.N)
        print("Prior for the pointing model (J = {0}): ".format(self.calibration.J))
        
        for j in range(coefs.shape[1]):
            m, n = ComplexZernike.j_to_mn(j)
            coefstr = "a({0:2},{1:2})".format(m, n)
            mean = np.mean(np.real(coefs[:, j]))
            std  = np.std(np.real(coefs[:, j]))
            print(
                "  Re[{0:8}] = {1:+.5e} ± {2:.1e} ({3:.2}%)".format(
                    coefstr, 
                    mean, 
                    std, 
                    std / np.abs(mean) * 100))
            
            mean = np.mean(np.imag(coefs[:, j]))
            std  = np.std(np.imag(coefs[:, j]))
            print(
                "  Im[{0:8}] = {1:+.5e} ± {2:.1e} ({3:.2}%)".format(
                    coefstr, 
                    mean, 
                    std, 
                    std / np.abs(mean) * 100))
            
            print("")
            
            if self.do_plot:
                fig, ax = plt.subplots(1, 2)
                ax[0].hist(np.real(coefs[:, j]), 50, density = True)
                ax[1].hist(np.imag(coefs[:, j]), 50, density = True)
                
                fig.suptitle("Histograms for $a^{{{0}}}_{{{1}}}$".format(m, n))
                ax[0].set_title("$Re(a^{{{0}}}_{{{1}}})$".format(m, n))
                ax[0].grid(True)
                
                ax[1].set_title("$Im(a^{{{0}}}_{{{1}}})$".format(m, n))
                ax[1].grid(True)
            
        
        if self.do_plot:    
            plt.show()
        
    def run(self):
        self.run_calculate_prior()
        
def config_from_cli():
    config = Configuration()
    
    parser = argparse.ArgumentParser(
        description = "Simulate HARMONI's pointing errors")
    
    parser.add_argument(
        "-c",
        "--config",
        dest = "config_file",
        default = "harmoni.ini",
        help = "set the location of the pointing model description")
    
    parser.add_argument(
        "-s",
        "--set",
        dest = "cli_tweaks",
        default = [],
        action = "append",
        nargs = '*',
        help = "override an entry in the configuration file")
    
    parser.add_argument(
        "-p",
        "--pattern",
        dest = "pattern",
        default = POINTINGSIM_DEFAULT_OUTPUT_PREFIX,
        help = "set file name pattern for output files (may include dirs)")
    
    parser.add_argument(
        "-N",
        "--number",
        dest = "N",
        type = int,
        default = 1000,
        help = "set the number of simulations to run")
    
    parser.add_argument(
        "-P",
        "--plot",
        dest = "plot",
        default = False,
        action = 'store_true',
        help = "plot simulations using Matplotlib")
    
    parser.add_argument(
        "-m",
        "--statistics",
        dest = "save_statistics",
        default = False,
        action = 'store_true',
        help = "save statistical heatmaps on the pointing errors")
    
    parser.add_argument(
        "-H",
        "--heatmap",
        dest = "heatmap",
        default = False,
        action = 'store_true',
        help = "compute full error heat maps on the field region")
    
    parser.add_argument(
        "-S",
        "--save-sim",
        dest = "save_sim",
        default = False,
        action = 'store_true',
        help = "save all intermediate simulations (handle with care!)")
    
    args = parser.parse_args()
    
    config["config.file"] = args.config_file
    config["config.tweaks"] = args.cli_tweaks
    config["simulation.N"] = args.N
    config["simulation.heatmap"] = args.heatmap
    
    config["artifacts.save-statistics"] = args.save_statistics
    config["artifacts.save-sim"] = args.save_sim
    config["artifacts.plot"] = args.plot
    
    return config

import traceback
########################### Program entry point ###############################
try:
    config = config_from_cli()
except Exception as e:
    print("Command line error: {0}".format(e))
    traceback.print_exc()
    sys.exit(1)
    
try:
    sim = PointingSimulator(config)
    sim.print_summary()
    
    print("Running...")
    sim.run()
    
except Exception as e:
    print("\033[1mSimulator exception: {0}\033[0m".format(e))
    print()
    traceback.print_exc()
    sys.exit(1)

