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
from harmoni_pm.common import FloatArray
from harmoni_pm.common import QuantityType
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as pch

import argparse, sys
import numpy as np
from harmoni_pm.common.configuration import Configuration

POINTINGSIM_DEFAULT_OUTPUT_PREFIX = datetime.now().strftime(
    "pointing_sim_%Y%m%d_%H%M%S")

class PointingSimulator:
    def _extract_config(self):
        self.path            = self.config["config.file"]
        self.N               = self.config["simulation.N"]
        self.J               = self.config["simulation.J"]
        self.gap             = self.config["simulation.gap"]
        self.type            = self.config["simulation.type"]
        self.point_count     = self.config["simulation.points"]
        self.do_plot         = self.config["artifacts.plot"]
        self.markers         = self.config["artifacts.markers"]
        
        # Parse config tweaks
        self.tweaks = []
        for i in self.config["config.tweaks"]:
            tweak = i[0].split("=", 1)
            if len(tweak) == 2:
                self.tweaks.append((tweak[0].strip(), tweak[1].strip()))
                
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
                print(model_config[tweak[0]])
            except SyntaxError:
                print(tweak[0])
                print(model_config[tweak[0]])
                model_config.set(tweak[0], tweak[1])
                print(model_config[tweak[0]])
        
        self.calibration = Calibration(model_config, J = self.J, gap = self.gap)

    def print_summary(self):
        print("PointingSim: the pointing error simulator")
        print("  Model configuration file: {0}".format(self.config["config.file"]))
        
        if len(self.tweaks) > 0:
            print("  Model overrides:")
            for t in self.tweaks:
                print("    {0} = {1}".format(t[0], t[1]))
        
        print("  Test type:          {0}".format(self.type))    
        print("  Nr. of simulations: {0}".format(self.N))
        print("  Nr. of polynomials: {0}".format(self.J))
        print("  GCU point count:    {0}".format(self.point_count))
        print("  Bearing gap:        {0} mm".format(self.gap * 1e3))
        print("  Plot results:       {0}".format("yes" if self.do_plot else "no"))
        print("  Show markers:       {0}".format("yes" if self.markers else "no"))
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
                ax[0].hist(np.real(coefs[:, j]), 100, density = True)
                ax[1].hist(np.imag(coefs[:, j]), 100, density = True)
                
                fig.suptitle("Histograms for $a^{{{0}}}_{{{1}}}$".format(m, n))
                ax[0].set_title("$Re(a^{{{0}}}_{{{1}}})$".format(m, n))
                ax[0].grid(True)
                
                ax[1].set_title("$Im(a^{{{0}}}_{{{1}}})$".format(m, n))
                ax[1].grid(True)
            
        
        if self.do_plot:    
            plt.show()
        
    def run_calibration_tests(self):
        numpoints = self.calibration.get_gcu_points().shape[0]
        
        mean_error = np.zeros([self.N, numpoints])
        
        plist = range(0, numpoints)
        for i in range(mean_error.shape[0]):
            for n in plist:
                if n % 10 == 0:
                    print(
                        "Calibrating with {0:3} points (run {1}/{2})\r".format(
                            n, 
                            i + 1, 
                            self.N), 
                        end = '')
                points = self.calibration.get_gcu_points(n + 1)
                params = self.calibration.calibrate(points)
                mean_error[i, n] = np.sqrt(self.calibration.get_mse(params))
                
        mean_mse = np.mean(mean_error, axis = 0)
        std_mse  = np.std(mean_error, axis = 0)
        
        if self.do_plot:
            x = np.array(plist[0:]) + 1
            plt.semilogy(
                x, 
                mean_mse[0:])
            plt.fill_between(
                x, 
                mean_mse[0:] - std_mse[0:], 
                mean_mse[0:] + std_mse[0:],
                alpha = 0.5, 
                edgecolor = '#ff0000', 
                facecolor = '#ff8080',
                antialiased = True)
            
            plt.grid(True)
            plt.xlabel("Calibration points")
            plt.ylabel("$||E||_2$")
            plt.title("Error in the GCU points")
            plt.show()
        
    def plot_heatmap(self):
        fig, ax = plt.subplots(1, 2)
        
        axes = FloatArray.make(self.calibration.get_axes())
        
        im = ax[0].imshow(
            1e6 * np.sqrt(self.calibration.get_error_map().transpose()), 
            cmap   = plt.get_cmap("inferno"),
            extent = 1e3 * axes)
        c = plt.colorbar(im, ax = ax[0])
        c.set_label("Pointing error (µm)")
        
        bearing = pch.Arc(
            (0, 0), 
            2e3 * self.calibration.model.R(), 
            2e3 * self.calibration.model.R(),
            edgecolor = 'white',
            linestyle = '--',
            linewidth = 2)
        ax[0].add_patch(bearing)
        
        ax[0].set_xlabel('X (mm)')
        ax[0].set_ylabel('Y (mm)')
        ax[0].set_title("Pointing error heatmap")
        
        ax[1].hist(1e6 * np.sqrt(self.calibration.get_error_map().flatten()), 100)
        ax[1].set_xlabel("Pointing error (µm)")
        ax[1].grid(True)
        ax[1].set_title("Error histogram")
        ax[1].set_yscale("log")
        
        return ax

    def run_show_error_map(self):
        self.calibration.test_model(None, True)
        self.plot_heatmap()
        plt.suptitle("Error heatmap (without model)")
        plt.show()
        
    def run_single_calibration_test(self):
        points = self.calibration.get_gcu_points(int(self.point_count))
        print("Info: using {0} calibration points".format(points.shape[0]))
        params = self.calibration.calibrate(points)
        print("Info: Calibration complete. Sampling... ", end = "")
        self.calibration.test_model(params, True)
        
        mse = np.mean(self.calibration.err_sq)
        max_se = np.max(self.calibration.err_sq)
        min_se = np.min(self.calibration.err_sq)
        
        print("done")
        
        print("Min  E: {0:4.3g} µm".format(1e6 * np.sqrt(min_se)))
        print("Max  E: {0:4.3g} µm".format(1e6 * np.sqrt(max_se)))
        print("Mean E: {0:4.3g} µm".format(1e6 * np.sqrt(mse)))
        
        if self.do_plot:
            ax = self.plot_heatmap()
            plt.suptitle("Error after model (J = {0})".format(self.J))
            
            if self.markers:
                ax[0].scatter(
                    1e3 * points[:, 0], 
                    1e3 * points[:, 1], 
                    20, 
                    marker = '+', 
                    c = '#00ff00',
                    linewidth = 1)
            plt.show()
        
    def run(self):
        if self.type == "prior":
            self.run_calculate_prior()
        elif self.type == "calplot":
            self.run_calibration_tests()
        elif self.type == "calmap":
            self.run_single_calibration_test()
        elif self.type == "errormap":
            self.run_show_error_map()
            
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
        "-J",
        "--polynomials",
        dest = "J",
        default = 3,
        type = int,
        help = "set the number of Zernike polynomials in the pointing model")
    
    
    parser.add_argument(
        "-C",
        "--calibration-points",
        dest = "calibration_points",
        default = 553,
        type = int,
        help = "set the number of calibration points")
    
    parser.add_argument(
        "-t",
        "--test-type",
        dest = "test_type",
        default = "prior",
        help = "Sets the test type (pass `list' to get a list of available tests)")
    
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
        "--markers",
        dest = "markers",
        default = False,
        action = 'store_true',
        help = "show GCU points used for calibration")

    parser.add_argument(
        "-g",
        "--gap",
        dest = "gap",
        type = QuantityType("mm"),
        default = QuantityType("mm", 10.0),
        help = "set the bearing gap width")

    args = parser.parse_args()
    
    if args.test_type == "list":
        print("errormap: Shows the uncorrected error map")
        print("prior:    Samples a prior for the pointing model coefficients")
        print("calplot:  Computes the error curve for different calibration point counts")
        print("calmap:   Computes an error heatmap for a given calibration strategy")
        sys.exit(0)
        
    config["config.file"]       = args.config_file
    config["config.tweaks"]     = args.cli_tweaks
    config["simulation.N"]      = args.N
    config["simulation.J"]      = args.J
    config["simulation.gap"]    = args.gap["meters"]
    config["simulation.type"]   = args.test_type
    config["simulation.points"] = args.calibration_points
    
    config["artifacts.plot"]    = args.plot
    config["artifacts.markers"] = args.markers
    
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

