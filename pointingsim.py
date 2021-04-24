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

from harmoni_pm.transform import PlaneSampler
from harmoni_pm.optics   import OpticalModel
from harmoni_pm.common import FloatArray
from harmoni_pm.imagegen import GCUImagePlane
from harmoni_pm.zernike import ComplexZernike, ZernikeSolver

from uncertainties import ufloat

import matplotlib.pyplot as plt
from datetime import datetime


import argparse, sys
import numpy as np
from harmoni_pm.common.configuration import Configuration
from numpy import std

POINTINGSIM_DEFAULT_OUTPUT_PREFIX = datetime.now().strftime(
    "pointing_sim_%Y%m%d_%H%M%S")

class PointingSimulator(PlaneSampler):
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
                
        self.model  = OpticalModel(model_config)
        self.gcu    = GCUImagePlane(model_config)
        
        # Initialize complex Zernike solver
        self.solver = ZernikeSolver(self.gcu.point_list() / 200e-3, 3)
        
        # Get transform
        self.pointing_transform = self.model.get_pointing_transform()
        
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
        
    def reset_measures(self):
        self.model.move_to(0, 0)
        self.model.generate()
        self.measures = np.zeros([self.cols, self.rows])

    def _process_region(self, ij, xy):
        Ninv = 1. / self.oversampling ** 2
        
        self.err_xy = xy - self.pointing_transform.backward(xy)
            
        err = np.linalg.norm(self.err_xy, axis = 1)
        
        ij[:, 1] = self.rows - ij[:, 1] - 1
         
        np.add.at(self.measures, tuple(ij.transpose()), Ninv * err)
        
    def precalculate(self):
        super().precalculate()
        self.reset_measures()
    
    def plot_heatmap(self):
        axes = FloatArray.make(
            [self.xmin(), self.xmax(), self.ymin(), self.ymax()])
        
        plt.imshow(
            1e6 * self.measures.transpose(), 
            cmap   = plt.get_cmap("inferno"),
            extent = 1e3 * axes)
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        
        c = plt.colorbar()
        c.set_label("Pointing error (µm)")
        
        plt.title("Pointing error heatmap")
 
    def plot_zernike(self, ax, a, angles = False):
        axes = FloatArray.make(
            [self.xmin(), self.xmax(), self.ymin(), self.ymax()])
        
        xp = np.linspace(-1, 1, self.cols)
        yp = np.linspace(-1, 1,  self.rows)
        
        ip   = np.linspace(0, self.cols - 1, self.cols)
        jp   = np.linspace(0, self.rows - 1, self.rows)
        
        X, Y = np.meshgrid(xp, yp)
        i, j = np.meshgrid(ip, jp)
        
        P    = np.vstack([X.ravel(), Y.ravel()]).transpose()
        ij   = np.vstack([i.ravel(), j.ravel()]).transpose()
                
        inside = (P * P).sum(axis = 1) <= 1
        P    = P[inside, :]
        ij   = ij[inside].astype(int)
        ij[:, 1] = ij.shape[1] - ij[:, 1] - 1
        
        bmap = np.zeros(X.shape)

        CZ = ComplexZernike(np.array(a))

        if angles:
            quantity = 180 / np.pi * np.angle(CZ(P))
        else:
            quantity = 1e6 * np.abs(CZ(P))
            
        bmap[ij[:, 0], ij[:, 1]] = quantity
        
        im = ax.imshow(
            bmap.transpose(), 
            extent = 1e3 * axes, 
            cmap = plt.cm.get_cmap("inferno"))
        ax.axis(1e3 * axes)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        
        c = plt.colorbar(im)
        if angles:
            ax.set_title("Angle")
            c.set_label("Pointing error angle (deg)")
        else:
            ax.set_title("Magnitude")
            c.set_label("Pointing error (µm)")

    def plot_gcu_points(self, ax, quantity, unitdesc):        
        axes = FloatArray.make(
            [self.xmin(), self.xmax(), self.ymin(), self.ymax()])
        
        sc = ax.scatter(
            1e3 * self.gcu_points[:, 0], 
            1e3 * self.gcu_points[:, 1], 
            cmap   = plt.get_cmap("inferno"),
            c = quantity, 
            s = 15)
        
        ax.axis(1e3 * axes)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_facecolor('black')

        ax.axes.set_aspect('equal', 'box', anchor='NE')

        c = plt.colorbar(sc)
        c.set_label(unitdesc)
    
    def plot_points(self):
        N = self.gcu_points.shape[0]
        plt.figure()
        plt.suptitle(
            "Pointing model ({0} Zernike coefficients, {1} points)".format(
                len(self.z_a),
                N))
        
        errors = self.measures[tuple(self.gcu_indices.transpose().tolist())]
        errang = 180 / np.pi * np.arctan2(self.err_xy[0:N, 1], self.err_xy[0:N, 0])
        
        ax = plt.subplot(221)
        self.plot_gcu_points(ax, 1e6 * errors, "Distance (µm)")
        ax.set_title("$||\\Delta\\vec{x}||$ (measured)")
        
        ax = plt.subplot(222)
        self.plot_gcu_points(ax, errang, "Angle (deg)")
        ax.set_title("$\\angle\\Delta\\vec{x}$ (measured)")
        
        ax = plt.subplot(223)        
        self.plot_zernike(ax, self.z_a, False)
        ax.set_title("$||\\Delta\\vec{x}||$ (model)")
        
        ax = plt.subplot(224)        
        self.plot_zernike(ax, self.z_a, True)
        ax.set_title("$\\angle\\Delta\\vec{x}$ (model)")
        
    def plot(self):
        if self.heatmap:
            self.plot_heatmap()
        else:
            self.plot_points()
                
    def show(self):
        plt.show()
        
    def simulate_heatmap(self):
        self.reset_measures()
        delays, mean, std, exec_time = self.process()
        return (delays, mean, std, exec_time)
    
    def solve_zernike_gcu(self):
        self.z_a = self.solver.solve_for(self.err_xy[0:self.gcu.point_list().shape[0], :])
        
    def simulate_gcu(self):
        self.reset_measures()
        self.gcu_points = self.gcu.point_list()
        
        self.gcu_indices, dt = self.process_points(self.gcu_points)
    
        self.solve_zernike_gcu()
        
        return (1, dt, 0, dt)
    
    def simulate(self):
        if self.heatmap:
            return self.simulate_heatmap()
        else:
            return self.simulate_gcu()
        
    def run(self):
        delays_total = 0
        mean_total   = 0
        std_total    = 0
        time_total   = 0
        means        = []
        
        for i in range(self.N):
            print("\rSimulation: {0:5}/{1}".format(i, self.N), end = '')
            
            delays, mean, std, exec_time = self.simulate()
            
            delays_total += delays
            mean_total   += mean
            std_total    += std
            time_total   += exec_time
            
            means.append(mean)
            
            if self.do_plot:
                self.plot()
            
            # Apply pointing model and try again
            self.model.set_pointing_model(self.z_a)
            
            delays, mean, std, exec_time = self.simulate()
            
            delays_total += delays
            mean_total   += mean
            std_total    += std
            time_total   += exec_time
            
            means.append(mean)
            
            if self.do_plot:
                self.plot()
                self.show()
                print(" done. Close window to simulate again.")
            else:
                print("done.")
                
        if std_total > 0:
            mean_total /= self.N
            std_total  /= self.N
        else:
            mean_total = np.mean(means)
            std_total  = np.std(means)
            
        return (delays_total, ufloat(mean_total, std_total), time_total)
    
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
    sim.set_sampling_properties(400, 400, 2.5 * .5e-3, 2.5 * .5e-3, radius = 0.2)
    sim.print_summary()
    
    print("Running...")
    slices, rt, tt = sim.run()
    
    print("{0} slices, {1} s per slice".format(
        slices, 
        str(rt).replace("+/-", "±")))
    print("Total execution time: {0} s".format(round(tt, 2)))
    
except Exception as e:
    print("\033[1mSimulator exception: {0}\033[0m".format(e))
    print()
    traceback.print_exc()
    sys.exit(1)

