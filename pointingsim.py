#!/usr/bin/env python3
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
from harmoni_pm.calibration import CalibrationStrategyCollection
from datetime import datetime

import matplotlib.pyplot  as plt
import matplotlib.patches as pch
import seaborn            as sns
import pandas             as pd # Required only to make kdeplot work

import argparse, sys
import numpy as np
from harmoni_pm.common.configuration import Configuration

POINTINGSIM_DEFAULT_OUTPUT_PREFIX = datetime.now().strftime(
    "pointing_sim_%Y%m%d_%H%M%S")
POINTINGSIM_DEFAULT_STRATEGY      = "random"

class PointingSimulator:
    def _extract_config(self):
        self.path            = self.config["config.file"]
        self.N               = self.config["simulation.N"]
        self.J               = self.config["simulation.J"]
        self.strategy        = self.config["simulation.strategy"]
        self.gap             = self.config["simulation.gap"]
        self.type            = self.config["simulation.type"]
        self.point_count     = self.config["simulation.points"]
        self.optimize        = self.config["simulation.optimize"]
        self.exponent        = self.config["simulation.exponent"]
        self.randstart       = self.config["simulation.randstart"]
        self.gcu_x_off       = self.config["simulation.gcu_x_off"]
        self.gcu_y_off       = self.config["simulation.gcu_y_off"]
    
        self.do_plot         = self.config["artifacts.plot"]
        self.markers         = self.config["artifacts.markers"]
        self.kde             = self.config["artifacts.kde"]
        self.save_files      = self.config["artifacts.save_files"]
        
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
            except SyntaxError:
                model_config.set(tweak[0], tweak[1])
        
        self.calibration = Calibration(model_config, J = self.J, gap = self.gap)

    def file_suffix(self):
        return ("_N" + str(self.N) 
                + "J" + str(self.J) 
                + "C" + str(self.point_count) 
                + "_" + self.strategy
                + ("_O" if self.optimize else ""))
        
    def showplot(self, plot, name):
        if (self.save_files):
            filename = name + self.file_suffix() + ".png"
            plot.savefig(filename)
            print("Figure saved to " + filename)
        else:
            plt.show()
            
    def dataproduct(self, name, array):
        if (self.save_files):
            filename = name + self.file_suffix() + ".dat"
            np.save(filename, array)
            print(
                "Data product saved to " 
                + filename 
                + " (shape: " + 'x'.join(list(map(str, array.shape))) + ")")
            
    def print_summary(self):
        print("PointingSim: the pointing error simulator")
        print("  Model configuration file: {0}".format(self.config["config.file"]))
        
        if len(self.tweaks) > 0:
            print("  Model overrides:")
            for t in self.tweaks:
                print("    {0} = {1}".format(t[0], t[1]))
        
        print("  Test type:            {0}".format(self.type))    
        print("  Nr. of simulations:   {0}".format(self.N))
        print("  Nr. of polynomials:   {0}".format(self.J))
        print("  Calibration strategy: {0}".format(self.strategy))
        print("  GCU point count:      {0}".format(self.point_count))
        print("  Bearing gap:          {0} mm".format(self.gap * 1e3))
        print("  Plot results:         {0}".format("yes" if self.do_plot else "no"))
        print("  Show markers:         {0}".format("yes" if self.markers else "no"))
        print("  ")


    def plot_correlations(self, coef, isImag = False):
        J = coef.shape[1]
        fig, ax = plt.subplots(J, J, figsize=(22.86, 20))
        plt.tight_layout()
        
        for i in range(J):
            Ci = np.imag(coef[:, i]) if isImag else np.real(coef[:, i])
            for j in range(J):
                Cj = np.imag(coef[:, j]) if isImag else np.real(coef[:, j])

                a = ax[j, i]
                
                if i == 0: # Column is 0: set ylabel
                    m, n = ComplexZernike.j_to_mn(j)
                    a.set_ylabel("$a^{{{0}}}_{{{1}}}$".format(m, n))

                if j == 0: # Row is 0: set title
                    m, n = ComplexZernike.j_to_mn(i)
                    a.set_title("$a^{{{0}}}_{{{1}}}$".format(m, n))
                    
                if i == j:
                    a.hist(Ci, 100, density = True)
                else:
                    a.scatter(Ci, Cj, .125, alpha = .15)
                    
    def print_correlations(self, coef, isImag = False):
        if isImag:
            print("Pearson correlation coefficient (Imaginary part)")
        else:
            print("Pearson correlation coefficient (Real part)")
        
        J = coef.shape[1]
        print("          ", end = '')
        
        for j in range(J):
            m, n = ComplexZernike.j_to_mn(j)
            print("a({0:2},{1:2})|".format(m, n), end = '')
        print('')

        print("         +", end = '')
        
        for j in range(J):
            m, n = ComplexZernike.j_to_mn(j)
            print("--------+".format(m, n), end = '')
        print('')
        
        for j in range(J):
            m, n = ComplexZernike.j_to_mn(j)
            Cj = np.imag(coef[:, j]) if isImag else np.real(coef[:, j])
            print("a({0:2},{1:2}) |".format(m, n), end = '')
            
            for i in range(J):
                Ci = np.imag(coef[:, i]) if isImag else np.real(coef[:, i])
                R = np.corrcoef(Ci, Cj)
                if i == j:
                    print("        |", end = '')
                else:
                    print("{:+.5f}|".format(R[0, 1]), end = '')
            print("")
        print("         +", end = '')
        
        for j in range(J):
            m, n = ComplexZernike.j_to_mn(j)
            print("--------+".format(m, n), end = '')
        print('')
        print("")
        
    def run_prior_xcorr(self):
        coefs = self.calibration.sample_pointing_model(self.N)

        self.print_correlations(coefs, False)
        self.print_correlations(coefs, True)
           
        if self.do_plot:
            self.plot_correlations(coefs, False)
            if self.save_files:
                self.showplot(plt, "xcorr_real")
            self.plot_correlations(coefs, True)
            self.showplot(plt, "xcorr_imag")

            
                
            
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
                fig, ax = plt.subplots(2, 2, figsize = (10, 10))
                
                self.plot_zernike_poly(j = j, ax = ax[0, 0])
                
                ax[1, 0].hist(np.real(coefs[:, j]), 100, density = True)
                ax[1, 1].hist(np.imag(coefs[:, j]), 100, density = True)
                
                if self.kde:
                    as2d = FloatArray.make([
                        np.real(coefs[:, j]),
                        np.imag(coefs[:, j])]).transpose()
                    
                    ax[0, 1].set_title("Kernel density estimation for $a^{{{0}}}_{{{1}}}$".format(m, n))
                    data = pd.DataFrame(
                        data = as2d, 
                        index = np.linspace(0, self.N - 1, self.N),
                        columns = ["re", "im"])
                    
                    sns.kdeplot(
                        data    = data,
                        x       = "re",
                        y       = "im",
                        ax      = ax[0, 1], 
                        fill    = True,
                        thresh  = 0,
                        levels  = 100,
                        cmap    = "inferno")
                else:
                    ax[0, 1].set_title("Scatter plot for $a^{{{0}}}_{{{1}}}$".format(m, n))
                    ax[0, 1].scatter(
                        np.real(coefs[:, j]),
                        np.imag(coefs[:, j]),
                        1)
                    
                # ax[2].hist2d(
                #    np.real(coefs[:, j]), 
                #    np.imag(coefs[:, j]), 
                #    50, 
                #    cmap   = plt.get_cmap("inferno"),
                #    density = True)
                
                fig.suptitle("Histograms for $a^{{{0}}}_{{{1}}}$".format(m, n))
                ax[1, 0].set_title("$Re(a^{{{0}}}_{{{1}}})$".format(m, n))
                ax[1, 0].grid(True)
                
                ax[1, 1].set_title("$Im(a^{{{0}}}_{{{1}}})$".format(m, n))
                ax[1, 1].grid(True)
            
                ax[0, 1].set_xlabel("$Re(a^{{{0}}}_{{{1}}})$".format(m, n))
                ax[0, 1].set_ylabel("$Im(a^{{{0}}}_{{{1}}})$".format(m, n))
                
                if self.do_plot and self.save_files:    
                    self.showplot(plt, "prior_a{0}{1}".format(m, n))
                    
        if self.do_plot and not self.save_files:    
            self.showplot(plt, "prior")
        
    def run_calibration_tests(self):
        numpoints = self.calibration.get_gcu_points().shape[0]
        
        if self.point_count < numpoints:
            numpoints = self.point_count
            
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
                points = self.calibration.generate_points(n + 1, self.strategy)
                params = self.calibration.calibrate(points)
                mean_error[i, n] = np.sqrt(self.calibration.get_mse(params))
                
        mean_mse = np.mean(mean_error, axis = 0)
        std_mse  = np.std(mean_error, axis = 0)
    
        
        self.dataproduct("cal_err", mean_error)
        
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
            
            if self.markers:
                plt.plot(
                    [self.J, self.J], 
                    [np.min(mean_error, axis = 0), np.max(mean_error, axis = 0)], 
                    color = 'gray', 
                    linewidth = 1)
            else:
                plt.xlim([self.J, numpoints])
                plt.ylim([1e-6, 1e-4])
                
            plt.grid(True)
            plt.xlabel("Calibration points")
            plt.ylabel("$||E||_2$")
            plt.title(
                "Error in the GCU points (J = {0}, strategy = {1})".format(
                self.J,
                self.strategy))
            
            self.showplot(plt, "calplot")
        
    def plot_heatmap(
        self,
        desc = "Pointing error (µm)",
        mul = 1e6):
        fig, ax = plt.subplots(1, 2, figsize = (16, 6))
        
        axes = FloatArray.make(self.calibration.get_axes())
        
        im = ax[0].imshow(
            mul * np.sqrt(self.calibration.get_error_map().transpose()), 
            cmap   = plt.get_cmap("inferno"),
            extent = 1e3 * axes)
        c = plt.colorbar(im, ax = ax[0])
        c.set_label(desc)
        
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
        
        valid = self.calibration.get_error_map().flatten()
        valid = valid[valid > 0]
        x     = mul * np.sqrt(valid)

        min_x = np.min(x)
        max_x = np.max(x)
        logbins = np.logspace(np.log10(min_x), np.log10(max(130e-6 * mul, max_x)), 100)

        ax[1].hist(x, bins = logbins)
        ax[1].set_xlabel(desc)
        ax[1].grid(True)
        ax[1].set_title("Error histogram")
        ax[1].plot([13e-6 * mul, 13e-6 * mul], ax[1].get_ylim(), color = 'red', label = 'Calibration goal (13 µm)')
        ax[1].set_xscale('log')
        ax[1].legend()
        return ax

    def plot_zernike_poly(self, j = 0, ax = None):
        if ax is None:
            fig, ax = plt.figure()
        
        m, n = ComplexZernike.j_to_mn(j)
        xy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
                
        cc = np.zeros(j + 1)
        cc[j] = 1
        Z     = ComplexZernike(cc)
        
        xx    = xy[0].flatten()
        yy    = xy[1].flatten()
        
        xyl   = FloatArray.make([xx, yy]).transpose()
        field = Z(xyl) / (2 * xy[0].shape[0])
        
        valid = xx ** 2 + yy ** 2 <= 1
         
        limit = pch.Arc(
                (0, 0), 
                2, 
                2,
                edgecolor = 'black',
                linestyle = '-',
                linewidth = 1)
        ax.add_patch(limit)
            
        ax.quiver(
            xx[valid], 
            yy[valid], 
            np.real(field[valid]), 
            np.imag(field[valid]),
            angles = 'xy',
            color = 'blue')
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        
        ax.set_title("Zernike polynomial for $a^{{{0}}}_{{{1}}}$".format(m, n))


    def run_show_error_map(self):
        self.calibration.test_model(None, True)
        self.plot_heatmap()
        plt.suptitle("Error heatmap (without model)")
        self.showplot(plt, "uncorrected")

    def run_show_gcu_misaligned_cal_error_map(self):
        self.calibration.test_model(None, True, offset = [self.gcu_x_off, self.gcu_y_off])
        self.plot_heatmap("Correction miss (nm)", mul=1e9)
        plt.suptitle(
            "Extrapolation error due to GCU misalignment ($\\Delta x$ = {0:g} µm, $\\Delta y$ = {1:g} µm)".format(
                self.gcu_x_off * 1e6,
                self.gcu_y_off * 1e6))
        self.showplot(plt, "uncorrected")
        
    def run_single_calibration_test(self):
        points = self.calibration.generate_points(
            int(self.point_count), 
            self.strategy)
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
                t, alpha, xy, t_p, p = self.calibration.get_calibration_path(
                    points,
                    optimize = self.optimize,
                    exponent = self.exponent)
                ax[0].plot(
                    1e3 * xy[:, 0], 
                    1e3 * xy[:, 1], 
                    linewidth = 1, 
                    color = 'white', 
                    alpha = .75)
            
                ax[0].scatter(
                    1e3 * points[:, 0], 
                    1e3 * points[:, 1], 
                    20, 
                    marker = '+', 
                    c = '#00ff00',
                    linewidth = 1)
            self.showplot(plt, "calibrated")
        
    def run_calibration_time_test(self):
        points = self.calibration.generate_points(
            int(self.point_count), 
            self.strategy)
        print("Info: computing path for {0} calibration points".format(points.shape[0]))
        t, alpha, xy, t_p, p = self.calibration.get_calibration_path(
            points,
            optimize = self.optimize,
            exponent = self.exponent,
            random   = self.randstart)
        print("done.")
        
        print(
            "Total calibration time: {0:.2f} s (~ {1:2.1f} min)".format(
                np.max(t), 
                np.max(t) / 60))
        
        if self.do_plot:
            fig, ax = plt.subplots(1, 2, figsize = (16, 7.5))
        
            axes = FloatArray.make(self.calibration.get_axes())

            bearing = pch.Arc(
                (0, 0), 
                2e3 * self.calibration.model.R(), 
                2e3 * self.calibration.model.R(),
                edgecolor = 'black',
                linestyle = '--',
                linewidth = 2)
            ax[0].add_patch(bearing)
            
            ax[0].set_xlabel('X (mm)')
            ax[0].set_ylabel('Y (mm)')
            ax[0].set_title("NGSS bearing")
            
            ax[0].plot(1e3 * xy[:, 0], 1e3 * xy[:, 1], linewidth = 1, color = 'blue', alpha = .65)
            
            ax[1].plot(t, alpha[:, 0] / np.pi * 180, label = '$\\theta$', color = 'red')
            ax[1].plot(t, alpha[:, 1] / np.pi * 180, label = '$\phi$', color = 'blue')
            
            if self.markers:
                ax[0].scatter(
                    1e3 * p[:, 0], 
                    1e3 * p[:, 1], 
                    40, 
                    marker = '+', 
                    c = '#003f00',
                    linewidth = 2)
                             
                ax[0].scatter(
                    1e3 * p[0, 0], 
                    1e3 * p[0, 1], 
                    40, 
                    marker = 'o', 
                    c = '#ff0000',
                    linewidth = 2)
                                   
                i = 0
                for t in t_p:
                    ax[0].text(
                        1e3 * p[i, 0] + 2.5,
                        1e3 * p[i, 1] + 2.5,
                        "$p_{{{0}}}$".format(i + 1, t))
                    i += 1
                    ax[1].plot(
                        [t, t], 
                        [-180, 180], 
                        linestyle = 'dashed',
                        color = 'gray',
                        linewidth = 1)
            
            ax[1].set_xlabel('Calibration time (s)')
            ax[1].set_ylabel('Axis angle (deg)')
            ax[1].set_title('Calibration path (strategy: {0})'.format(self.strategy))
            ax[1].legend()
            
            self.showplot(plt, "caltime")
        
    def run_calibration_time_distribution_test(self):
        print(
            "Info: computing paths for {0} calibration points".format(
                self.point_count))
        
        t_list = []
        for n in range(self.N):
            points = self.calibration.generate_points(
                int(self.point_count), 
                self.strategy)
            t, alpha, xy, t_p, p = self.calibration.get_calibration_path(
                points,
                optimize = self.optimize,
                exponent = self.exponent,
                random   = self.randstart)
            t_list.append(np.max(t))
            if n % 100 == 0:
                print("{0:4}/{1} tests performed\r".format(n, self.N), end = "")
        
        mean = np.mean(t_list)
        max  = np.max(t_list)
        min  = np.min(t_list)
        
        print("Min  calibration time: {0:.2f} s (~ {1:.2f} min)".format(min, min / 60))
        print("Max  calibration time: {0:.2f} s (~ {1:.2f} min)".format(max, max / 60))
        print("Mean calibration time: {0:.2f} s (~ {1:.2f} min)".format(mean, mean / 60))
        
        if self.do_plot:
            plt.hist(np.array(t_list) / 60, 50)
            plt.xlabel('Calibration time (min)')
            plt.title('Calibration time histogram for strategy {0}'.format(self.strategy))
            plt.grid()
            self.showplot(plt, "caldist")
            
    def run(self):
        if self.type == "prior":
            self.run_calculate_prior()
        elif self.type == "xcorr":
            self.run_prior_xcorr()
        elif self.type == "calplot":
            self.run_calibration_tests()
        elif self.type == "calmap":
            self.run_single_calibration_test()
        elif self.type == "errormap":
            self.run_show_error_map()
        elif self.type == "gcumacalmap":
            self.run_show_gcu_misaligned_cal_error_map()
        elif self.type == "caltime":
            self.run_calibration_time_test()
        elif self.type == "caldist":
            self.run_calibration_time_distribution_test()
    
def config_from_cli():
    config = Configuration()
    
    parser = argparse.ArgumentParser(
        description = "Simulate HARMONI's pointing errors")
    
    parser.add_argument(
        "-c",
        dest = "config_file",
        default = "harmoni.ini",
        help = "set the location of the pointing model description")
    
    parser.add_argument(
        "-s",
        dest = "cli_tweaks",
        default = [],
        action = "append",
        nargs = '*',
        help = "override an entry in the configuration file")
    
    parser.add_argument(
        "-J",
        dest = "J",
        default = 3,
        type = int,
        help = "set the number of Zernike polynomials in the pointing model")
    
    parser.add_argument(
        "-C",
        dest = "calibration_points",
        default = 553,
        type = int,
        help = "set the number of calibration points")
    
    parser.add_argument(
        "-t",
        dest = "test_type",
        default = "prior",
        help = "Sets the test type (pass `list' to print a list of available tests)")
    
    parser.add_argument(
        "-S",
        dest = "strategy",
        default = POINTINGSIM_DEFAULT_STRATEGY,
        help = "Sets the calibration strategy (pass `list' to print a list of available strategies)")
    
    parser.add_argument(
        "-p",
        dest = "pattern",
        default = POINTINGSIM_DEFAULT_OUTPUT_PREFIX,
        help = "set file name pattern for output files (may include dirs)")
    
    parser.add_argument(
        "-o",
        dest = "optimize",
        default = False,
        action = 'store_true',
        help = "optimize calibration path")
    
    parser.add_argument(
        "-k",
        dest = "kde",
        default = False,
        action = 'store_true',
        help = "compute kernel density estimation (prior test only)")
    
    parser.add_argument(
        "-r",
        dest = "randstart",
        default = False,
        action = 'store_true',
        help = "use random starting point for calibration optimization")
    
    parser.add_argument(
        "-e",
        dest = "exponent",
        default = np.inf,
        type = float,
        help = "norm of the distance used for path calibration")
    
    parser.add_argument(
        "-N",
        dest = "N",
        type = int,
        default = 1000,
        help = "set the number of simulations to run")
    
    parser.add_argument(
        "-P",
        dest = "plot",
        default = False,
        action = 'store_true',
        help = "plot simulations using Matplotlib")

    parser.add_argument(
        "-m",
        dest = "markers",
        default = False,
        action = 'store_true',
        help = "show GCU points used for calibration")

    parser.add_argument(
        "-w",
        dest = "write",
        default = False,
        action = 'store_true',
        help = "save plots to files instead of opening windows")

    parser.add_argument(
        "-g",
        dest = "gap",
        type = QuantityType("mm"),
        default = QuantityType("mm", 10.0),
        help = "set the bearing gap width")

    parser.add_argument(
        "-x",
        dest = "gcu_x_off",
        type = QuantityType("um"),
        default = QuantityType("um", 100.0),
        help = "set the X offset for GCU displacement sensitivity")

    parser.add_argument(
        "-y",
        dest = "gcu_y_off",
        type = QuantityType("um"),
        default = QuantityType("um", 0.0),
        help = "set the Y offset for GCU displacement sensitivity")

    args = parser.parse_args()
    
    if args.test_type == "list":
        print("errormap: Shows the uncorrected error map")
        print("prior:    Samples a prior for the pointing model coefficients")
        print("calplot:  Computes the error curve for different calibration point counts")
        print("calmap:   Computes an error heatmap for a given calibration strategy")
        print("caltime:  Plots the calibration path over time for a given strategy")
        print("caldist:  Calculate the time distribution of a given calibration strategy")
        print("xcorr:    Calculate cross correlations between coefficients")
        
        sys.exit(0)
        
    if args.strategy == "list":
        first = True
        for s in CalibrationStrategyCollection().get_strategies():
            if not first:
                print(", ", end = "")
            print(s, end = "")
            first = False
        print("")
        sys.exit(0)
            
    config["config.file"]          = args.config_file
    config["config.tweaks"]        = args.cli_tweaks
    config["simulation.N"]         = args.N
    config["simulation.J"]         = args.J
    config["simulation.strategy"]  = args.strategy
    config["simulation.gap"]       = args.gap["meters"]
    config["simulation.type"]      = args.test_type
    config["simulation.points"]    = args.calibration_points
    config["simulation.optimize"]  = args.optimize
    config["simulation.exponent"]  = args.exponent
    config["simulation.randstart"] = args.randstart
    config["simulation.gcu_x_off"] = args.gcu_x_off["meters"]
    config["simulation.gcu_y_off"] = args.gcu_y_off["meters"]
    config["artifacts.plot"]       = args.plot
    config["artifacts.markers"]    = args.markers
    config["artifacts.kde"]        = args.kde
    config["artifacts.save_files"] = args.write
    
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
    print("\033[1;30m")
    traceback.print_exc()
    print("\033[0m")
    sys.exit(1)

