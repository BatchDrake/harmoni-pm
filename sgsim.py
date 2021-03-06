#!/usr/bin/python3
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

import argparse, sys
import numpy as np

from datetime import datetime
from uncertainties import ufloat
from harmoni_pm.common import QuantityType, Configuration
from harmoni_pm.optics   import OpticalModel
from harmoni_pm.imagegen import GCUImagePlane
from harmoni_pm.imagesampler import ImageSampler

SGSIM_DEFAULT_OUTPUT_PREFIX = datetime.now().strftime("sgsim_%Y%m%d_%H%M%S")
SGSIM_DEFAULT_CCD_WIDTH        = 1024   # 1
SGSIM_DEFAULT_CCD_HEIGHT       = 1024   # 1
SGSIM_DEFAULT_CCD_PIXEL_WIDTH  = 14e-6  # m
SGSIM_DEFAULT_CCD_PIXEL_HEIGHT = 14e-6  # m
SGSIM_DEFAULT_CCD_FOCAL_LENGTH = 1      # m
SGSIM_DEFAULT_POA_THETA        = 0      # 1 (rad)
SGSIM_DEFAULT_POA_PHI          = 0      # 1 (rad)

class SGSimulator:
    def __init__(self, config):
        self.config = config
        
        # Load model configuration
        self.path    = self.config["config.file"]
        model_config = Configuration()
        model_config.load(self.path)
        
        # Parse config tweaks
        self.tweaks = []
        for i in self.config["config.tweaks"]:
            tweak = i[0].split("=", 1)
            if len(tweak) == 2:
                self.tweaks.append((tweak[0].strip(), tweak[1].strip()))
        
        for tweak in self.tweaks:
            try:
                model_config.parse(tweak[0], tweak[1])
            except SyntaxError:
                model_config.set(tweak[0], tweak[1])
            
        # Create model objects    
        if not config.have("source.plane"):
            raise RuntimeError("No source specified")
        elif config["source.plane"] == "gcu":
            self.plane = GCUImagePlane(model_config)
        else:
            raise RuntimeError(
                "Unsupported source image plane: \"{0}\"".format(config["source.plane"]))
    
        self.model = OpticalModel(model_config)
        
        # Configure arm
        self.model.move_to(config["poa.theta"], config["poa.phi"])
        
        
        self.sampler = ImageSampler(self.plane, self.model)
        
        self.sampler.set_parallel(config["integrator.parallel"])
        self.sampler.set_oversampling(config["integrator.oversampling"])
        
        self.sampler.set_detector_geometry(
            config["ccd.width"],
            config["ccd.height"],
            config["ccd.pixel-width"],
            config["ccd.pixel-height"],
            1. / config["ccd.focal-length"])
        
    def print_summary(self):
        print("SGSim: the secondary guiding simulator")
        print("  Model configuration file: {0}".format(self.config["config.file"]))
        print("  Pick-off arm configuration: ")
        print("    theta = {0}??".format(self.config["poa.theta"] / np.pi * 180))
        print("    phi   = {0}??".format(self.config["poa.phi"] / np.pi * 180))
        print("  CCD geometry: {0}x{1}".format(
            self.config["ccd.width"], 
            self.config["ccd.height"]))
        print("  Pixel size: {0} ??m x {1} ??m".format(
            self.config["ccd.pixel-width"] * 1e6,
            self.config["ccd.pixel-height"] * 1e6))
        print("  Plate scale: {0} \"/mm".format(
            206264.80624709636e-3 / config["ccd.focal-length"]))
        print("  Oversampling: {0}x{1} ({2})".format(
            config["integrator.oversampling"],
            config["integrator.oversampling"],
            config["integrator.oversampling"] ** 2))
        print("  Parallelize: {0}".format(
            "yes" if config["integrator.parallel"] else "no"))
        
        if config["ccd.exposition"] > 0:
            print("  Exposition time: {0} s".format(config["ccd.exposition"]))
            
        print("  Output file: {0}".format(config["artifacts.output"]))
        print("  ")
        
    def run(self):
        perf = self.sampler.integrate()
        
        dt = None
        if self.config["ccd.exposition"] > 0:
            dt = self.config["ccd.exposition"]
            
        self.sampler.save_to_file(
            self.config["artifacts.output"], 
            delta_t = dt)
            
        return (perf[0], ufloat(perf[1], perf[2]), perf[3])

def config_from_cli():
    config = Configuration()
    
    parser = argparse.ArgumentParser(
        description = "Simulate HARMONI's optics as projected in a CCD")
    
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
        "-o",
        dest = "output",
        default = None,
        help = "set the output PNG file name (default: generate from options)")
    
    parser.add_argument(
        "-W",
        dest = "width",
        type = int,
        default = 1024,
        help = "set CCD's width in pixels")
    
    parser.add_argument(
        "-H",
        dest = "height",
        type = int,
        default = 1024,
        help = "set CCD's height in pixels")
    
    parser.add_argument(
        "--px-width",
        dest = "px_width",
        type = QuantityType("meter"),
        default = QuantityType("meter", SGSIM_DEFAULT_CCD_PIXEL_WIDTH),
        help = "set pixel width")
    
    parser.add_argument(
        "--px-height",
        dest = "px_height",
        type = QuantityType("meter"),
        default = QuantityType("meter", SGSIM_DEFAULT_CCD_PIXEL_HEIGHT),
        help = "set pixel height")
    
    parser.add_argument(
        "-f",
        dest = "focal_length",
        type = QuantityType("meter"),
        default = QuantityType("meter", SGSIM_DEFAULT_CCD_FOCAL_LENGTH),
        help = "set detector's focal length")
        
    parser.add_argument(
        "-S",
        dest = "source",
        default = "gcu",
        help = "set observation source (default: gcu)")
    
    parser.add_argument(
        "-t",
        dest = "theta",
        type = QuantityType("radian"),
        default = QuantityType("radian", SGSIM_DEFAULT_POA_THETA),
        help = "pick-off arm theta angle configuration (default: 0??)")
    
    parser.add_argument(
        "-p",
        dest = "phi",
        type = QuantityType("radian"),
        default = QuantityType("radian", SGSIM_DEFAULT_POA_PHI),
        help = "pick-off arm theta angle configuration (default: 0??)")
    
    parser.add_argument(
        "-O",
        dest = "oversampling",
        type = int,
        default = 8,
        help = "set oversampling per pixel dimension (default: 8)")
    
    parser.add_argument(
        "-P",
        dest = "parallel",
        default = False,
        action = 'store_true',
        help = "enable parallelization (default: false)")
    
    parser.add_argument(
        "-e",
        dest = "exposition",
        type = QuantityType("second"),
        default = QuantityType("second", 0),
        help = "exposition time (default 0, only fluxes are represented)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = SGSIM_DEFAULT_OUTPUT_PREFIX + \
            "_" + str(args.width) + "x" + str(args.height) + ".png"
    
    config["config.file"]          = args.config_file
    config["config.tweaks"]        = args.cli_tweaks
        
    config["artifacts.output"] = args.output 
    
    config["source.plane"] = args.source
    
    config["integrator.oversampling"] = args.oversampling
    config["integrator.parallel"] = args.parallel
    
    config["ccd.width"] = args.width
    config["ccd.height"] = args.height
    config["ccd.pixel-width"] = args.px_width["meter"]
    config["ccd.pixel-height"] = args.px_height["meter"]
    config["ccd.focal-length"] = args.focal_length["meter"]
    config["ccd.exposition"] = args.exposition["second"]
    
    config["poa.theta"] = args.theta["radian"]
    config["poa.phi"]   = args.phi["radian"]
    
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
    sim = SGSimulator(config)
    sim.print_summary()
    
    print("Tracing...")
    slices, rt, tt = sim.run()
    
    print("{0} slices, {1} s per slice".format(
        slices, 
        str(rt).replace("+/-", "??")))
    print("Total execution time: {0} s".format(round(tt, 2)))
    
except Exception as e:
    print("\033[1mSimulator exception: {0}\033[0m".format(e))
    print()
    traceback.print_exc()
    sys.exit(1)
