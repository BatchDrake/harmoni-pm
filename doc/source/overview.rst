Project summary
================
:code:`harmoni-pm` is a Python 3 project that comprises both the :code:`harmoni_pm` package and the command line tools necessary to perform the different simulation tasks over HARMONI's pointing model. The ultimate goal of this project is to validate the requirement R-HRM-153 that states that the WCS of each spaxel shall be known to within 1.0 spatial resolution element, which translates to an overall error budget 10 mas. This error budget will be splitted in three main groups:

* Any spaxel with respect to the IFU's field centre (dominated by calibration incertitudes)
* The IFU field centre itself with respect to the NGS reference pixel (this is dominated by NGSS mechanisms, the IFS derotator and other IFU mechanisms)
* The NGS reference pixel within the focal plane (this is dominated by optical effects and calibration errors).

The simulation will take into account mechanical tolerances, optical aberrations and other effects used as an input in order to quantify the uncertainty of the knowledge of the spaxel position via Monte Carlo testing. 

The primary user interface of this project will be a set of command-line applications configured either via command line arguments or configuration files that will produce different simulation products, either in the form of human readable text or as formatted data files.  The system-wide availability of the core package :code:`harmoni_pm` is also contemplated.

:code:`harmoni-pm` also aims to become the foundation of a more sophisticated simulation tool that takes other effects that were not initially taken into account in its earlier versions. In order to achieve this goal, a good software architecture is critical.


Source tree and directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The source tree is currently hosted in the LAM_'s gitlab repository at :code:`git@gitlab.lam.fr:harmoni/harmoni-sandbox/pointing-model.git`. The development branch is named :code:`develop`.

Source files are organized according directory structure that attempts to reflect the component architecture described in the next section. The current directory structure is as follows:
   
::

   ├── doc                   Documentation directory
   │   └── design            Software design documents
   └── harmoni_pm            Python package directory
       ├── common            Common API and utility classes
       ├── imagegen          Definition of image planes as intensity maps
       ├── imagesampler      Simulation of the behaviour of a CCD
       ├── optics            Optical model goes here
       ├── poasim            Pick Off Arm-related classes
       └── transform         Coordinate transform between image planes


.. _LAM: https://lam.fr
