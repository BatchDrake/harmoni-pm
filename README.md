# harmoni-pm
This projects holds the source code for **harmoni-pm**, the HARMONI's pointing model software. The optomechanical model of the instrument (along with support code) can be found uder the package directory `harmoni-pm`. Additionally, two executable Python 3 scripts are provided:

* `pointingsim.py`: the HARMONI's pointing simulator. It exploits different aspects of the model's pointing error pipeline.
* `sgsim.py`: a prototype for the secondary guiding simulator. It is actually a generic detector simulator that exploits the optomechanical model to generate simulated images as seen by the POA.

Both scripts are command-line applications. You can get a detailed list of options and switches of each script by running:

```
$ python3 pointingsim.py --help
$ python3 sgsim.py --help
```

## FAQ
**Does this project support Python 2.7?**

[No](https://www.python.org/doc/sunset-python-2/).


**What are the system requirements?**

You will need a more or less updated Unix-like operating system with Python3. GNU/Linux is a reasonable choice for most users. Additionally, you will need to install the dependencies listed in `requirements.txt`. These dependencies can be installed automatically by executing the following command:

```
$ pip3 install -r requirements.txt
```

**Can this project be used in Windows systems?**

As of today, no tests have been executed outside Unix-like systems. However, as I tried to avoid OS-specific calls, the software should work out of the box. The requirements are as for Unix-like systems: Python 3 and the dependencies inside `requirements.txt`.

If you run into some unexpected behavior, do not hesitate [to contact me](mailto:BatchDrake@gmail.com).
