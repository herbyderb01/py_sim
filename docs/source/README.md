**Table of Contents**
- [Overview](#overview)
  - [Quick Run Guide](#quick-run-guide)
    - [Simulation examples](#simulation-examples)
    - [Planning examples](#planning-examples)
- [Installing](#installing)
  - [Creating and activating a Python virtual environment](#creating-and-activating-a-python-virtual-environment)
  - [Installing the `py_sim` package](#installing-the-py_sim-package)
- [Code Compliance](#code-compliance)
  - [Compliant Tools](#compliant-tools)
    - [Pylint](#pylint)
    - [Isort](#isort)
    - [Mypy](#mypy)
  - [Other potential tools](#other-potential-tools)
    - [Pydocstyle](#pydocstyle)
    - [Black](#black)
- [Generating Code Documentation](#generating-code-documentation)
  - [Generate the Initial Documentation](#generate-the-initial-documentation)

# Overview
## Quick Run Guide
After [installing](#installing) and activating the virtual environment, several example files can be used to exercise different capabilities. Use the command
```bash
    python .\py_sim\launch\[file],
```
where *[file]* can be replaced with the options that follow. These options are categorized into simulation and planning example.

### Simulation examples
The simulation examples demonstrate the vehicle actively controlling and moving about the environment.
* `simple_sim.py`: Provides an example of using a control law within the simulation with position and state plotting occurring actively during the movement of the vehicle.
* `simple_vector_fields.py`: Provides and example of using a vector field control law with the vector field overlaid onto the live plotting of the moving vehicle.
* `vector_field_nav.py`: Provides an example with an obstacle world and range sensors in combination with a vehicle moving about the environment.
* `graph_planning.py`: Provides an example of path planning using a graph search with a topology graph, visibility graph, and Voronoi graph
* `grid_planning.py`: Provides a framework for visualization of planning using an occupancy grid. Basic graph search techniques are implemented, including breadth-first, depth-first, Dijkstra, A*, and greedy searches.

### Planning examples
The planning examples show static deliberative planning capabilities (i.e., planning a path through an environment from the start position to the end position)



C:\Users\greg\Documents\python-robot-sim\documentation\figures


# Installing
## Creating and activating a Python virtual environment
It is assumed that the python virtual environment is located at the same level as the `py_sim` folder and is named `venv`. This is not a requirement. To create the virtual environment, navigate the terminal to the parent of `py_sim` and run the following command:

```
python -m venv venv
```
You may need to replace `python` with `python3` or `python3.10`.

If successful, the `venv` should have been created. **Each time you open a terminal, you must activate the virtual environment**. From the parent folder, activate the virtual environment with the following command.

```
Windows:
venv\Scripts\activate

Ubuntu:
source venv/bin/activate
```

## Installing the `py_sim` package
The `py_sim` package is setup to be installed using the `pip` package installer. To install the `py_sim` package, activate the `venv` virtual environment, navigate to the parent directory of the `py_sim` code, and run the following.
```
pip install -e .
```
The `-e` option allows the Python code to be modified without the need to rerun the installation. The `requirements.txt` file defines all of the package dependences. The `setup.py` file defines the creation of the `py_sim` package.


# Code Compliance
An effort has been made to ensure that the code maintains a certain level of quality. The current code is compliant with respect to three different static analysis tools. Two additional tools could be used to maintain further compliance on documentation and formatting.

## Compliant Tools
The code is compliant to a large degree with three different tools:
* Pylint
* Isort
* mypy

These tools are now explained in more detail.

### Pylint
Pylint is a tool that checks for errors in Python code, tries to enforce a coding standard and looks for code smells. It can also look for certain type errors, it can recommend suggestions about how particular blocks can be refactored and can offer you details about the code's complexity.

```
python -m pylint --jobs 0 --rcfile .pylintrc py_sim/
```

The `.pylintrc` file defines several exceptions to the standard Python style-guide. The code could be improved significantly by removing many of these exceptions.

### Isort
isort is a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type. It provides a command line utility, Python library and plugins for various editors to quickly sort all your imports. It requires Python 3.6+ to run but supports formatting Python 2 code too.

```
python -m isort py_sim
```
Keep in mind that `isort` does not evaluate the code. It simply reorders the imports to be compliant.

### Mypy
Mypy is a static type checker for Python 3 and Python 2.7.

```
python -m mypy py_sim --explicit-package-bases
```
The code is nearly completely `mypy` compliant. There are a few excpetions, which can be found by looking for the "type: ignore" comments that exist in some of the code.

## Other potential tools
In addition to the above tools, two more tools are installed by default. However, the code is not compliant with respect to these tools.

### Pydocstyle
pydocstyle is a static analysis tool for checking compliance with Python docstring conventions.

```
python -m pydocstyle mav_sim book_assignments
```
We are not compliant with `pydocstyle` due to the sheer amount of documentation that needed to be created. However, with a low amount of work, `pydocstyle` would be a great addition to the static code analysis tools already in place.

### Black
Black is the uncompromising Python code formatter. By using it, you agree to cede control over minutiae of hand-formatting. In return, Black gives you speed, determinism, and freedom from pycodestyle nagging about formatting. You will save time and mental energy for more important matters.

We do not use this tool as the code is formated in a way to help understand the material and `Black` destroys some of that formatting.

```
python -m black mav_sim book_assignments
```

# Generating Code Documentation
This section details how to generate the initial code documentation as well as update the documentation.

## Generate the Initial Documentation
The following references were used in detail to create the documentation
* [Idiot's guide to documentation with Sphinx](https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/)
* [Sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html)
* [Sphinx-tutorial](https://www.sphinx-doc.org/en/master/tutorial/index.html)

After activating the virtual environment, create the docs folder
```bash
    sphinx-quickstart docs
```
Select "y" for separating directories

Update the `conf.py` and `index.rst` to create the desired interface.

Autogenerate the documentation of the code. Run the following from within the `docs` folder.
```bash
    sphinx-apidoc -o source/pysim/ ../py_sim -f -e --implicit-namespaces
```

Build the documentation. Run from within the `docs` folder.
```bash
    sphinx-build -b html source/ build/html
```
