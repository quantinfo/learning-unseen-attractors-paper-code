Learning Unseen Coexisting Attractors
=====================================

This is the code for the results and figures in our paper "Learning Unseen
Coexisting Attractors". They are written in [Python][] version 3.9.12, and
require recent versions of [NumPy][], [SciPy][], and
[matplotlib][]. If you are using a Python environment like
[Anaconda][], these are likely already installed.

  [Python]: https://www.python.org/
  [NumPy]: https://numpy.org/
  [SciPy]: https://www.scipy.org/
  [matplotlib]: https://matplotlib.org/
  [Anaconda]: https://www.anaconda.com/

Python Virtual Environment
--------------------------

If you are not using Anaconda, or want to run this code on the command
line in vanilla Python, you can create a virtual environment with the
required dependencies by running:

    python3 -m venv env
    ./env/bin/pip install -r requirements.txt

This will install the most recent version of the requirements
available to you. If you wish to use the exact versions we used, use
`requirements-exact.txt` instead.

You can then run the individual scripts, for example:

    ./env/bin/python SprottChaos-Torus-QuadraticNVAR-Figs1-5.py
