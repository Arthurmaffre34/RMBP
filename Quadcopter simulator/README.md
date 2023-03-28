<h1>Quadcopter simulator</h1>
This program allows to train the stabilization model of the quadcopter.
The engine of the simulator is a Unity project (image soon...). The project is compatible on Windows, Linux, MacOS (x86, Silicon) platforms

<h2>to train model</h2>
You have to use the train.py program and the Unity project at the same time, the project is not compile, and their no way to compile it (actually). This program produces a TCP bridge on <b>localhost</b> on PORT <b>5268</b> between the Unity project and the python program that trains the model. First of all you have to perform the unity project.

<h3>Unity Simulator Motor</h3>

You have to install Unity from internet, the free version is enough. Then when you are in the hub you must import this file as a project. You can upgrade the project version if it is not up to date, this should not cause any compatibility problem.

Then all you have to do is to launch the arrow, if the quadcopter starts flying around it is normal, this means that the simulator is running. The Unity project works like a TCP server, you have to launch the engine <b>before</b> the Python program to make them work. Now you have to install the python program.

<h3>Python train.py</h3>

***

<h4>for Conda user</h4>

create a python 3.9 environment to operate the machine learning libraries and install librairies (Pytorch). The name of the conda env is quadcopter_simulator

`conda create -n quadcopter_simulator python=3.9 pytorch`

<h4>for only Python user</h4>

We recommend you to use version 3.9 of python, to install the necessary libraries you can enter:

`pip3 install -y pytorch`

***

Now you can simply run the Python train program
`python3 train.py`

<br>
For more technical information on how Quadcoper Simulator works, I invite you to consult the wiki! <link>https://github.com/Arthurmaffre34/RMBP/wiki
