<h1>Quadcopter Simulator</h1>

<h3>Two different ways to train and test models</h3>
<h4>because the UI is not required to train the model on a large volume of data. It is preferable to offer two different solutions for testing and training models. One is based on a Unity project with TCP communication between the project and the Python client, and the other is based on a Python-coded engine that allows much faster execution. What's more, the Python engine is much more stable in that instructions are synchronous, unlike the Unity project. It's also more optimized for training over long periods. </h4>
<br>
<h3>Python engine</h3>
The Python script uses a standardized gym environment. It integrates with the step function with a default dt of 10-3 seconds. The values for mass, propeller distance from the center of gravity in m, maximum thrust in N/s^2 and gravity in m/s^2 are to be modified in the engine.py file.
