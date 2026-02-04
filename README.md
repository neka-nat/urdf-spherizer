urdf-spherizer
==============

[![PyPI version](https://badge.fury.io/py/urdf-spherizer.svg)](https://badge.fury.io/py/urdf-spherizer)


Convert URDF visual geometry into conservative sphere-based collision geometry.

Usage
-----

```bash
uvx urdf-spherizer path/to/model.urdf -o path/to/model.spheres.urdf --max-spheres 64 --margin 0.002
```

Optional visualization with rerun:

```bash
uvx urdf-spherizer path/to/model.urdf --viz
```

If your URDF uses `package://` URLs, pass package roots:

```bash
uvx urdf-spherizer path/to/model.urdf --package-dir my_robot=/path/to/my_robot
```


Results
--------

https://github.com/user-attachments/assets/494a6be0-eeb3-41ec-a807-08dbced9c3a8
