Neural Sound Field Reconstruction<!-- omit in toc -->
This repository contains the code (in PyTorch) for reconstructing 3D sound fields using deep implicit representations. Our work builds upon the framework introduced in the following paper:
ImplicitVol: Sensorless 3D Ultrasound Reconstruction with Deep Implicit Representation
[Paper] [Project Page]
We have modified and extended the original codebase to adapt it for sound field reconstruction using simulated data from a unity-based simulator. The original codebase can be found at ImplicitVol GitHub Repository.
Contents<!-- omit in toc -->

Dependencies
Getting Started with Rubik's Cube Simulated Data

Dependencies

Python (3.7), other versions should also work
PyTorch (1.12), other versions should also work
scipy
skimage
nibabel

Getting Started with Rubik's Cube Simulated Data

Ensure that the Rubik's Cube simulated data is located in the example/RubiksCubeInSphere directory.
Open the train_unity.py file and modify the DATA_DIR variable to point to the location of the Rubik's Cube data on your system:
pythonCopyDATA_DIR = 'path/to/your/RubiksCubeInSphere'

Run the train_unity.py script to start the training process:
bashCopypython train_unity.py
The script will do the following:

Sample a set of 2D slices from the Rubik's Cube volume, mimicking the acquisition of 2D ultrasound videos with known plane locations.
Train the implicit representation model using the sampled 2D slices.
Generate novel views (i.e., 2D slices sampled from the volume, perpendicular to the training slices) from the trained model.


Training logs and generated images will be saved in a timestamped directory under the logs folder. The directory name will be in the format YYYYMMDD_HHMMSS, representing the date and time when the training session started.
For example, if the training started on May 15, 2023, at 10:30:00, the logs and generated images will be saved in a directory named 20230515_103000 under the logs folder.

Feel free to explore and experiment with the code to further enhance the sound field reconstruction process.
If you have any questions or encounter any issues, please don't hesitate to reach out.
Happy sound field reconstruction!
