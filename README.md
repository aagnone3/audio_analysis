# Audio Analysis with Python

These scripts and modules form a simple, yet somewhat scalable approach to audio analysis with Python 2. The modules are organized as follows:
- common: functionality shared by the other modules
- features: audio feature extraction
- learning: using machine learning to generalize to future audio
- recording: record audio for training and testing machine learning models
- visualization: visualize audio streams in the time and spectral domains

# Motivation

Digital signal processing theory can get dry, but its applications are numerous, immersive, and exciting!

# Installation

This package uses the [Essentia](https://github.com/MTG/essentia) C++ audio feature extraction library under the hood, which necessitates Python 2 as the driver. Ensure that you have [Essentia](https://github.com/MTG/essentia) installed before proceeding.

For basic visualization, clone and run! There is sample audio included with the project.

For machine learning applications, run the following bash commands to obtain train and test data:

```sh
cd setup
sh check_directory_structure.sh
```

Machine Learning Data Sources:
- [ELSDSR](http://www2.imm.dtu.dk/~lfen/elsdsr/)
- (Soon) [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)

# Visualization Snapshot

![Visualization Screenshot]
(https://raw.githubusercontent.com/aagnone3/audio_analysis/master/res/images/viz_screenshot.png)

# API Reference

Coming soon, as the project grows!

## Contributors

Currently me, myself, and I. I'm happy to enhance this project with others, don't hesitate to reach out!

## License

This software is released with the Apache License. Download it, use it, change it, share it. Just keep the license!
