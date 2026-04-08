# TCASL

TCASL is a lightweight, pure Python inference engine for predicting American Sign Language (ASL) gestures using Temporal Contrast, simulating a Dynamic Vision Sensor (DVS).

## Features

* **Zero-Bloat Inference:** A strictly defined PyTorch wrapper built specifically for rapid prediction.
* **Temporal Contrast Processing:** Built-in methods to convert standard webcam video into DVS-style event frames.
* **Auto-Formatting:** Automatically center-crops and down-scales raw video arrays to the 128x128 resolution required by the network.

## Installation

You can install the latest release of TCASL from PyPI using `pip`:

```bash
pip install TCASL
```

## Usage in Python

Examples of how the library can be used can be found in [examples/](examples/). You should not edit this code unless you read the documentation thoroughly, which is located at [src/tcasl/tcasl.py](src/tcasl/tcasl.py).

## TCASL Project

You can find information about TCASL on the [GitHub page](TODO).