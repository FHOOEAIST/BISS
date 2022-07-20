[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6821322.svg)](https://doi.org/10.5281/zenodo.6821322)

# BISS - Bio image segmentation service

This software performs the segmentation of blood vessels in MRI-Volumes of mural brains.  
A combination of a neuronal net (UNet) and various image processing steps is used.

![Image of algorithm segmentation pipeline](<./misc/algorithm.png>) 


## Getting started

**Install using Windows:**
cd biss
python setup.py build_ext --compiler=msvc install

**Install using Linux:**
pip install ./biss

## Acknowledgments

This work was done within a students' project during the "software project engineering" lecture in the third and fourth semester of the bachelor course of Medical and Bioinformatics at the University of Applied Sciences Upper Austria at the campus Hagenberg. It was carried out in cooperation with the Vienna BioCenter Core Facilities, the department for Austrian BioImaging (CMI), and the Aalen University. This article is also based upon work from the COST Action COMULIS (CA17121), supported by COST (European Cooperation in Science and Technology). 

## Contributing

**First make sure to read our [general contribution guidelines](https://fhooeaist.github.io/CONTRIBUTING.html).**
   
## Licence

Copyright (c) 2022 the original author or authors.
DO NOT ALTER OR REMOVE COPYRIGHT NOTICES.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

## Research

If you are going to use this project as part of a research paper, we would ask you to reference this project by citing
it. 

[In review]()
