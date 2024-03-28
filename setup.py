#!/usr/bin/env python

import setuptools

VER = "0.0.1"

reqs = ["numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        ]

setuptools.setup(
    name = "meshInterp",
    version = VER,
    author = "Daniel Douglas",
    author_email = "dougl215@slac.stanford.edu",
    description = "interpolate fields on an irregular grid",
    url = "https://github.com/DanielMDouglas/meshInterp",
    packages = setuptools.find_packages(),
    install_requires = reqs,
    classifiers = ["Development Status :: 2 - Pre-Alpha",
                   "Intended Audience :: Developers",
                   "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python :: 3",
                   "Topic :: Scientific/Engineering :: Physics",
                   ],
    python_requires = ">=3.2",
)
