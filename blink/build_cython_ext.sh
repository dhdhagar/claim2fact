#!/bin/bash

set -exu

pushd joint
rm blah.c*
cythonize -a -i blah.pyx
popd
