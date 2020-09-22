#!/bin/bash

source ~/anaconda3/bin/activate tensorflow_gpu

ipython kernel install

cd notebooks/

jupyter notebook --no-browser
