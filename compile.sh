#!/bin/bash
cd src
nvcc main.cu graph.cu node.cu -o a_star
mv a_star ../
cd ..
