#!/bin/sh
PYTHONPATH=src torchrun --nproc_per_node=16 -m foundation_spatial_tumor_dynamics.cli.train
