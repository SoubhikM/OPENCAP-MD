#!/bin/bash -l

export OMP_NUM_THREADS=28

export MOLCAS=/projectnb/kbravgrp/soubhikm/qcsoftware/molcas_build/
# Change to the path of the opencap-md
export OPTIMIZER_PATH=/path_TO_opencapmd/optimizer/
export OPTIMIZER="$OPTIMIZER_PATH/optimizer_MOLCAS.py"

chmod +rwx $OPTIMIZER

# shellcheck disable=SC2046
$OPTIMIZER `pwd` > ethene.out
