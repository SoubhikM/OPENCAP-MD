#!/bin/sh

export OMP_NUM_THREADS=16

export COLUMBUS=/usr/columbus/Col7.2.2_2023-09-08_linux64.ifc_molcas_bin/Columbus
# Change to the path of the opencap-md
export OPTIMIZER_PATH=/path_TO_opencapmd/optimizer/
export OPTIMIZER="$OPTIMIZER_PATH/optimizer_columbus.py"

chmod +rwx $OPTIMIZER

# shellcheck disable=SC2046
$OPTIMIZER `pwd` > dinitrogen.out
