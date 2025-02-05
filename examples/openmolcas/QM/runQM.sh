#!/bin/sh

export OMP_NUM_THREADS=28

# Change to the path of the opencap-md
export SCRIPT_PATH=/path_TO_opencapmd/sharc-interface/SHARC_MOLCAS_OPENCAP.py

chmod +rwx "$SCRIPT_PATH"
cd QM || exit

sed -i 's/SOC/H/g' QM.in

"$SCRIPT_PATH" QM.in >> QM.log 2>> QM.err

exit $?
