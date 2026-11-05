#!/bin/sh
# Full pipeline. Author: Aaryaman Singh (Reg. No. 2430010383), Manipal University Jaipur
# Usage: bash run_all.sh   (dataset must be in ./data or set DATA_DIR)
set -e
python prep.py
python diag.py
python baseline.py block
python baseline.py group
python train.py cnn block 10
python train.py cnn group 10
python train.py resnet18 block 6
python train.py resnet18 group 6
python figs.py
echo "Done. Results and figures are in the current folder"
