#!/usr/bin/env bash
TESTPATH="/mnt/casa/CasMVSNet/dtu"
TESTLIST="/mnt/casa/CasMVSNet/lists/dtu/test.txt"
CKPT_FILE="/mnt/casa/CasMVSNet/casmvsnet.ckpt"
# Save_results_dir="/casa/CasMVSNet/outputs"
python test.py --dataset=general_eval --batch_size=1 --testpath=$TESTPATH  --testlist=$TESTLIST --loadckpt=$CKPT_FILE --interval_scale=1.06 ${@:2}
