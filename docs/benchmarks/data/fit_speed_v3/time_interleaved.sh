#!/bin/bash
# Usage: time_interleaved.sh MAIN_PYTHON BRANCH_PYTHON
# Median of 5 interleaved fits per shape; shapes from ../fit_speed_v2/shapes.py.
export OPENBLAS_NUM_THREADS=1
MAIN=$1; BRANCH=$2
cd "$(dirname "$0")/../fit_speed_v2"
run() { n=$1; r=$2; t=$3; m=(); b=()
  for i in 1 2 3 4 5; do
    m+=($("$MAIN" time_fit.py $n $r $t | awk '{print $3}'))
    b+=($("$BRANCH" time_fit.py $n $r $t | awk '{print $3}'))
  done
  python3 -c "
import statistics as s;m=list(map(float,'${m[*]}'.split()));b=list(map(float,'${b[*]}'.split()))
print('$n',$r,$t,'main',s.median(m),'branch',s.median(b),'ratio',round(s.median(b)/s.median(m),3),flush=True)"
}
for t in 1 4; do
 run reg_60k_x18 300 $t; run bin_40k_x8 300 $t; run bin_900_x20_lowcard 1000 $t; run reg_800_x17 300 $t
 run bin_3k_x1000_sparse 300 $t; run mc10_1800_x216 100 $t; run mc26_7k_x300 30 $t
done
