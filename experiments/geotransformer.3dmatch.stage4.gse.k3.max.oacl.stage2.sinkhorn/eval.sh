# if [ "$3" = "test" ]; then
    # python3 test.py --test_epoch=$1 --benchmark=$2
# fi
# python3 eval.py --test_epoch=$1 --benchmark=$2 --method=lgr
for n in 250 500 1000 2500 5000; do
    python3 eval.py --num_corr=$n --benchmark="3DLoMatch" --method="ransac"
done
