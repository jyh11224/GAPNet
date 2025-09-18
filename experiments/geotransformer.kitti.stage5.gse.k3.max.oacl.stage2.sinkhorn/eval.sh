# if [ "$2" = "test" ]; then
#     python test.py --test_epoch=$1
# fi
# python eval.py --test_epoch=$1 --method=lgr

for n in 250 500 1000 2500 5000; do
    python3 eval.py --num_corr=$n --method="ransac"
done
