
python -u ./test_dqnflat.py --load_path $1 --mode greedy

python -u ./test_dqnflat.py --load_path $1 --mode epsilon --trial_num 5 --early_stop true --epsilon 0.3
python -u ./test_dqnflat.py --load_path $1 --mode epsilon --trial_num 10 --early_stop true --epsilon 0.3
python -u ./test_dqnflat.py --load_path $1 --mode epsilon --trial_num 20 --early_stop true --epsilon 0.3
python -u ./test_dqnflat.py --load_path $1 --mode epsilon --trial_num 5 --early_stop true --epsilon 0.1
python -u ./test_dqnflat.py --load_path $1 --mode epsilon --trial_num 10 --early_stop true --epsilon 0.1
python -u ./test_dqnflat.py --load_path $1 --mode epsilon --trial_num 20 --early_stop true --epsilon 0.1