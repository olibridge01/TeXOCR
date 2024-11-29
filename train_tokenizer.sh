# Train a BPE tokenizer from LaTeX equation data
# python tokenizer/tokenizer.py -v [vocab_size] -t -d [data_path] -s [save_path] --special [special_tokens] --verbose
echo "<<Training tokenizer...>>"
python tokenizer/tokenizer.py \
    -v 1000 \
    -t \
    -d data/master_labels.txt \
    -s tokenizer/tok_test.txt \
    --special tokenizer/special_tokens.txt \
    --verbose