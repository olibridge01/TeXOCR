# Generate dataset pickle files for train, test and val splits
# python data_wrangling/pickle_data.py -c [config_path] --split [split] -s [save_path]

echo "<<Generating pickle file for train split...>>"
python data_wrangling/pickle_data.py -c config/data_config.yml --split train -s data/train/trainset.pkl

echo "<<Generating pickle file for test split...>>"
python data_wrangling/pickle_data.py -c config/data_config.yml --split test -s data/test/testset.pkl

echo "<<Generating pickle file for val split...>>"
python data_wrangling/pickle_data.py -c config/data_config.yml --split val -s data/val/valset.pkl