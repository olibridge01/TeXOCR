# Generate the train, test and val splits
# python data_wrangling/split_data.py [input_file] [output_dir] -c [config_file]
echo "<<Generating dataset splits...>>"
python data_wrangling/split_data.py data/master_labels.txt data -c config/data_config.yml

# Render images
# python data_wrangling/render_data.py [input_file] [output_dir] -c [config_file]
echo "<<Rendering images for each split...>>"

echo "<<Rendering train split...>>"
python data_wrangling/render_data.py data/train -c config/data_config.yml

echo "<<Rendering test split...>>"
python data_wrangling/render_data.py data/test -c config/data_config.yml

echo "<<Rendering val split...>>"
python data_wrangling/render_data.py data/val -c config/data_config.yml