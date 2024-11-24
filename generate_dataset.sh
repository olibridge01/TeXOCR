# Generate the train, test and val splits
# python data_wrangling/split_data.py [input_file] [output_dir] -c [config_file]
echo "<<Generating dataset splits...>>"
python data_wrangling/split_data.py data2/master_labels.txt data2 -c config/data_config.yml

# Render images
# python data_wrangling/render_data.py [input_file] [output_dir] -c [config_file]
echo "<<Rendering images for each split...>>"
python data_wrangling/render_data.py data2/train/labels.txt data2/train -c config/data_config.yml
python data_wrangling/render_data.py data2/test/labels.txt data2/test -c config/data_config.yml
python data_wrangling/render_data.py data2/val/labels.txt data2/val -c config/data_config.yml