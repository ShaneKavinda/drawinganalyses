# Install

```
https://github.com/cedricsueur/drawinganalyses.git
cd drawinganalyses
conda env create --file environment.yml
pip install -e .
```

Add MADE_DATA_DIR environment variable pointing to where your data are stored :

```
export MADE_DATA_DIR=/path/to/data_folder
```

Expected folder organization:

```
Data_folder
├── Molly
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── labels.csv
├── Humans
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── labels.csv
```

Expected data format for labels.csv:

```
labels.csv
├── img1,0
├── img2,1
├── img3,4
├── img4,0
```

Different scripts are provided to obtain this formatting.

# Configuration

Most of the configuration happens in the `config.py` file. Multiple examples are provided there, for the three datasets we used. The formatting of the folder should follow the guidelines provided above. Please uncomment the configuration you want to use and comment all the other ones.

# Usage

## Data formatting

Three formatting scripts are provided in the `utils` folder, one for each of the dataset we used `ArtistAnimals`, `Molly` and `Human drawings`. These scripts are designed to take as inputs the datasets as they were provided, and format them correctly according to the guidelines above. An example of usage is :

```
python ./drawinganalyses/utils/molly_data_formmating.py
```

## Training 

Once the data are formatted, you can train your model. Make sure that you use the correct version of `config.py`, depending on which dataset you want to train on. Once this is done, you can run the following command :

```
python ./drawinganalyses/scripts/run_training.py
```

## Notebooks

The principal notebook is `Interpretability`. In this notebook, we illustrate how we can use [Captum](https://captum.ai/) to have an understanding of the results of our trained networks. This notebook can be applied to any of the datasets metionned above. Just make sure to configure `config.py` accordingly. To apply the algorithms illustrated in this notebook to you whole dataset, please refer to the next section.

Other notebooks are provided to illustrate some of the concept used in this repo:
- `examples_artists` contains a short example to explore the datasets,
- `full_example_artists` contains all the different steps presented above, so you can play with it. A bit outdated however,
- `Test_pyfeats` is an ongoing work to train and analyze the results obtained using [Pyfeats](https://github.com/giakou4/pyfeats) to extract features, train a random forest and  get features importance,
- `analysis_explainability` is an ongoing work on trying to find interesting statics about the channels of the images.
- `feature_extraction.ipynb` is an example where ResNet (fine-tuned on our dataset or not) is used as feature extractor. Then, clustering is applied on these representations. Examples are provided using PCA and t-SNE as well.

## Interpretability scripts

To apply the `Captum` algorithms to a whole dataset, just run :

```
python ./drawinganalyses/scripts/run_interpretability.py
```

This may take a while (~5 to 7 hours on `CPU`). The analyzed images are stored in the folder according to the configuration.
