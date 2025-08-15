# Data preparation
For each dataset, create a folder like the provided example. Folder name should be the dataset name.

Inside each folder, prepare 3 files

- `<dataset_name>.part1.inter`: training data file
- `<dataset_name>.part2.inter`: validation data file (used to choose hyper-parameters)
- `<dataset_name>.part3.inter`: test data file (used to report the final performance)

For each dataset, their format follows instruction from RecBole: https://recbole.io/docs/user_guide/usage/running_new_dataset.html