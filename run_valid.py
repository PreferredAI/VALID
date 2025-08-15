import argparse
from datetime import datetime
import os
from logging import getLogger
import pandas as pd
import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders
from recbole.utils import init_logger, init_seed, set_color, get_trainer


from core import VALID


def load_config(config_file_full_path):
    with open(config_file_full_path, 'r', encoding='utf-8') as reader:
        config_dict = yaml.safe_load(reader)
        return config_dict


def run(args):
    dataset = args.dataset
    config_dict = load_config(args.config_file)
    config_dict['gpu_id'] = str(args.device_id)

    config_file_list = ['./configs/common.yaml']
    config = Config(model=VALID,
                    dataset=dataset,
                    config_file_list=config_file_list,
                    config_dict=config_dict)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)

    # dataset splitting
    original_model = config['model']
    config['model'] = 'MacridVAE'
    train_data, valid_data, test_data = data_preparation(config, dataset)
    config['model'] = original_model
    del original_model
    if config['save_dataloaders']:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    # model loading and initialization
    model = VALID(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    saved = True
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    columns = ['Time'] + [f'v-{key}' for key in list(best_valid_result.keys())] + \
              [f't-{key}' for key in list(test_result.keys())]
    df_data = [[datetime.now().strftime('%Y%m%d-%H%M%S')] +
               list(best_valid_result.values()) + list(test_result.values())]
    df = pd.DataFrame(data=df_data, columns=columns)
    path = './outputs'
    if not os.path.isdir(path):
        os.mkdir(path)
    path = f'{path}/results'
    if not os.path.isdir(path):
        os.mkdir(path)
    output_dir = f'{path}/{args.dataset}'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_file = f'{output_dir}/{os.path.split(args.config_file)[-1]}.csv'
    df.to_csv(output_file, mode='a', index=False, header=not os.path.isfile(output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, default='gowalla', help='Dataset')
    parser.add_argument('-device', '--device_id', type=int, default=2, help='GPU id')
    parser.add_argument('-cfg', '--config_file', type=str, default=None, help='Config file')
    args = parser.parse_args()

    run(args)
