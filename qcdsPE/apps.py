from django.apps import AppConfig
from configparser import ConfigParser
import json
import pickle
import os

config = ConfigParser()
config.read(os.path.join("qcdsPE", "config", "config.ini"))

cat_cos_list = config['path']['cat_cols_list']
with open(cat_cos_list, 'r') as f:
    categorical_columns = json.load(f)

label_encoders = {}
for col in categorical_columns:
    file_name = os.path.join('qcdsPE', 'config', 'label_encoder_{}.pkl'.format(col))

    with open(file_name, 'rb') as f:
        encoder = pickle.load(f)
        label_encoders[col] = encoder


class QcdspeConfig(AppConfig):

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'qcdsPE'

    categorical_columns = categorical_columns
    label_encoders = label_encoders

    cpt_allowed_dict_path = config['path']['cpt_allowed_dict']
    with open(cpt_allowed_dict_path, 'r') as f:
        cpt_allowed_dict_json = json.load(f)
        cpt_allowed_dict = {tuple(k.split("*and*")): v for k, v in cpt_allowed_dict_json.items()}

    cpt_avg_allowed_path = config['path']['cpt_avg_allowed_path']
    with open(cpt_avg_allowed_path, 'r') as f:
        cpt_avg_allowed = json.load(f)

    most_frequent_payer = config['data']['most_frequent_payer']

    # Load the model
    model_path = config['path']['model_path']
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        model_fit_feature_order = model.feature_names_in_.tolist()
        model.verbose = False
        print('model loaded')

