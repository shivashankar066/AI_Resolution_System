from django.apps import AppConfig
from configparser import ConfigParser
import json
import pickle
import os

config = ConfigParser()
config.read(os.path.join("qcdsPE", "config", "config.ini"))

# cat_cos_list = config['path']['cat_cols_list']
# with open(cat_cos_list, 'r') as f:
#     categorical_columns = json.load(f)

# label_encoders = {}
# for col in categorical_columns:
#     file_name = os.path.join('qcdsPE', 'config', 'label_encoder_{}.pkl'.format(col))
#
#     with open(file_name, 'rb') as f:
#         encoder = pickle.load(f)
#         label_encoders[col] = encoder


class QcdspeConfig(AppConfig):

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'qcdsPE'

   # categorical_columns = categorical_columns
   # label_encoders = label_encoders

    cpt_allowed_dict_path = config['path']['cpt_allowed_dict']
    with open(cpt_allowed_dict_path, 'r') as f:
        cpt_allowed_dict_json = json.load(f)
        cpt_allowed_dict = {tuple(k.split("*and*")): v for k, v in cpt_allowed_dict_json.items()}

    cpt_avg_allowed_path = config['path']['cpt_avg_allowed_path']
    with open(cpt_avg_allowed_path, 'r') as f:
        cpt_avg_allowed = json.load(f)

    payer_mapping_path = config['path']['payer_mapping_path']
    with open(payer_mapping_path, 'r') as f:
        payer_mapping = json.load(f)

    most_frequent_payer = config['data']['most_frequent_payer']

# cpt count
    cpt_count_path = config['path']['procedure_count_path']
    with open(cpt_count_path, 'r') as f:
        cpt_count_dic = json.load(f)

# Top Denied Cpt
    top_denied_cpt_dic_path=config['path']['top_denied_cpt_path']
    with open(top_denied_cpt_dic_path,'r') as f:
        top_denied_cpt=json.load(f)

# Top Denied Payer
    top_denied_payer_dic_path = config['path']['top_denied_payer_path']
    with open(top_denied_payer_dic_path, 'r') as f:
        top_denied_payer = json.load(f)

# Reimbursement Comment Count
    reimburse_comment_count_path=config['path']['reimburse_comment_count_path']
    with open(reimburse_comment_count_path,'r') as f:
        reim_comment_count=json.load(f)

    patient_sex_path=config['path']['patient_sex_path']
    with open(patient_sex_path,'r') as f:
        patient_sex=json.load(f)

    patient_marital_path = config['path']['patient_marital_path']
    with open(patient_marital_path, 'r') as f:
        patient_marital_status = json.load(f)

    categorical_cols_list_path = config['path']['categorical_cols_list_path']
    with open(categorical_cols_list_path,'r') as f:
        categorical_cols_list=json.load(f)


    cpt_diag_count_path=config['path']['cpt_diag_count_path']
    with open(cpt_diag_count_path,'r') as f:
        cpt_diagnosis_dic=json.load(f)

    average_amount_cpt_path = config['path']['average_amount_cpt_path']
    with open(average_amount_cpt_path, 'r') as f:
        cpt_avg_amount_dic = json.load(f)

# Load Model
    cat_model_path=config['path']['model_path_cat']
    with open(cat_model_path, 'rb') as f:
        cat_model = pickle.load(f)
        cat_model_fit_feature_order = cat_model.feature_names_
        cat_model.verbose = False



