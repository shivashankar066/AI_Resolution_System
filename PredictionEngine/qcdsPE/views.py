import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from configparser import ConfigParser

from .apps import QcdspeConfig

config = ConfigParser()
config.read("config/config.ini")


def get_payer_allowed_value(row):

    allowed = row['Allowed']

    if not np.isnan(allowed):

        payor_allowed = allowed

    else:

        cpt = row['Procedure_Code']
        payor = row['Original_Carrier_Name']

        try:
            payor_allowed = QcdspeConfig.cpt_allowed_dict[payor, str(cpt)]
        except KeyError:
            try:
                payor_allowed = QcdspeConfig.cpt_avg_allowed[cpt]
            except KeyError:
                service_fee = row['Service_Fee']
                service_units = row['Service_Units']
                payor_allowed = service_fee * QcdspeConfig.cpt_avg_allowed['allowed_ratio_q3'] * service_units

    return payor_allowed


class PredictScore(APIView):

    def post(self, request):

        request_data = request.data

        inp = [
            34,
            'Sarah Gorial PA-C',
            'EPHDav  ',
            'LAB CHEM',
            'LAB     ',
            '48212',
            'F',
            'Blue Cross Complete (Medicaid)',
            'HAMTRAMCK',
            'MI ',
            0.0,
            0.0,
            'E11.9     ',
            '36416',
            1.0,
            6.5,
            12.8,
            0.0
        ]
        col_names = [
            'patient_age',
            'Actual_Dr_Name',
            'Place_of_Service_Abbr',
            'Proc_Category_Abbr', #*
            'Type_of_Service_Abbr',
            'patient_zip_code', # need strategy to handle new zip code category
            'patient_sex',
            'Original_Carrier_Name', #*
            'Patient_City',
            'Patient_State',
            'CoInsurance', #*
            'CoPayment', #*
            'Primary_Diagnosis_Code',
            'Procedure_Code',
            'Service_Units', #*
            'Service_Fee', #*
            'Allowed', #*
            'Deductible' #*
        ]

        X = pd.DataFrame([inp], columns=col_names)

        # Missing value treatment
        X['Allowed'] = X.apply(get_payer_allowed_value, axis=1)
        X['CoInsurance'] = X.CoInsurance.fillna(0)
        X['CoPayment'] = X.CoPayment.fillna(0)
        X['Deductible'] = X.Deductible.fillna(0)
        X['Original_Carrier_Name'] = X.Original_Carrier_Name.fillna(QcdspeConfig.most_frequent_payer)

        le = LabelEncoder()

        for col in QcdspeConfig.categorical_columns:
            le.classes_ = np.array(QcdspeConfig.label_codes[col], dtype='object')
            X[col] = le.transform(X[col])

        predictions = QcdspeConfig.model.predict(X)
        response = {'pred_score': round(predictions[0], 4)}

        return Response(response)
