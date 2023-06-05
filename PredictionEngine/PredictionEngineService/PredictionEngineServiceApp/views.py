import logging
import os
import pickle
import time
import warnings
from configparser import ConfigParser
import pandas as pd
from django.conf import settings
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView

from .Integration import Integration
from .PatientData import PatientData
from .ResponseJson import ResponseJson
from .errorcode import ErrorCodes
from django.db import connection
from django.db.utils import DataError

warnings.filterwarnings("ignore")

config = ConfigParser()
config.read("config/config.ini")
model_aggregated_data = pd.read_excel(config["path"]["model_aggregated_data_path"])
epic_rules_data = pd.read_csv(config["path"]["epic_rules_data_path"])
cat_model = pickle.load(open(config['path']['cat_model_path'], "rb"))
log_dir = settings.LOG_DIR


def handle_empty_patient_data(patient_id, rule_engine_recommended_code, start_time):

    pred_scores = [0.0] * len(rule_engine_recommended_code)
    rec = {str(patient_id): {'Proc_code': list(rule_engine_recommended_code.keys()),
                             'Pred_score': pred_scores}}

    response = {
        "message": "Prediction Engine Service completed successfully.",
        "status": "Success",
        "statusCode": 200,
        "respTime": time.time() - start_time,
        "patiend_id": str(patient_id),
        "recommended_code": str(rec)
    }

    return response


def log_file(self):
    try:
        file_name = "PredictionEngine" + ".log"
        f = open(os.path.join(log_dir, file_name), "r")
        file_contents = f.read()
        f.close()
        return HttpResponse(file_contents, content_type="text/plain")
    except Exception as e:
        return HttpResponse(str(e), content_type="text/plain")


class PredictProcedure(APIView):
    """
        This View takes rule engine output and assigns payment scores to the procedure codes
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.errorObj = ErrorCodes()
        self.responseObj = ResponseJson()
        self.s_Name = "Prediction Engine Service"
        self.status = 0
        self.patient_id = "-1"
        self.recommended_code = {}
        extra = {
            "cls_name": self.__class__.__name__,
        }
        self.logger = logging.LoggerAdapter(self.logger, extra)

    def post(self, request):
        start_time = time.time()
        request_data = request.data
        # input_request_validation_obj = InputRequestValidation()
        try:
            patient_id = request_data["Patient_ID"]
            cursor = connection.cursor()
            # query to get count of records with the given patient_id
            query = """SELECT COUNT(1) FROM PM.vwGenSvcInfo WHERE Patient_ID = %s """
            try:
                cursor.execute(query, (patient_id,))
                cursor_out = cursor.fetchone()
                patient_rec_count = cursor_out[0]
            except DataError:
                self.logger.exception("Invalid patient id")
                return Response({"Invalid patient id"}, status=212)

            except Exception as e:
                self.logger.exception(str(e))
                return Response({"DB error"}, status=213)

            if patient_rec_count == 0:
                return Response({"Patient_id does not exist"}, status=210)

            else:
                rule_engine_recommended_code = request_data["Rule_Engine_Recommended_Code"]
                self.logger.info("Predict Procedure Started = " + str(patient_id))

                patient_df = PatientData().get_historical_data(patient_id)
                self.logger.info("patient Data fetched")

                if patient_df.shape[0] == 0:

                    response = handle_empty_patient_data(patient_id, rule_engine_recommended_code, start_time)
                    self.logger.info("patient Data empty. Returning all zero scores for recommended CPTs")

                    return Response(response)

                # Passing the single patient df to preprocessing data,
                # preparing data for historically followed procedure code
                aggregated_data_of_one_patient_records = PatientData().preprocessing_historical_data(patient_df)
                self.logger.info("pre processing complete")

                if aggregated_data_of_one_patient_records.shape[0] == 0:
                    response = handle_empty_patient_data(patient_id, rule_engine_recommended_code, start_time)
                    self.logger.info("patient Data empty after applying pre-processing steps."
                                     " Returning all zero scores for recommended CPTs")

                    return Response(response)

                epic_rules = epic_rules_data["Procedure_Code"].value_counts().keys().tolist()
                epic_rules = [str(code) for code in epic_rules]

                final_df = PatientData().resultant_df(aggregated_data_of_one_patient_records, model_aggregated_data,
                                                      epic_rules)
                if final_df.shape[0] == 0:

                    response = handle_empty_patient_data(patient_id, rule_engine_recommended_code, start_time)
                    self.logger.info("patient Data doesn't have original_carrier or primary diag code. "
                                     "Returning all zero scores for recommended CPTs")

                    return Response(response)

                self.logger.info("Resultant DF complete")

                ml_recommendation, patient_number = PatientData().data_preparation_model_prediction(final_df, cat_model)
                self.logger.info("ml recommendation complete")
                self.logger.info(ml_recommendation)

                if not ml_recommendation:
                    response = handle_empty_patient_data(patient_id, rule_engine_recommended_code, start_time)
                    self.logger.info("patient Data empty due to nulls in date_paid and/or original_billing_date cols."
                                     " Returning all zero scores for recommended CPTs")
                    return Response(response)

                final_recommendation = Integration().integration(rule_engine_recommended_code, ml_recommendation,
                                                                 patient_id, patient_number)
                self.logger.info("Response = " + str(final_recommendation))
                end_time = time.time()
                self.status = 200
                self.patient_id = patient_id
                self.recommended_code = final_recommendation

        except Exception as e:
            end_time = time.time()
            self.logger.exception(e)
            self.status = 210

        if self.status == 200:
            status_msg = self.errorObj.SuccessMsg
        else:
            status_msg = self.errorObj.FailureMsg

        response = self.responseObj.response_json_object(
            self.s_Name + str(self.errorObj.return_error_message(str(self.status))), end_time - start_time, self.status,
            status_msg, str(self.patient_id), str(self.recommended_code))

        return Response(response)
