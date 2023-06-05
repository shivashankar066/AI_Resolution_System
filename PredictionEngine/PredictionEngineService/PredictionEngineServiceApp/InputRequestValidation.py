import logging
import sys
import os
import datetime
from datetime import datetime, time, date
format = '%Y/%m/%d'


class InputRequestValidation:
    """
        This class defines the functions for validating Input Json from Rest API
    """

    def __init__(self):
        """
        To initiate logger variable
        """
        self.logger = logging.getLogger(__name__)
        extra = {
            "cls_name": self.__class__.__name__,
        }
        self.logger = logging.LoggerAdapter(self.logger, extra)

    def request_validation(self, request):
        """
        :param request:
        :return:
        """
        # Validating Input Request Json
        input_signature = ["Patient_id","appointment_date"]
        flag = 0
        try:
            if all(field in request for field in input_signature):
                if not (isinstance(request["patient_id"], int)):
                    self.logger.error("patient_id TypeError:Expecting Integer")
                    flag = 1
                else:
                    if request["patient_id"] <=0:
                        self.logger.error("patient_id is not in the specified range")
                        flag = 1
                if not (isinstance(request["appointment_date"], datetime)):
                    self.logger.error("appointment_date TypeError:Expecting date format")
                    flag = 1
                else:
                    datetime.strptime(request["appointment_date"], "%Y-%m-%d")
                    print(request["appointment_date"])
            else:
                flag = 1
                self.logger.error("Missing required fields in input_signature")
            if flag == 1:
                return "not valid parameters"
            else:
                return "valid"
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            f_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.logger.error(str([exc_type, f_name, exc_tb.tb_lineno]))
            self.logger.error(str(e))
            return "internal error"