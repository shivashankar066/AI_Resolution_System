import logging
import numpy as np
import pandas as pd
from django.db import connection

logger = logging.getLogger(__name__)


class PatientData:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sName = "Recommend Procedure Service"
        self.status = 0
        extra = {
            "cls_name": self.__class__.__name__,
        }
        self.logger = logging.LoggerAdapter(self.logger, extra)

    def get_historical_data(self, patient_id):
        """
        this function will fetch the all records
        for the given patient id
        :param int patient_id:
        :return: a dataframe
        """
        result_dict = {"response_key": "-1"}
        try:
            cursor = connection.cursor()
            cursor.execute("""
            Select t1.Service_ID,t1.Patient_ID,t1.Patient_Number,d.IMREDEM_CODE,t5.patient_age,
            t1.Actual_Dr_Name,t1.Place_of_Service_Abbr,t1.Proc_Category_Abbr,t1.Type_of_Service_Abbr,
            t5.patient_zip_code,t5.patient_sex,t1.Original_Carrier_Name, t5.Patient_City, t5.Patient_State,
            t2.CoPayment,t2.CoInsurance,t1.Primary_Diagnosis_Code,t1.Procedure_Code,t1.Service_Units,
            convert(Date,t1.Service_Date_From) as Service_Date_From, t1.Claim_Number,
            convert(Date, t1.Original_Billing_Date) as Original_Billing_Date,Convert(Date, t2.Date_Paid) as Date_Paid,
            t1.Service_Fee,t2.Amount, t2.Allowed, t2.Deductible, t2.Transaction_Type, t4.Abbreviation,
            t4.Description, t4.Self_Pay_TranCode
            from PM.vwGenSvcInfo as t1
            left join PM.[vwGenSvcPmtInfo] t2 ON t1.Service_Id=t2.Service_Id 
            left join PM.Reimbursement_Detail t3 on t1.Service_Id=t3.Service_Id
            left join [dbo].[vUAI_Transaction_Codes] t4 ON t2.Transaction_Code_Abbr=t4.Abbreviation
            left join [EMR].[HPSITE].[DEMOGRAPHICS_VIEW] as d on t1.Patient_Number = d.DEM_EXTERNALID
            left join PM.vwGenPatInfo as t5 ON T1.Patient_Number=T5.Patient_Number
            where t1.Service_Date_From > '2017-01-01' and t4.Self_Pay_TranCode=0 and  
            (t2.Transaction_Type !='A') and (t2.Transaction_Type !='T') and (t2.Transaction_Type !='B') 
            and (t1.Service_Fee > 0) and (t2.Amount >0) and 
            ((t1.Primary_Diagnosis_Code between 'E08' and 'E13') OR 
            (t1.Primary_Diagnosis_Code='R73.03')) and t1.Patient_ID= %s """, (patient_id,))

            #and t1.Original_Carrier_Name is not null

            result = pd.DataFrame([list(elem) for elem in cursor.fetchall()])

            if not result.empty:
                result.columns = ["Service_ID", "Patient_ID", "Patient_Number", "IMREDEM_CODE", "patient_age",
                                  "Actual_Dr_Name", "Place_of_Service_Abbr", "Proc_Category_Abbr",
                                  "Type_of_Service_Abbr","patient_zip_code", "patient_sex", "Original_Carrier_Name",
                                  "Patient_City", "Patient_State", "CoPayment", "CoInsurance", "Primary_Diagnosis_Code",
                                  "Procedure_Code", "Service_Units", "Service_Date_From", "Claim_Number",
                                  "Original_Billing_Date", "Date_Paid", "Service_Fee", "Amount", "Allowed", "Deductible",
                                  "Transaction_Type", "Abbreviation", "Description", "Self_Pay_TranCode"]

            # self.logger.info("result == " + str(result))
        except Exception as e:
            self.logger.info("Error at Patient History ", e)
            result = {"failed "}
        return result

    def preprocessing_historical_data(self, one_patient_data):
        """
                This function returns the one patient of pre-processed data
                :param one_patient_data:
                :return: return aggregation of preprocessed data of the patient in dataframe
        """
        one_patient_data["Description"] = one_patient_data["Description"].apply(
            lambda x: -999 if str(x).startswith("Self") else x)
        self_pay_index = one_patient_data.loc[one_patient_data["Description"] == -999].index
        one_patient_data = one_patient_data.drop(self_pay_index)
        list_of_col_drop = ["Self Pay Transfer", "Self Pay Adjustment", "Self Pay Financial Hardship"]
        one_patient_data = one_patient_data[~one_patient_data.Description.isin(list_of_col_drop)]

        # drop IMREDEM_COE and Patient_ID keep only Patient_Number
        one_patient_data = one_patient_data.drop(["IMREDEM_CODE", "Patient_ID"], axis=1)

        # Preprocessing
        one_patient_data = one_patient_data.drop_duplicates()
        one_patient_data = one_patient_data.loc[one_patient_data["Original_Billing_Date"].notna()]
        one_patient_data = one_patient_data.loc[one_patient_data["Date_Paid"].notna()]
        one_patient_data["Original_Billing_Date"] = pd.to_datetime(one_patient_data["Original_Billing_Date"])
        one_patient_data["Date_Paid"] = pd.to_datetime(one_patient_data["Date_Paid"])
        one_patient_data["Delay_in_days"] = pd.to_datetime(one_patient_data["Date_Paid"]) - pd.to_datetime(one_patient_data["Original_Billing_Date"])
        one_patient_data["Delay_in_days"] = one_patient_data["Delay_in_days"].apply(lambda x: int(str(x).split(" ")[0]))
        data_df = one_patient_data[one_patient_data['Delay_in_days'] >= 0]
        data_df1 = data_df[data_df["Service_Fee"] >= data_df["Amount"]]
        df_by_service= data_df1.groupby(["Service_ID"], as_index=False).agg({
            "Patient_Number": "first",
            'patient_age': "max",
            'Actual_Dr_Name': 'first',
            'Place_of_Service_Abbr': 'first',
            'Proc_Category_Abbr': 'first',
            'Type_of_Service_Abbr': 'first',
            'patient_zip_code': "first",
            'patient_sex': "first",
            'Original_Carrier_Name': "first",
            'Patient_City': "first",
            'Patient_State': "first",
            "Description": "first",
            'CoInsurance': "sum",
            'CoPayment': "sum",
            "Primary_Diagnosis_Code": "first",
            "Procedure_Code": "first",
            'Service_Units': "max",   #sum
            'Service_Date_From': "first",
            # "Claim_Number": "first",
            "Original_Billing_Date": "first",
            "Date_Paid": 'last',
            "Service_Fee": "max",
            "Amount": "sum",   #max
            'Allowed': 'max',
            'Delay_in_days':'max',
            'Deductible': 'max',
            "Transaction_Type": "count"
        })
        df_by_service1 = df_by_service[df_by_service.Amount <= df_by_service.Service_Fee]
        final_df=df_by_service1
        return final_df

    def patient_procedure_code_followed_not_followed(self, history_records_followed, client_procedure_code):
        """
        :param history_records_followed:
        :param client_procedure_code:
        :return:
        """
        records_followed = history_records_followed
        procedure_code_not_followed = dict()
        patient_id = records_followed["Patient_Number"].tolist()[0]
        followed_procedure_code_list = records_followed["Procedure_Code"].value_counts().keys().tolist()

        not_followed_procedure_code = []

        for proc_code in client_procedure_code:
            if proc_code not in followed_procedure_code_list:
                not_followed_procedure_code.append(proc_code)

        procedure_code_not_followed[patient_id] = not_followed_procedure_code

        patient_age = records_followed["patient_age"].iloc[0]
        patient_zip_code = records_followed["patient_zip_code"].iloc[0]
        return patient_id, patient_age, patient_zip_code, records_followed, procedure_code_not_followed

    def resultant_df(self, final_data, ml_model_aggregated_data, client_procedure_code):
        """
        :param final_data:
        :param ml_model_aggregated_data:
        :param client_procedure_code:
        :return:
        """

        patient_id, patient_age, patient_zip_code, records_followed, procedure_code_not_followed = \
            self.patient_procedure_code_followed_not_followed(final_data, client_procedure_code)

        list_of_columns_to_keep = ['Service_ID','Patient_Number', 'Original_Carrier_Name', 'Primary_Diagnosis_Code',
                                   'Procedure_Code','patient_age', 'Actual_Dr_Name', 'Place_of_Service_Abbr',
                                   'Proc_Category_Abbr', 'Type_of_Service_Abbr', 'patient_zip_code', 'patient_sex',
                                   'Patient_City', 'Patient_State', 'CoInsurance', 'CoPayment', 'Service_Units',
                                   'Service_Date_From', 'Original_Billing_Date','Date_Paid', 'Service_Fee',
                                   'Amount', 'Allowed', 'Deductible', 'Description', 'Transaction_Type']

        # reading data from the final dataset, Either can execute the query for these aggregated data are
        # of 5 years records so reading it in excel
        list_of_columns_to_keep_df = ml_model_aggregated_data[list_of_columns_to_keep]

        # drop the duplicates
        list_of_columns_to_keep_df = list_of_columns_to_keep_df.drop_duplicates()

        # Remove the white trailing spaces
        list_of_columns_to_keep_df["Primary_Diagnosis_Code"] = list_of_columns_to_keep_df[
            "Primary_Diagnosis_Code"].str.strip()

        # collect patient past payers list
        records_followed_payer_name = records_followed["Original_Carrier_Name"].value_counts().keys().tolist()

        # Collect the unique primary diagnosis code in list
        records_followed_primary_diag_name = records_followed[
            "Primary_Diagnosis_Code"].str.strip().value_counts().keys().tolist()

        # Collect the data w.r.t to the payer where one patient is registered(filtering the data w.r.t payer)
        resultant_df = pd.DataFrame()
        for rec in records_followed_payer_name:
            res = list_of_columns_to_keep_df.loc[list_of_columns_to_keep_df["Original_Carrier_Name"] == rec]
            resultant_df = resultant_df.append(res)

        if resultant_df.empty:
            return resultant_df

        # filtering the data w.r.to diagnosis code where the patient is suffered
        result_df = pd.DataFrame()
        for rec in records_followed_primary_diag_name:
            res = resultant_df.loc[resultant_df["Primary_Diagnosis_Code"] == rec]
            result_df = result_df.append(res)

        if result_df.empty:
            return result_df

        # List of procedure code where in the history not performed by the patient
        list_of_procedure_need_to_be_add = procedure_code_not_followed[patient_id]

        # dataframe creation where the procedure code not performed
        semi_final_df = pd.DataFrame()

        for proc in list_of_procedure_need_to_be_add:
            res = result_df.loc[result_df["Procedure_Code"] == proc]
            semi_final_df = semi_final_df.append(res)

        if semi_final_df.empty:
            return semi_final_df

        # aggregation required to create a uniqueness of the data records
        semi_final_df = semi_final_df.reset_index(drop=True)
        # Adding the columns where the attributes are required for ML model
        semi_final_df["Patient_Number"] = patient_id
        semi_final_df["patient_age"] = patient_age
        semi_final_df["patient_zip_code"] = patient_zip_code
        # Rearrange the columns as per the ML model input
        semi_final_df = semi_final_df.loc[:,
                        ['Patient_Number', 'Original_Carrier_Name', 'Primary_Diagnosis_Code', 'Procedure_Code',
                         'patient_age', 'Actual_Dr_Name', 'Place_of_Service_Abbr',
                         'Proc_Category_Abbr', 'Type_of_Service_Abbr', 'patient_zip_code', 'patient_sex',
                         'Patient_City', 'Patient_State', 'CoInsurance',
                         'CoPayment', 'Service_Units', 'Service_Date_From', 'Original_Billing_Date',
                         'Date_Paid', 'Service_Fee', 'Amount', 'Allowed', 'Deductible', 'Description', 'Transaction_Type']]

        # Columns need for creating input dataset to the model
        columns_needed_for_df = list_of_columns_to_keep

        records_followed = records_followed[columns_needed_for_df]

        records_followed = records_followed.reset_index(drop=True)
        semi_final_df = semi_final_df.reset_index(drop=True)
        final_df_1 = records_followed.append(semi_final_df)
        final_df_1 = final_df_1.reset_index(drop=True)
        return final_df_1

    def data_preparation_model_prediction(self, df, model):
        """
        :param df:
        :param model:
        :return:
        """
        ml_recommendation = dict()
        patient_number = -1
        try:

            df["Patient_Number"] = df["Patient_Number"].astype("category")
            df["Service_Date_From"] = pd.to_datetime(df["Service_Date_From"])
            df["Original_Billing_Date"] = pd.to_datetime(df["Original_Billing_Date"])
            df["Date_Paid"] = pd.to_datetime(df["Date_Paid"])
            patient_number = df.loc[:, "Patient_Number"][0]

            # drop columns
            df = df.drop(["Service_ID","Patient_Number", "Amount", "Transaction_Type", "Deductible",
                          "Date_Paid", "Original_Billing_Date", "Service_Date_From"], axis=1)
            df = df[df["CoInsurance"] >= 0]
            column_name = 'Original_Carrier_Name'
            df.dropna(subset=[column_name], inplace=True)
            # Fill the allowed column to be 0
            df["Allowed"] = df["Allowed"].fillna(0)
            df = df.drop(["CoInsurance", "CoPayment"], axis=1)
            # convert the data types
            df["patient_age"] = df["patient_age"].astype("int64")
            df["Actual_Dr_Name"] = df["Actual_Dr_Name"].astype("category")
            df["Place_of_Service_Abbr"] = df["Place_of_Service_Abbr"].astype("category")
            df["Proc_Category_Abbr"] = df["Proc_Category_Abbr"].astype("category")
            df["Type_of_Service_Abbr"] = df["Type_of_Service_Abbr"].astype("category")
            df["patient_zip_code"] = df["patient_zip_code"].astype("category")
            df["patient_sex"] = df["patient_sex"].astype("category")
            df["Original_Carrier_Name"] = df["Original_Carrier_Name"].astype("category")
            df["Patient_City"] = df["Patient_City"].astype("category")
            df["Patient_State"] = df["Patient_State"].astype("category")
            df["Description"] = df["Description"].astype("category")
            df["Primary_Diagnosis_Code"] = df["Primary_Diagnosis_Code"].astype("category")
            df["Procedure_Code"] = df["Procedure_Code"].astype("category")
            df["Service_Units"] = df["Service_Units"].astype("float64")
            df["Service_Fee"] = df["Service_Fee"].astype("float64")
            df["Allowed"] = df["Allowed"].astype("float64")
            # Rearrange the columns
            df = df[['patient_age', 'Actual_Dr_Name', 'Place_of_Service_Abbr',
                    'Proc_Category_Abbr', 'Type_of_Service_Abbr', 'patient_zip_code',
                     'patient_sex', 'Original_Carrier_Name', 'Patient_City', 'Patient_State',
                     'Description', 'Primary_Diagnosis_Code', 'Procedure_Code', 'Service_Units',
                     'Service_Fee', 'Allowed']]
            if df.empty:
                self.logger.info('patient data empty due to null values in date_paid and/or original_billing_date')
                return ml_recommendation, patient_number
            prediction = model.predict(df)
            prediction = np.round(prediction, 2)
            df = df.reset_index(drop=True)
            df["Predicted_Score"] = pd.Series(prediction, name="Predicted_Score")
            df = df.sort_values(["Predicted_Score"], ascending=False)
            result = df.loc[:, ["Procedure_Code", "Predicted_Score"]]
            result = result.loc[result["Predicted_Score"] >= 0]
            self.logger.info(result)

            res_dict = dict()
            for col in result.columns:
                res_dict[col] = result[col].values.tolist()
            ml_recommendation[patient_number] = res_dict

        except Exception as e:
            self.logger.exception(str(e))
        return ml_recommendation, patient_number
