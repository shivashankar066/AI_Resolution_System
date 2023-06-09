import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from django.db import connection

import numpy as np
from configparser import ConfigParser
from time import time
from datetime import date

import warnings
warnings.filterwarnings("ignore")

from .apps import QcdspeConfig

config = ConfigParser()
config.read("config/config.ini")


def get_payer_allowed_value(row):

    allowed = row['Allowed']

    try:
        is_na = pd.isnull(allowed)
    except TypeError:
        is_na = np.isnan(allowed)

    if not is_na:
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
                payor_allowed = float(service_fee) * float(QcdspeConfig.cpt_avg_allowed['allowed_ratio_q3']) * float(service_units)

    return payor_allowed


def age_gender_loc_similarity(df, patient_sex, patient_state, age_range=3):
    """ Runs demographic similarity checks between
    patients in the input dataframe and the current patient
    :param df: Pandas dataframe with data of patients with the same carrier as current patient
    :param age_range: age difference threshold

    :returns: d3: a filtered input dataframe (df) with matched patient records
    """
    d1 = df[df.age_diff == age_range]
    d2 = d1[d1.patient_sex == patient_sex]
    d3 = d2[d2.Patient_State == patient_state]
    return d3


def get_similar_patient_info(missing_cpts, current_carrier, patient_age, diagnosis_codes, patient_sex, patient_state):
    """
    Finds patients similar to the current patient in terms of
    procedure cods, payer, diagnosis codes, age, gender and location
    :param missing_cpts: Procedure codes that were not performed on the patient in the past
    :return: Dataframe with similar patient records
    """

    if current_carrier:
        # Extracting historical data of records for the patient's recent carrier
        result = pd.DataFrame() #"query db where Original_Carrier_Name == current_carrier"

        # Filtering records where the CPT codes are matching with the missing CPTs
        result_cpt = result[result.Procedure_Code.isin(missing_cpts)]
    else:
        # If the patient data doesn't have carrier name at all
        # use historical data where same diag codes and cpt codes were used
        result = pd.DataFrame() #"query db where Diagnosis_Code.isin(diagnosis_codes)

        result_cpt = result[result.Procedure_Code.isin(missing_cpts)]

    # Checking if any of the missing CPTs are not present in the current carrier data
    current_carrier_cpts = result_cpt.Procedure_Code.unique().tolist()
    missing_cpt_in_curr_carrier = [p for p in missing_cpts if p not in current_carrier_cpts]

    # If there are missing CPTs in current carrier data, considering historical data for the CPTs irrespective of
    # carrier
    if len(missing_cpt_in_curr_carrier) > 0:
        missing_cpt_result = pd.DataFrame() #"query db where Procedure_Code.isin(missing_cpt_in_curr_carrier)
        result_cpt = pd.concat([result_cpt, missing_cpt_result], axis=0)

    # Calculate difference in age between the patient and those in the historical data
    result_cpt.insert(len(result_cpt.columns),'age_diff',abs(result_cpt.patient_age - patient_age),True)

    # Iterating over each of the missing CPTs to find similar patient cohort based on age, diag_codes, and location
    cpt_dfs = []
    null_cpts = []

    for cpt in missing_cpts:
        cpt_df = result_cpt[result_cpt.Procedure_Code == str(cpt)]
        if cpt_df.empty:

            # There are CPTs for which records are getting wiped off in the data cleaning steps.
            # Eg: Service_Fee is always zero for 3 CPTs in the 15 CPT list given by EPIC.
            # There are 44 such CPTs in the diabetes funsd_dataset
            # logging such CPTs here in the null_CPTs list
            null_cpts.append(cpt)
            continue

        diagnosis_codes = [i.strip() for i in diagnosis_codes]
        similar_pats = cpt_df.Patient_Number.unique().tolist()

        sim_pat_diag = {}
        # Diagnosis codes of patients with this cpt
        for similar_pat in similar_pats:
            d_codes = cpt_df[cpt_df.Patient_Number == similar_pat].Diagnosis_Code.unique().tolist()
            d_codes = [i.strip() for i in d_codes]
            sim_pat_diag[similar_pat] = d_codes

        # Finding patients with similar diagnosis codes based on maximum no of matching codes
        diag_sim = {}
        for k,v in sim_pat_diag.items():
            sim = len(set(diagnosis_codes) & set(v))
            try:
                diag_sim[sim].append(k)
            except KeyError:
                diag_sim[sim] = [k]

        # Sorting the diag code similarity dictionary in descending order
        diag_sim = dict(sorted(diag_sim.items(), reverse=True))

        # Iterating over the diag code dictionary starting with patients with maximum similarity score
        filtered_df = pd.DataFrame()
        for k in diag_sim:
            k_df = cpt_df[cpt_df.Patient_Number.isin(diag_sim[k])]
            # Selected patient list is filtered again for demographic similarity based on age, gender and location
            filtered_df = age_gender_loc_similarity(k_df,patient_sex, patient_state)
            # If demographic similarity is nil for the above selected patients,
            # re-run the iteration on patients with lesser diag_code similarity
            if filtered_df.shape[0] == 0:
                continue
            else:
                break
        # In cases where we couldn't find patients with diag similarity
        if filtered_df.shape[0] == 0:
            # First look for only demographic similarity with age difference of +or- 3 years
            filtered_df = age_gender_loc_similarity(cpt_df,patient_sex, patient_state)
            if filtered_df.shape[0] == 0:
                # If still no similar records, relax the age difference to +or- 4 years
                filtered_df = age_gender_loc_similarity(cpt_df, patient_sex, patient_state, age_range=4)
                if filtered_df.shape[0] == 0:
                    # If still no similar records, relax the age difference to +or- 5 years
                    filtered_df = age_gender_loc_similarity(cpt_df,patient_sex, patient_state, age_range=5)
                    if filtered_df.shape[0] == 0:
                        # If still no similar records, consider whatever records are available
                        filtered_df = cpt_df
        cpt_dfs.append(filtered_df)

    if len(cpt_dfs) == 0:
        similar_patient_data = pd.DataFrame()
    else:
        # Combine all the similar patient records relating to all missing CPTs
        similar_patient_data = pd.concat(cpt_dfs, axis=0)
        similar_patient_data.drop(['age_diff'], axis='columns', inplace=True)

    return similar_patient_data, null_cpts


def get_patient_details(patient_id, allscripts=False):

    cursor = connection.cursor()

    if not allscripts:

        # cdm_cols
        column_names = [
            'Patient_ID',
            'DoB',
            "Proc_Category_Abbr",
            'patient_zip_code',
            'patient_sex',
            "Original_Carrier_Name",
            'Patient_City',
            'Patient_State',
            'CoPayment',
            'CoInsurance',
            "Primary_Diagnosis_Code",
            "Procedure_Code",
            "Service_Units",
            "Service_Fee",
            "Allowed",
            "Deductible"
        ]

        # cdm query
        query = """SELECT T1.person_id,T1.birth_datetime,
        t3.procedure_category_abbr,  t8.zip, t1.gender_source_value, t6.payer_source_value,t8.city,
        t8.state, t5.paid_patient_copay, t5.paid_patient_coinsurance,t4.condition_source_value,t3.procedure_source_value,
        T3.quantity, t5.total_charge,t5.amount_allowed, t5.paid_patient_deductible
        FROM cdm.person AS T1
        LEFT JOIN cdm.visit_occurrence AS T2 ON T1.person_id = T2.person_id
        LEFT join cdm.procedure_occurrence as T3 ON T2.visit_occurrence_id = T3.visit_occurrence_id
        left join cdm.condition_occurrence as T4 on T2.visit_occurrence_id = T4.visit_occurrence_id
        left join cdm.cost as T5 on T2.visit_occurrence_id =T5.cost_event_id
        left join cdm.payer_plan_period as T6 on T1.person_id = T6.person_id
        left join cdm.provider as T7 on T2.provider_id = T7.provider_id
        left join cdm.location as T8 on t1.Location_id = t8. Location_id
        left join cdm.care_site as T9 on t8.Location_id = t9. Location_id
        LEFT JOIN cdm.p_ref_transaction_codes as T10 on t5.transaction_code_abbr = T10.abbrevation
        where (t10.self_pay_trans_cde=0)
        and (T10.Description not like '%%Self%%' And T10.Description not like '%%Adj%%')
        And (t10.transaction_type!='A') and (t10.transaction_type !='T') and (t10.transaction_type!='B')
        and (T2.visit_occurrence_id > 0)  and ( t5.total_paid >= 0) and (t5.total_charge > 0)
        and ((T4.condition_source_value BETWEEN 'E08' AND 'E13')
            OR (T4.condition_source_value = 'R73.03'))
        and t1.person_id= %s """

    else:
        # allscripts cols
        column_names = ["Service_ID", "Patient_ID", "Patient_Number", "IMREDEM_CODE", "patient_age",
                        "Actual_Dr_Name", "Place_of_Service_Abbr", "Proc_Category_Abbr",
                        "Type_of_Service_Abbr","patient_zip_code", "patient_sex", "Original_Carrier_Name",
                        "Patient_City", "Patient_State", "CoPayment", "CoInsurance", "Primary_Diagnosis_Code",
                        "Procedure_Code", "Service_Units", "Service_Date_From", "Claim_Number",
                        "Original_Billing_Date", "Date_Paid", "Service_Fee", "Amount", "Allowed", "Deductible",
                        "Transaction_Type", "Abbreviation", "Description", "Self_Pay_TranCode"]
        # allscripts query
        query = """
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
                (t1.Primary_Diagnosis_Code='R73.03')) and t1.Patient_ID= %s """

    cursor.execute(query, (patient_id,))

    db_response = cursor.fetchall()

    result_df = pd.DataFrame([list(elem) for elem in db_response])

    if result_df.empty:
        result_df = pd.DataFrame()

    else:
        result_df.columns = column_names

    return result_df


def handle_empty_patient_data(patient_id, rule_engine_recommended_code, start_time):

    pred_scores = [1.0] * len(rule_engine_recommended_code)
    rec = {str(patient_id): dict(zip(rule_engine_recommended_code,pred_scores))}

    response = {
        "message": "Prediction Engine Service completed successfully.",
        "status": "Success",
        "statusCode": 200,
        "respTime": round((time() - start_time), 3),
        "patiend_id": str(patient_id),
        "recommended_code": str(rec)
    }

    return response


def get_age(dob):
    today = date.today()
    years = today.year - dob.year
    if today.month < dob.month or (today.month == dob.month and today.day < dob.day):
        years -= 1
    return years


class PredictScore(APIView):

    def post(self, request):

        allscripts = False

        start = time()
        request_data = request.data

        patient_id = request_data['Patient_ID']
        rec_cpt_dict = request_data['Rule_Engine_Recommended_Code']
        rec_cpts = [k for k, v in rec_cpt_dict.items() if int(v) == 1]

        X = get_patient_details(patient_id, allscripts)

        if X.shape[0] == 0:
            response = handle_empty_patient_data(patient_id, rec_cpts, start)
            return Response(response)

        X = X.drop_duplicates()
        # Missing value treatment
        X['Original_Carrier_Name'] = X.Original_Carrier_Name.fillna(QcdspeConfig.most_frequent_payer)
        X['CoInsurance'] = X.CoInsurance.fillna(0)
        X['CoPayment'] = X.CoPayment.fillna(0)
        X['Deductible'] = X.Deductible.fillna(0)

        X['Allowed'] = X.apply(get_payer_allowed_value, axis=1)

        #calculate patient age based on DoB

        if 'patient_age' not in X.columns:
            X['DoB'] = pd.to_datetime(X['DoB'])
            X['patient_age'] = get_age(X['DoB'][0])

        col_names = [
            'Original_Carrier_Name',  # *
            'patient_age',
            'Proc_Category_Abbr',  # *
            'patient_zip_code',  # need strategy to handle new zip code category
            'patient_sex',
            'Patient_City',
            'Patient_State',
            'CoInsurance',  # *
            'CoPayment',  # *
            'Procedure_Code',
            'Allowed',  # *
            'Deductible'  # *
        ]

        X = X[col_names]

        X = X[X.Procedure_Code.isin(rec_cpts)]

        cpts_in_patient_history = X.Procedure_Code.unique().tolist()
        missing_cpts = [cpt for cpt in rec_cpts if cpt not in cpts_in_patient_history]

        for col in QcdspeConfig.categorical_columns:
            le = QcdspeConfig.label_encoders[col]
            try:
                X[col] = le.transform(X[col])
            except ValueError:
                # Some variables having trailing spaces in train set but not at prediction time.
                col_classes = X[col].unique().tolist()
                le_classes = le.classes_.tolist()
                missing_classes = [lbl for lbl in col_classes if lbl not in le_classes]
                missing_indx_dict = {}
                le_dict = {i: idx for idx, i in enumerate(le_classes)}

                for missing_cls in missing_classes:
                    missing_cls_trimmed = missing_cls.strip()
                    indx = None
                    if missing_cls_trimmed in le_classes:
                        indx = le_classes.index(missing_cls_trimmed)
                    else:
                        le_classes_trimmed = [i.strip() for i in le_classes]
                        if missing_cls_trimmed in le_classes_trimmed:
                            indx = le_classes_trimmed.index(missing_cls_trimmed)
                    if indx:
                        missing_indx_dict[missing_cls] = indx

                le_dict.update(missing_indx_dict)

                X[col] = [le_dict[label] for label in X[col]]


        # Reorder input columns in line with their order at training time
        X = X[QcdspeConfig.model_fit_feature_order]

        input_cpts = X.Procedure_Code.tolist()
        cpt_encoder = QcdspeConfig.label_encoders["Procedure_Code"]
        input_cpts = cpt_encoder.inverse_transform(input_cpts)

        predictions = QcdspeConfig.model.predict(X)

        predictions_list = np.round(predictions, 3).tolist()
        output = dict(zip(input_cpts, predictions_list))

        # Zero score to CPTs missing in patient history
        output.update(dict(zip(missing_cpts, [0]*len(missing_cpts))))

        sorted_output = dict(sorted(output.items(), key=lambda x: x[1], reverse=True))
        end = time()

        response = {
            "message": "Prediction Engine Service completed successfully.",
            "status": "Success",
            "statusCode": 200,
            "respTime": round(end-start, 3),
            "patient_id": str(patient_id),
            "recommended_code": str(sorted_output)
        }

        return Response(response)
