import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from django.db import connection
from django.conf import settings
from decimal import Decimal

import numpy as np
from configparser import ConfigParser
from time import time
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings("ignore")

import logging
import os

from .apps import QcdspeConfig

config = ConfigParser()
config.read("config/config.ini")


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename=os.path.join(settings.LOG_DIR, settings.LOG_FILE), level=logging.INFO)


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
                payor_allowed = float(service_fee) * float(QcdspeConfig.cpt_avg_allowed['allowed_ratio_q3']) * \
                                float(service_units)

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
        result = pd.DataFrame() # "query db where Diagnosis_Code.isin(diagnosis_codes)

        result_cpt = result[result.Procedure_Code.isin(missing_cpts)]

    # Checking if any of the missing CPTs are not present in the current carrier data
    current_carrier_cpts = result_cpt.Procedure_Code.unique().tolist()
    missing_cpt_in_curr_carrier = [p for p in missing_cpts if p not in current_carrier_cpts]

    # If there are missing CPTs in current carrier data, considering historical data for the CPTs irrespective of
    # carrier
    if len(missing_cpt_in_curr_carrier) > 0:
        missing_cpt_result = pd.DataFrame() # "query db where Procedure_Code.isin(missing_cpt_in_curr_carrier)
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

def get_patient_details(patient_id,allscripts=False):
    """ function for fetching patient details from database"""
    cursor = connection.cursor()

    if not allscripts:

        # cdm_cols
        column_names = [
            'visit_detail_id',
            'person_id',
            'pm_patient_number',
            'birth_datetime',
            'location_source_value',
            'procedure_category_abbr',
            'type_of_service_abbr',
            'zip',
            'gender_source_value',
            'payer_source_value',
            'city',
            'state',
            'paid_patient_copay',
            'paid_patient_coinsurance',
            'condition_source_value',
            'procedure_source_value',
            'quantity',
            'visit_detail_start_datetime',
            'preceding_visit_detail_id',
            'original_billing_date',
            'date_paid',
            'total_charge',
            'total_paid',
            'amount_allowed',
            'paid_patient_deductible',
            'transaction_type',
            'abbrevation',
            'self_pay_trans_cde'
        ]

        # cdm query
        query = """select T2.visit_detail_id, T1.person_id, t1.pm_patient_number,  t1.birth_datetime,
                T8.location_source_value,t3.procedure_category_abbr,t3.type_of_service_abbr, 
                T8.zip, t1.gender_source_value, t6.payer_source_value,T8.city,T8.state, 
                t5.paid_patient_copay, t5.paid_patient_coinsurance,t4.condition_source_value,
                t3.procedure_source_value,T3.quantity,  T2.visit_detail_start_datetime, t2.preceding_visit_detail_id,
                t2.original_billing_date,
                t5.date_paid , t5.total_charge, t5.total_paid,t5.amount_allowed, t5.paid_patient_deductible, 
                t10.transaction_type,t10.abbrevation,
                T10.self_pay_trans_cde from cdm.person AS T1 LEFT JOIN cdm.visit_detail AS T2 ON 
                T1.person_id = T2.person_id   
                LEFT join cdm.procedure_occurrence as T3 ON T2.visit_detail_id= T3.visit_detail_id    
                left join cdm.condition_occurrence as T4 on T2.visit_detail_id = T4.visit_detail_id   
                left join cdm.cost as T5 on T2.visit_detail_id =T5.cost_event_id       
                LEFT JOIN cdm.p_ref_transaction_codes as T10 on t5.transaction_code_abbr = T10.abbrevation  
                left join cdm.payer_plan_period as T6 on T1.person_id = T6.person_id   
                left join cdm.care_site as T7 on T1.care_site_id =T7.care_site_id  
                left join cdm.location as T8 on T8.location_id =t7.location_id where
                ((t4.condition_source_value BETWEEN 'E08' AND 'E13') OR (t4.condition_source_value = 'R73.03') OR
                (t4.condition_source_value BETWEEN 'E66.0' AND 'E66.99') OR 
                (t4.condition_source_value BETWEEN 'I10' AND 'I16')
                OR (t4.condition_source_value BETWEEN 'I25.00' AND 'I25.99')) And (t10.transaction_type!='A') and 
                (t10.transaction_type !='T') and (t10.transaction_type!='B')
                and  (t5.total_paid >= 0) and (t5.total_charge > 0) and
                t1.person_id= %s """

    else:
        # allscripts cols
        column_names = ["Service_ID", "Patient_ID", "Patient_Number", "patient_age",
                        "Actual_Dr_Name", "Place_of_Service_Abbr", "Proc_Category_Abbr",
                        "Original_Carrier_Category_Abbr","Patient_Marital_Status",
                        "Type_of_Service_Abbr","patient_zip_code", "patient_sex", "Original_Carrier_Name",
                        "Patient_City", "Patient_State", "CoPayment", "CoInsurance", "Primary_Diagnosis_Code",
                        "Procedure_Code", "Service_Units", "Service_Date_From", "Claim_Number",
                        "Original_Billing_Date", "Date_Paid", "Service_Fee", "Amount", "Allowed", "Deductible",
                        "Transaction_Type", "Abbreviation", "Description", "Self_Pay_TranCode","Patient_Date_Reg",
                        "Reimbursement_Comment_Abbr","Denied"]
        # allscripts query
        query = """
                SELECT t1.Service_ID, t1.Patient_ID, t1.Patient_Number, t5.patient_age, t1.Actual_Dr_Name, 
                t1.Place_of_Service_Abbr,
                t1.Proc_Category_Abbr, t1.Original_Carrier_Category_Abbr, t5.Patient_Marital_Status, 
                t1.Type_of_Service_Abbr,
                t5.patient_zip_code, t5.patient_sex, t1.Original_Carrier_Name, t5.Patient_City, t5.Patient_State, 
                t2.CoPayment,
                t2.CoInsurance, t1.Primary_Diagnosis_Code, t1.Procedure_Code, t1.Service_Units, 
                CONVERT(Date,t1.Service_Date_From)
                AS Service_Date_From, t1.Claim_Number, CONVERT(Date, t1.Original_Billing_Date) AS Original_Billing_Date,
                CONVERT(Date, t2.Date_Paid) AS Date_Paid, t1.Service_Fee, t2.Amount, t2.Allowed, t2.Deductible,
                t2.Transaction_Type, t4.Abbreviation, t4.Description, t4.Self_Pay_TranCode, 
                CONVERT(Date,t5.Patient_Date_Reg)
                AS Patient_Date_Reg, t2.Reimbursement_Comment_Abbr, t3.Denied
                FROM PM.vwGenSvcInfo AS T1
                LEFT JOIN PM.[vwGenSvcPmtInfo] T2 ON T1.Service_Id=T2.Service_Id
                LEFT JOIN PM.Reimbursement_Detail T3 ON T1.Service_Id=T3.Service_Id
                LEFT JOIN [dbo].[vUAI_Transaction_Codes] T4 ON T2.Transaction_Code_Abbr=T4.Abbreviation
                LEFT JOIN [EMR].[HPSITE].[DEMOGRAPHICS_VIEW] AS d ON t1.Patient_Number = d.DEM_EXTERNALID
                LEFT JOIN PM.vwGenPatInfo AS T5 ON T1.Patient_Number=T5.Patient_Number
                WHERE (T2.Transaction_Type != 'A') AND (T2.Transaction_Type != 'T') AND (T2.Transaction_Type != 'B') AND
                (T1.Service_Fee > 0) AND (t2.Amount >= 0) AND
                ((t1.Primary_Diagnosis_Code BETWEEN 'E08' AND 'E13') OR (t1.Primary_Diagnosis_Code = 'R73.03') OR
                (t1.Primary_Diagnosis_Code BETWEEN 'E66.0' AND 'E66.99') OR (t1.Primary_Diagnosis_Code BETWEEN 
                'I10' AND 'I16') OR
                (t1.Primary_Diagnosis_Code BETWEEN 'I25.00' AND 'I25.99')) AND t2.Date_Paid >= DATEADD(day, -455, 
                GETDATE())
                AND t1.Service_Date_From BETWEEN DATEADD(month, -15, GETDATE()) AND DATEADD(month, -3, GETDATE())
                AND t1.Patient_ID = %s
            """

    cursor.execute(query,(patient_id,))
    db_response = cursor.fetchall()

    result_df = pd.DataFrame([list(elem) for elem in db_response])
    cursor.close()
    if result_df.empty:
        result_df = pd.DataFrame()

    else:
        result_df.columns = column_names
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        # Change column name w.r.t training Data

        new_column_names = {
            'visit_detail_id': 'Service_ID',
            'person_id': 'Patient_ID',
            'pm_patient_number': 'Patient_Number',
            'birth_datetime': 'DoB',
            'provider_name': 'Actual_Dr_Name',
            'location_source_value': 'Place_of_Service_Abbr',
            'procedure_category_abbr': 'Proc_Category_Abbr',
            # 'Patient_Marital_Status':'Patient_Marital_Status',
            'type_of_service_abbr': 'Type_of_Service_Abbr',
            'zip': 'patient_zip_code',
            'gender_source_value': 'patient_sex',
            'payer_source_value': 'Original_Carrier_Name',
            'city': 'Patient_City',
            'state': 'Patient_State',
            'paid_patient_copay': 'CoPayment',
            'paid_patient_coinsurance': 'CoInsurance',
            'condition_source_value': 'Primary_Diagnosis_Code',
            'procedure_source_value': 'Procedure_Code',
            'quantity': 'Service_Units',
            'visit_detail_start_datetime': 'Service_Date_From',
            # 'preceding_visit_detail_id':'',
            'original_billing_date': 'Original_Billing_Date',
            'date_paid': 'Date_Paid',
            'total_charge': 'Service_Fee',
            'total_paid': 'Amount',
            'amount_allowed': 'Allowed',
            'paid_patient_deductible': 'Deductible',
            'transaction_type': 'Transaction_Type',
            'abbrevation': 'Abbreviation',
            'self_pay_trans_cde': 'Self_Pay_TranCode',
            # 'Patient_Date_Reg':'',
            # 'revenue_code_source_value':'',
            # 'modifier_concept_id':''
        }
        result_df.rename(columns=new_column_names, inplace=True)
    return result_df

def handle_empty_patient_data(patient_id, primary_diag_code,rule_engine_recommended_code, start_time):
    """function for handling empty patient data in database"""

    pred_scores = [0.999] * len(rule_engine_recommended_code)
    proc_code = dict(zip(rule_engine_recommended_code,pred_scores))
    icd_sorted_output = {}
    icd_sorted_output['ICD'] = primary_diag_code
    icd_sorted_output['Proc_code'] = proc_code

    return icd_sorted_output

def get_age(dob):
    """function to return actual age"""
    today = date.today()
    years = today.year - dob.year
    if today.month < dob.month or (today.month == dob.month and today.day < dob.day):
        years -= 1
    return years

def q3(x):
    """function to return 3rd quartile"""
    return x.quantile(0.75)

# function for calculating the payout_ratio
def get_payout_ratio(row,cpt_payment_dict):
    cpt = row['Procedure_Code']
    payor = row['Original_Carrier_Name']
    amt = row['Amount_per_serv_unit']
    med_payout = cpt_payment_dict[payor, str(cpt)]

    if med_payout == 0:
        if amt > med_payout:
            payout_ratio = 1
        else:
            payout_ratio = 0
    else:
        if amt > med_payout:
            amt = med_payout

        payout_ratio = amt / med_payout

    return payout_ratio


def get_delay(row):
    date_paid = row['Date_Paid']
    billing_date = row['Original_Billing_Date']

    diff = (date_paid - billing_date).days

    amt = row["Amount"]

    if amt == 0:
        return 0
    else:
        return diff

def onehot_encoding_sex(row):
    pat_sex=row['patient_sex']
    pat_sex_conv=QcdspeConfig.patient_sex[pat_sex]
    return pat_sex_conv

def onehot_encoding_marital_status(row):
    pat_marital=row['Patient_Marital_Status']
    pat_marital_conv=QcdspeConfig.patient_marital_status[pat_marital]
    return pat_marital_conv

def update_procedure_count(row):
    cpt = row['Procedure_Code']
    cpt_count = QcdspeConfig.cpt_count_dic[str(cpt)]
    return cpt_count

def update_cpt_diag_count(row):
    cpt = row['Procedure_Code']
    cpt_diag_count = QcdspeConfig.cpt_diagnosis_dic[str(cpt)]
    return cpt_diag_count

# Function for identifying the average amount w.r.t procedure_code
def update_cpt_average_amount(row):
    cpt = row['Procedure_Code']
    cpt_avg_amount = QcdspeConfig.cpt_avg_amount_dic[str(cpt)]
    return cpt_avg_amount

# Function for check whether Procedure_Code belongs to top denied list or not
def top_denied_cpt_update(row):
    cpt=row['Procedure_Code']
    top_denied_cpt_list = QcdspeConfig.top_denied_cpt
    if cpt in top_denied_cpt_list:
        return 1
    else:
        return 0

# Function for check whether Original_Carrier_Name in top denied payer list or not
def top_denied_payer_update(row):
    payer = row['Original_Carrier_Name']
    top_denied_payer_list = QcdspeConfig.top_denied_payer
    if payer.strip() in top_denied_payer_list:
        return 1
    else:
        return 0

# Function to determine marital status based on age
def get_marital_status(age):
    return 'Married' if age > 30 else 'Single'

def reimbursement_comment_count(df):
    remburse_json=QcdspeConfig.reim_comment_count
    df_proc_list=list(df.Procedure_Code)
    keys=remburse_json[1].keys()
    reim_df = pd.DataFrame(columns=keys)
    for reim_dic in remburse_json:
        for proc in df_proc_list:
            if reim_dic['Procedure_Code'] == proc:
                reim_values=list(reim_dic.values())
                reim_df.loc[len(reim_df)] = reim_values

    return reim_df

def get_normed_delay(row,cpt_delay_dict):
    cpt = row['Procedure_Code']
    payor = row['Original_Carrier_Name']
    amt = row['Amount']
    delay = row['delay_in_days']
    max_delay = cpt_delay_dict[payor, str(cpt)]

    if amt == 0:
        delay_normed = 0
    else:
        try:
            delay_normed = 1 - (delay / max_delay)
        except ZeroDivisionError:
            delay_normed = 0

    return delay_normed


def find_conflict_cpts(rec_cpt_list):
    """
    identify the CPTs that are tagged to more than one ICD
    input to this function is the recommendation from the rule engine output.
    returns a dictionary in {cpt:[icds]} format
    """
    reps = {}

    for icd_dict in rec_cpt_list:

        icd = icd_dict['ICD']
        proc_list = icd_dict['CPT']

        for cpt in proc_list:

            try:
                reps[cpt].append(icd)
            except KeyError:
                reps[cpt] = [icd]

    reps = {k: v for k, v in reps.items() if len(v) > 1}

    return reps


def conflict_resolution(cpts_diagnosis, conflict_cpts):
    # for each of the CPTs in the conflict_cpts identify the ICD that has maximum score

    cpts_max_scores = {}
    cpts_max_icds = {}

    # same score for both cpt
    icd_list = [entry['ICD'] for entry in cpts_diagnosis]
    print(icd_list)
    # iterate through the model predictions dictionary and store the ICD that has max score
    for icd_dict in cpts_diagnosis:
        icd = icd_dict['ICD']
        cpt_score_dict = icd_dict["Proc_code"]

        for cpt, score in cpt_score_dict.items():
            if cpt in conflict_cpts.keys():
                try:
                    #else:
                    existing_score = cpts_max_scores[cpt]
                    if existing_score < score:
                        cpts_max_scores[cpt] = score
                    # store the ICD that yas the best score till now
                        cpts_max_icds[cpt] = icd

                except KeyError:
                    cpts_max_scores[cpt] = score
                    cpts_max_icds[cpt] = icd

    new_output = []
    # iterate over the model predictions dictionary again
    # to remove the CPT from the ICD dict, if the ICD is not in the cpts_max_icds dict.
    for icd_dict in cpts_diagnosis:
        icd = icd_dict['ICD']
        cpt_dict = icd_dict['Proc_code'].copy()
        cpts = cpt_dict.keys()

        for cpt in conflict_cpts.keys():
            # check if the conflict cpt is there in the ICD dict. if not, no need to check futher
            if cpt in cpts:
                # if conflict cpt in the ICD dict, check if the ICD is the same as the one in the cpts_max_icds dict.
                # If it is, then that means the CPT in this dict is with the highest score, so we retain it here.
                if icd == cpts_max_icds[cpt]:
                    continue
                else:
                    # If the ICD is different, that means this is not the max score for the cpt.
                    # we need to remove the CPT from this dictionary
                    cpt_dict.pop(cpt)

        new_dict = {"ICD": icd, "Proc_code": cpt_dict}
        new_output.append(new_dict)

    return new_output

class PredictScore(APIView):

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def post(self, request):

        allscripts = False

        start = time()
        request_data = request.data

        patient_id = request_data['Patient_ID']

        rec_cpt_list = request_data['recommendation']

    # Identify conflict cpt and Icd combination #
        conflict_cpts = find_conflict_cpts(rec_cpt_list)
        #print(conflict_cpts)

        cpts_diagnosis=[]
        for rec_cpt_dic in rec_cpt_list:
            primary_diag_code=rec_cpt_dic['ICD']
            rec_cpts = rec_cpt_dic['CPT']
            self.logger.info("Predict Procedure Started for: " + str(patient_id))
            X = get_patient_details(patient_id,allscripts)
            self.logger.info("patient Data fetched")
            if X.shape[0] == 0:
                patient_empty_out = handle_empty_patient_data(patient_id,primary_diag_code,rec_cpts, start)
                cpts_diagnosis.append(patient_empty_out)
                end = time()
                continue
            else:
                X = X.drop_duplicates()
                X = X[X.Procedure_Code.isin(rec_cpts)]
                # Check whether cpt appears first time or not
                if X.shape[0] == 0:
                    print("Entered here")
                    cpt_output = handle_empty_patient_data(patient_id, primary_diag_code, rec_cpts, start)
                    self.logger.info("None of the recommneded CPTs in patient history. "
                                     "Returning all ones as scores")
                    cpts_diagnosis.append(cpt_output)
                    end = time()
                    continue
                else:
            # calculate patient age based on DoB
                    if 'patient_age' not in X.columns:
                        X['DoB'] = pd.to_datetime(X['DoB'])
                        #X['patient_age'] = get_age(X['DoB'])
                        X['patient_age'] = X['DoB'].apply(lambda x: relativedelta(datetime.now(), x).years)

            # Identify Patient Marital Status

                    if 'Patient_Marital_Status' not in X.columns:
                        X['Patient_Marital_Status'] = X['patient_age'].apply(get_marital_status)
            # Missing value treatment
            # For replacing missing payer values with most frequent payer in the past one year
                    X['Original_Carrier_Name'] = X.Original_Carrier_Name.fillna(QcdspeConfig.most_frequent_payer)
            # Filling coinsurance, copay and deductible with zeros
                    X['CoInsurance'] = X.CoInsurance.fillna(0)
                    X['CoPayment'] = X.CoPayment.fillna(0)
                    X['Deductible'] = X.Deductible.fillna(0)
                    X['Service_Fee'] = X.Service_Fee.fillna(0)
            # Replacing allowed value with information from past data
                    X['Allowed'] = X.apply(get_payer_allowed_value, axis=1)
            # # Patient Marital Status Null Value Imputation#####
            #         X.loc[(X['patient_age'] < 30) & (
            #             X['Patient_Marital_Status'].isnull()), 'Patient_Marital_Status'] = 'Single'
            #         X.loc[(X['patient_age'] >= 30) & (
            #             X['Patient_Marital_Status'].isnull()), 'Patient_Marital_Status'] = 'Married'

            # New Aggregates related to Patient Profile
                    X['Service_Date_From'] = pd.to_datetime(X['Service_Date_From'])
            # frequency of procedure code for last one year
                    X_freq=X.groupby(['Patient_Number', 'Procedure_Code']).size().reset_index(name='patient_procedure_freq')
                    X=X.merge(X_freq)
            # Frequency of Medical Consultantion for last One year
                    X_med_freq = X.groupby(['Patient_Number']).size().reset_index(name='medical_consultation_freq')
                    X = X.merge(X_med_freq)

            # Frequency of cpt for particular patient in a day

                    X_freq1 = X.groupby(['Patient_Number', 'Service_Date_From', 'Procedure_Code']).size().reset_index(
                        name='patient_procedure_freq_day')
                    X = X.merge(X_freq1)

            # Medical consultation Frequency for particular patient in a day
                    X_med_freq1 = X.groupby(['Patient_Number', 'Service_Date_From']).size().reset_index(
                        name='medical_consultation_freq_day')
                    X = X.merge(X_med_freq1)

            # Total amount paid for Particular CPT
                    X_ins_amount = X.groupby(["Patient_Number", "Procedure_Code"], as_index=False).agg({"Amount": 'sum'})
            # Total copayment paid for a particular CPT over a year
                    X_copay_amount = X.groupby(["Patient_Number", "Procedure_Code"], as_index=False).agg({"CoPayment": 'sum'})
                    X_total_amount = X_ins_amount.merge(X_copay_amount)
                    X_total_amount["patient_total_amount"] = X_total_amount['Amount'] + X_total_amount['CoPayment']
                    X_total_amount1 = X_total_amount[["Patient_Number", "Procedure_Code", "patient_total_amount"]]
                    X = X.merge(X_total_amount1)

            # Amount Spend for one year
                    X_amount = X.groupby(["Patient_Number"], as_index=False).agg({"Amount": 'sum'})
                    X_copay = X.groupby(["Patient_Number"], as_index=False).agg({"CoPayment": 'sum'})
                    X_amount_spend = X_amount.merge(X_copay)
                    X_amount_spend['Total_Amount_Spend_1year'] = X_amount_spend['Amount'] + X_amount_spend['CoPayment']
                    X_amount_spend1 = X_amount_spend[['Patient_Number', 'Total_Amount_Spend_1year']]
                    X = X.merge(X_amount_spend1)

            # CPT frequency for a particular patient (Num of Visits)
                    X['Visit_count'] = X.groupby(['Patient_Number', 'Procedure_Code']).cumcount() + 1
            # Cummulative Score Calculations
                    X_score=X.copy()
                    X_score['Allowed']=X_score['Allowed'].astype(float)
                    X_score['Service_Units']=X_score['Service_Units'].astype(float)
                    X_score['Allowed_per_serv_unit'] =X_score['Allowed'] / X_score['Service_Units']
                    X_score['Amount']=X_score['Amount'].astype(float)
                    X_score['Amount_per_serv_unit'] = X_score['Amount'] / X_score['Service_Units']
            # finding the q3 for a given cpt by a given payer
                    cpt_payment_q3 = X_score.groupby(['Original_Carrier_Name', 'Procedure_Code']).agg({'Amount_per_serv_unit': q3})
                    cpt_payment_dict = cpt_payment_q3.to_dict('dict')['Amount_per_serv_unit']
            # Applying the function to the 'Amount' column
                    X_score['payout_ratio'] = X_score[['Procedure_Code', 'Original_Carrier_Name', 'Amount_per_serv_unit']].apply(
                        lambda row: get_payout_ratio(row, cpt_payment_dict), axis=1)
                    X_score['payout_ratio'] = X_score['payout_ratio'].round(2)
                    X_score["Original_Billing_Date"] = pd.to_datetime(X_score["Original_Billing_Date"])
                    X_score["Date_Paid"] = pd.to_datetime(X_score["Date_Paid"])
                    X_score['delay_in_days'] = X_score[['Original_Billing_Date', 'Date_Paid', 'Amount']].apply(get_delay, axis=1)
            # finding the median payment for a given cpt by a given payer
                    cpt_delay_max = X_score.groupby(['Original_Carrier_Name', 'Procedure_Code'])[['delay_in_days']].max()
                    cpt_delay_dict = cpt_delay_max.to_dict('dict')['delay_in_days']
                    X_score['normalized_delay'] = X_score[
                        ['Procedure_Code', 'Original_Carrier_Name', 'Amount', 'delay_in_days']].apply(
                        lambda row:get_normed_delay(row,cpt_delay_dict), axis=1)
            # Final score
                    payment_wt = 0.75
                    delay_wt = 0.25
                    X_score["Score"] = (payment_wt * X_score["payout_ratio"]) + (delay_wt * X_score["normalized_delay"])
            # Average Score from Cummulative visit
                    X_score['Cumulative_Score'] = X_score.groupby(['Patient_Number', 'Procedure_Code'])['Score'].cumsum()
                    X_score['Cumulative_Score'] = X_score['Cumulative_Score'] - X_score['Score']
                    X_score['Average_Score'] = X_score['Cumulative_Score'] / (X_score['Visit_count'] - 1)
                    X_score.loc[X_score['Visit_count'] == 1, 'Average_Score'] = 9999
                    X_score=X_score[['Service_ID','Average_Score']]
                    X=X.merge(X_score)
            # cpt count
                    X['procedure_count'] = X.apply(update_procedure_count,axis=1)
            # Top Denied Cpt
                    X['top_denied_cpt'] = X.apply(top_denied_cpt_update, axis=1)
            # Top Denied Payer
                    X['top_denied_payer']=X.apply(top_denied_payer_update, axis=1)

            # reimbursement comment count
                    X_new=reimbursement_comment_count(X)
                    X = X.merge(X_new)
            # dignosis count w.r.t cpt
                    X['diagnosis_count']=X.apply(update_cpt_diag_count,axis=1)
            # Average amount
                    X['Average_Amount'] = X.apply(update_cpt_average_amount,axis=1)

            # One hot encoding

                    X['patient_sex']=X.apply(onehot_encoding_sex, axis=1)
                    X['Patient_Marital_Status']=X.apply(onehot_encoding_marital_status, axis=1)

                   # categorical_col=QcdspeConfig.categorical_cols_list

            # Filter Dataframe with ICD
                    X['Primary_Diagnosis_Code'] = X['Primary_Diagnosis_Code'].str.strip()
                    primary_diag_code = primary_diag_code.strip()
                    X_diag = X[X['Primary_Diagnosis_Code'] == primary_diag_code]
        # Check whether ICD is first time or not
                    if X_diag.shape[0] != 0:
                        X = X_diag
                    cpts_in_patient_history = X.Procedure_Code.unique().tolist()
                    missing_cpts = [cpt for cpt in rec_cpts if cpt not in cpts_in_patient_history]

         # mapping to handle cases of payer name changes, acquisitions, and new payers
                    payer_mapping_dict = QcdspeConfig.payer_mapping['payer_mapping']
                    X = X.replace({"Original_Carrier_Name": payer_mapping_dict})

        # Reorder input columns in line with their order at training time
                    X = X[QcdspeConfig.cat_model_fit_feature_order]

                    input_cpts = X.Procedure_Code.tolist()
                    self.logger.info("pre processing complete")
                    predict_data = X
        # Model Prediction
                    predictions = QcdspeConfig.cat_model.predict(predict_data)
                    self.logger.info("ml prediction complete")
                    predictions_list = np.round(predictions, 3).tolist()
                    output = dict(zip(input_cpts, predictions_list))

         # One score to CPTs missing in patient history
                    self.logger.info("Some of the recommended CPTs in patient history.")
                    self.logger.info("Missing CPTs are:")
                    self.logger.info(missing_cpts)
                    self.logger.info("Returning all ones as scores for these CPTs")

                    output.update(dict(zip(missing_cpts, [0.999]*len(missing_cpts))))

                    sorted_output = dict(sorted(output.items(), key=lambda x: x[1], reverse=True))
                    icd_sorted_output={}
                    icd_sorted_output['ICD'] = primary_diag_code
                    icd_sorted_output['Proc_code'] = sorted_output
                    cpts_diagnosis.append(icd_sorted_output)
            #print(cpts_diagnosis)
        CRE_recommendation = conflict_resolution(cpts_diagnosis, conflict_cpts)
            #print(CRE_recommendation)
        end = time()
        response = {
            "message": "Prediction Engine Service completed successfully.",
            "status": "Success",
            "statusCode": 200,
            "respTime": round(end-start, 3),
            "patient_id": str(patient_id),
            "intermediate_recommended_code": str(cpts_diagnosis),
            "recommended_code": str(CRE_recommendation)
        }

        return Response(response)
