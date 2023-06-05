import pandas as pd


class Integration:

    def integration(self, rule_engine_recommended_code, ml_recommendation, patient_id, patient_number):
        """takes dictionary, lists, patient id as inputs and returns pandas dataframe"""
        resultant_df = dict()
        # r73_procs = ['82985','82043']
        re_procedure_code = [k for k, v in rule_engine_recommended_code.items() if v == '1']

        list_val = []
        ml_pro_code = ml_recommendation[patient_number]["Procedure_Code"]
        ml_score = ml_recommendation[patient_number]["Predicted_Score"]

        for k in re_procedure_code:

            # if k in r73_procs:
            # list_val.append([k, 0, 1])
            if k in ml_pro_code:
                pos = ml_pro_code.index(k)
                sco = ml_score[pos]
                list_val.append([k, sco, 1])
            else:
                list_val.append([k, 0, 1])

        final_proc_codes = pd.DataFrame(list_val, columns=['Proc_code', 'Pred_score', 'Eligibility'])
        final_proc_codes['PatientID'] = patient_id
        final_proc_codes = final_proc_codes.sort_values(by='Pred_score', ascending=False)
        final_proc_codes = final_proc_codes.loc[:, ["Proc_code", "Pred_score"]]
        final_proc_codes = final_proc_codes.reset_index(drop=True)
        # res = final_proc_codes.to_dict("list")
        # Create the nested dictionary with Proc_code as keys and Pred_score as values
        res = {patient_id: {'Proc_code': {code: score for code, score in final_proc_codes.values}}}
        # resultant_df[patient_id] = res
        # return resultant_df
        resultant_df.update(res)
        return res
