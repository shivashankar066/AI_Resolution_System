class ResponseJson:
    def response_json_object(
            self,
            s_name,
            time_taken,
            status_code,
            status,
            patient_id,
            recommended_code
    ):
        return {
            "message": s_name,
            "status": status,
            "statusCode": status_code,
            "respTime": int(time_taken * 1000),
            "patient_id": patient_id,
            "recommended_code": recommended_code
        }
