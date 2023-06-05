class ErrorCodes:
    """
    class to define error codes
    """

    def __init__(self):

        self.Success = 200
        self.APIValidationFailure = 1001
        self.OrientationFailure = 201
        self.DataUpdateFailure = 152
        self.InternalError = 210
        self.ClassNotFound = 1002
        self.SuccessMsg = "Success"
        self.FailureMsg = "Failure"

    def return_error_message(self, error_code):
        """

        :type error_code:

        """
        error_message = {
            "200": " completed successfully.",
            "1001": " failed to start. Invalid API request",
            "151": " failed to insert record in database.",
            "152": " failed to update record in Local.",
            "210": " failed. Internal error",
            "1002": " Class Not Found In Database",
        }
        return error_message[error_code]
