
#  exception.py
import sys
import logging

def error_message_detail(error, error_detail: sys):
    """Capture detailed error info: filename, line number, and message."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in Python script name [{file_name}] "
        f"at line number [{exc_tb.tb_lineno}] "
        f"with message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """Custom Exception class for detailed error tracking."""
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
