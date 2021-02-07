from http import HTTPStatus
from typing import Dict, List
from transformers import RobertaTokenizer
import logging
import base64

import numpy as np
import tornado
from adserver.protocols.request_handler import (
    RequestHandler,
)  # pylint: disable=no-name-in-module

def create_text_from_onnx(data: str, ty: int, tokenizer) -> str:
    
    # ty : onnx.TensorProto.DataType in onnx_ml_pb2.py
    
    npty = np.float
    if ty == 9:
        npty = np.bool
    elif ty == 2:
        npty = np.uint8
    elif ty == 4:
        npty = np.uint16
    elif ty == 12:
        npty = np.uint32
    elif ty == 13:
        npty = np.uint64
    elif ty == 3:
        npty = np.int8
    elif ty == 5:
        npty = np.int16
    elif ty == 6:
        npty = np.int32
    elif ty == 7:
        npty = np.int64
    elif ty == 1:
        npty = np.float32
    elif ty == "FP32":
        npty = np.float32
    elif ty == 11:
        npty = np.float64
    else:
        raise ValueError(f"ONNX unknown type or type that can't be coerced {ty}")

    np_data = np.frombuffer(base64.b64decode(data), dtype=npty)
    np_data = np_data.reshape(1,-1)
    logging.info(np_data.shape)
    logging.info(np_data)
    return np_data
    #text_data = tokenizer.decode(np_data)
    #text_data = text_data.split("</s>")[0].strip("<s>")
    #logging.info(text_data)
    #return text_data

class OnnxRequestHandler(RequestHandler):
    def __init__(self, request: Dict):  # pylint: disable=useless-super-delegation
        super().__init__(request)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")        

    def validate(self):
        if not "inputs" in self.request:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='Expected key "data" in request body',
            )

    def extract_request(self) -> List:
        #logging.info(self.request)

        return create_text_from_onnx(self.request["inputs"]["tweet_input"]["rawData"],
                                     self.request["inputs"]["tweet_input"]["dataType"],
                                     self.tokenizer).tolist()
        # return [
        #     create_text_from_onnx(self.request["inputs"]["tweet_input"]["rawData"],
        #                           self.request["inputs"]["tweet_input"]["dataType"],
        #                           self.tokenizer 
        #     )
        # ]
