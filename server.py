#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
Env: python
File: serving.py
"""
import sys
import logging
import pandas as pd
from flask import Flask
from flask import json
from flask import request
from model import PyModel

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pymodel = PyModel()

port_num = int(sys.argv[1]) if len(sys.argv) > 1 else 8080


@app.route('/v1/api', methods=['POST'])
def get_pred():
    """
    :return:
    """
    data = json.loads(request.data)["data"]
    columns = pymodel.original_feature_name
    df = pd.DataFrame(data, columns=columns)
    df_output = pymodel.transform(df)
    resp = {"result": df_output.values.tolist()}

    return json.dumps(resp, ensure_ascii=False)


def main():
    """
    :return:
    """
    app.run(host="0.0.0.0", port=port_num, debug=False)


if __name__ == '__main__':
    logger.info("sample request:")
    logger.info('curl -H "Content-Type: application/json" -X POST '
                '-d \'{"data": [[161,415,0,0,0,332.9,67,56.59,317.8,97,27.01,160.6,128,7.23,5.4,9,1.46,4]]}\' '
                'localhost:8080/v1/api')
    logger.info("request data feature sequence required: {}".format(pymodel.original_feature_name))
    main()
