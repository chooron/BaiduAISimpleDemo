import pandas as pd
import numpy as np
import joblib

LABEL_NAME = 'class'


class PyModel(object):
    """
    Sklearn 自定义算法代码
    """

    def __init__(self):
        """
        类的构造函数
        """
        self.feature_name = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.model = None
        self.load()

    def load(self):
        """
        load the real model
        :return:
        """
        # model_path the relative path of model, must in the same path of this code file
        model_path = 'model.pkl'
        self.model = joblib.load(model_path)

    def transform(self, dataset):  # type:(pd.DataFrame)->pd.DataFrame
        """

        :param dataset:
        :return:
        """
        sample_df = dataset[['sample']].dropna(axis=0)
        sample_df_convert = pd.DataFrame(sample_df.to_dict()['sample']).T
        param_df = dataset[['params']].dropna(axis=0)
        param_dict = param_df.to_dict()['params']

        def slide_window(df, look_back, lead_time, feature_cols, target_cols):
            x_list = []
            for i in range(0, len(df) - look_back - lead_time + 1, 1):
                temp_x = df[feature_cols].values[i:i + look_back, :].reshape(1, -1)
                x_list.append(temp_x)
            x = np.concatenate(x_list, axis=0)
            y = df[target_cols].values[lead_time + look_back - 1:, :].reshape(-1, 1)
            return x, y

        x, y = slide_window(sample_df_convert, param_dict['look_back'], param_dict['lead_time'],
                            param_dict['feature_cols'], param_dict['target_cols'])

        # 使用和训练时相同的 feature name
        prediction = self.model.predict(x)
        return pd.DataFrame({'pred': prediction})

