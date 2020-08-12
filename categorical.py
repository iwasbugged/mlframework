from sklearn import preprocessing


class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_types, handle_na=False):
        """
        df = pandas dataframe
        categorical_features = list of columns name for encoding
        encoding types = encoding type e.g (label, binary, ohe)
        handle_na = whether "NaN " value to be handle or not (True/Flase)
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_types = encoding_types
        self.handle_na = handle_na
        self.label_encoder = dict()
        self.binary_encoder = dict()
        self.ohe_encoder = dict()


        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna(-9999)
        self.output_df =  self.df.copy(deep= True)

    def _lable_encoder(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:,c] = lbl.transform(self.df[c].values)
            self.label_encoder[c] = lbl
        return self.output_df

    def _binary_encoder(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c , axis = 1)
            """
            since "LabelBinarization" provide ndarray we need new columns for ever columns of the array
            """
            for j in range(val.shape[1]):
                new_col_name = c+ f"_bin{j}"
                self.output_df[new_col_name] = val[:,j]
            self.binary_encoder[c] = lbl
        return self.output_df

    def _one_hot(self):
        for c in self.cat_feats:
            ohe = preprocessing.OneHotEncoder()
            ohe.fit(self.df[c].values.reshape(-1,1))
            val = ohe.transform(self.df[c].values.reshape(-1,1))
            self.output_df = self.output_df.drop(c, axis = 1)
            for j in range(val.shape[1]):
                new_col_name = c + f"_bin{j}"
                self.output_df[new_col_name] = val[:,j]
            self.ohe_encoder[c]=ohe
        return self.output_df

    
    def fit_transform(self):
        if self.enc_types == 'label':
            return self._lable_encoder()
        elif self.enc_types == 'binary':
            return self._binary_encoder()
        elif self.enc_types == "ohe":
            return self._one_hot()

      

    def transform(self,dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:,c] = dataframe.loc[:,c].astype(str).fillna(-9999)

        if self.enc_types == 'label':
            for c , lbl in self.label_encoder.items():
                dataframe.loc[:,c ] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_types == 'binary':
            for c , lbl in self.binary_encoder.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c , axis = 1)
                for j in range(val.shape[1]):
                    new_col_name = c +f"_bin{j}"
                    dataframe[new_col_name] = val[:,j]
            return dataframe

        elif self.enc_types == "ohe":
            for c in self.cat_feats:
                val = ohe.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis = 1)
                for j in range(val.shape[1]):
                    new_col_name = c + f"_bin{j}"
                    dataframe[new_col_name] = val[:,j]
            return dataframe



if __name__ ==  "__main__":
    import pandas as pd 
    df_train = pd.read_csv("hr_train.csv")
    df_test = pd.read_csv('hr_test.csv')
    sample = pd.read_csv('hr_sample_submission.csv')

    df_train.drop('employee_id', axis= 1 ,inplace= True)
    df_test.drop('employee_id', axis= 1, inplace= True)

    train_len = len(df_train)

    df_test['is_promoted'] = -1
    full_data = pd.concat([df_train , df_test])

    cols =  ['department', 'region', 'education', 'gender','recruitment_channel']
    cat_feats = CategoricalFeatures(df = full_data,
                                    categorical_features = cols,
                                    encoding_types = 'ohe',
                                    handle_na = True)

    full_data_transformed = cat_feats.fit_transform()
    print(full_data_transformed)