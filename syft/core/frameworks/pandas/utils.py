import pandas
import pandas as pd
import json


class Serialiser():
    """#TODO Hook the pandas functions here and remove all the pandas
    functions from the Workers class, so that you could add more to the
    pandas functions"""
    def __init__(self):
        pass

    def serialise(self, pandas_obj):
        """This function Serielises the pandas dataframe obects namely series and to JSON
        objects. It first checks that the object is an instance of the pandas DataType"""
        assert isinstance(pandas_obj, (pandas.core.series.Series, pandas.core.frame.DataFrame))
        obj = {}
        if isinstance(pandas_obj, pandas.core.series.Series):
                obj['data'] = pandas_obj.tolist()
                obj['dtype'] = 'Series'
                obj['index'] = {}
                if(isinstance(pandas_obj.index, pandas.RangeIndex)):
                    obj['index']['type'] = 'RangeIndex'
                    params = pandas_obj.index.__str__()[11:-1].split(",")
                    start = params[0].split("=")[1]
                    stop = params[1].split("=")[1]
                    step = params[2].split("=")[1]
                    obj['index']['start'] = start
                    obj['index']['stop'] = stop
                    obj['index']['step'] = step
                if(pandas_obj.name is not None):
                    obj["name"] = obj.name
                return json.dumps(obj)

        if isinstance(pandas_obj, pandas.core.frame.DataFrame):
                obj = {}
                height, columns = pandas_obj.shape
                obj['data'] = []
                obj['dtype'] = 'DataFrame'
                for _ in range(columns):
                    obj['data'].append(pandas_obj.iloc[:, _].tolist())
                obj['coloumn_values'] = pandas_obj.columns.values.tolist()
                obj['index'] = {}
                if(isinstance(pandas_obj.index, pandas.core.indexes.range.RangeIndex)):
                    obj['index']['type'] = 'RangeIndex'
                    params = pandas_obj.index.__str__()[11:-1].split(",")
                    start = params[0].split("=")[1]
                    stop = params[1].split("=")[1]
                    step = params[2].split("=")[1]
                    obj['index']['start'] = start
                    obj['index']['stop'] = stop
                    obj['index']['step'] = step
                return json.dumps(obj)

    def deserialise(self, obj_json):
        # TODO rangeindex, multiindex, CategoricalIndex, IntervalIndex on return function
        assert isinstance(obj_json, str)
        raw_obj = json.loads(obj_json)
        if 'Series' == raw_obj['dtype']:
            print('lol')
            return pd.Series(data=raw_obj['data'], index=raw_obj['index'])
        if 'DataFrame' in raw_obj['dtype']:
            df = pd.DataFrame(raw_obj['data'])
            df = df.transpose()
            df.columns = raw_obj['coloumn_values']
            return df
