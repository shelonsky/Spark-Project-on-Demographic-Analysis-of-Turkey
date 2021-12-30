# -*- coding: utf-8 -*-
# @Time    : 2021/5/30 16:06
# @Author  : Xiao Lulu

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import re
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import RFormula
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# import spark.implicits._


sparkconf = SparkConf().setAppName('Mernis')
sparkconf.set('spark.executor.memory', '10g')
sparkconf.set('spark.driver.memory', '10g')
sparkconf.set("spark.sql.debug.maxToStringFields", "100")
spark = (SparkSession
         .builder
         .appName("Mernis")
         .config(conf=sparkconf)
         .getOrCreate())

# sc = SparkContext.getOrCreate()
# 加载数据
file_path = '/root/myfile/mernis/data_dump.sql'
data = spark.sparkContext.textFile(file_path). \
    filter((lambda line: re.findall('^\d{6}', line))). \
    map(lambda line: line.split('\t')[:-1])

schema = "uid STRING, national_identifier STRING, first STRING, last STRING, mother_first STRING, " \
         "father_first STRING, gender STRING, birth_city STRING, date_of_birth STRING," \
         "id_registration_city STRING, id_registration_district STRING, address_city STRING," \
         "address_district STRING, address_neighborhood STRING,street_address STRING," \
         "door_or_entrance_number STRING"

df = spark.createDataFrame(data, schema)


# total_count = df.count()  # total_count = 49611709

def format_date(line):
    li = line.split('/')
    if len(li[2]) == 4 and 0 < len(li[1]) <= 2 and 0 < len(li[1]) <= 2:
        return li[2] + '-' + li[1].zfill(2) + '-' + li[0].zfill(2)
    else:
        return 'null'


format_date_udf = udf(format_date, returnType=StringType())

df.createOrReplaceTempView('citizens')
df_format_date = df.withColumn("date_of_birth", format_date_udf(df["date_of_birth"]))
df_format_date = df_format_date.filter(expr("""date_of_birth != 'null'"""))
df_format_date = df_format_date.withColumn('date_of_birth', to_date('date_of_birth')).\
    withColumn('month_of_birth',month('date_of_birth')).\
    withColumn('year_of_birth', year('date_of_birth'))
df_format_date.show(3)

###TODO: N6 计算前10大人口城市人口密度，其中城市的面积可Google搜索，面积单位使用平方千米；

def N6():
    print('=' * 20, 'problem N6', '=' * 20)
    # The top10 city with most citizens
    df_n6 = df_format_date. \
        select('address_city'). \
        groupBy('address_city'). \
        agg(count('*').alias('total')). \
        orderBy('total', ascending=False). \
        limit(10)

    sc = SparkContext.getOrCreate()
    area = [('ADANA', 14030), ('ISTANBUL', 5343), ('BURSA', 10891), ('IZMIR', 7340), ('AYDIN', 8007),
            ('ANKARA', 30715), ('ANTALYA', 1417), ('KOCAELI', 3418), ('KONYA', 38257), ('MERSIN', 15737)]

    df_area = spark.createDataFrame(area, ['address_city', 'area'])
    df_area = df_n6.join(df_area, 'address_city', 'left_outer').orderBy('area')
    df_area.show(10)
    density_df = df_area.withColumn('desity', round(df_area['total'] / df_area['area'], 2))
    density_df.show(10)

N6()

## TODO: N7 根据人口的出身地和居住地，分别统计土耳其跨行政区流动人口和跨城市流动人口占总人口的 比例
def N7():
    print('=' * 20, 'problem N7', '=' * 20)
    total_num = 49611709
    df_n7_district = df_format_date. \
        select('id_registration_district', 'address_district'). \
        filter(col('id_registration_district') != col('address_district'))
    propor_district = df_n7_district.count() / total_num
    print('Proportion of cross-district floating population:%.3f' % propor_district)

    df_n7_city = df_format_date. \
        select('id_registration_city', 'address_city'). \
        filter(col('id_registration_city') != col('address_city'))
    propor_city = df_n7_city.count() / total_num
    print('Proportion of cross-city floating population:%.3f' % propor_city)

N7()

# 将出生日期中的年和月提取出来构成新的列,'year_of_birth'和'month_of_birth'，
# 以便于转换成特征。由于总的数据量过大，从中抽取出4900余份样本进行训练和预测。
df_h1 = df_format_date.sample(False, 0.00005, seed=2018)
df_h1.show(10)
df_h1 = df_h1.dropna()
print(df_h1.count())
feature_col = ['first', 'last', 'mother_first', 'father_first', 'gender', 'birth_city',
               'month_of_birth', 'year_of_birth', 'id_registration_city', 'id_registration_district',
               'address_district', 'address_neighborhood', 'street_address', 'address_city'
               ]

indexOutputCols = [x + '_Index' for x in feature_col]
oheOutputCols = [x + '_OHE' for x in feature_col]
stringIndexer_features = StringIndexer(inputCols=feature_col, outputCols=indexOutputCols,
                                       handleInvalid="skip")
oheEncoder_features = OneHotEncoder(inputCols=indexOutputCols, outputCols=oheOutputCols)
pipeline = Pipeline(stages=[stringIndexer_features, oheEncoder_features])
model = pipeline.fit(df_h1)
res = model.transform(df_h1)

# Split the dataset into training, validation and test set with prob 0.7,0.2 and 0.1.
(trainingData, validData, testData) = res.randomSplit([0.7, 0.2, 0.1], seed=100)
trainingData.persist()
validData.persist()
testData.persist()

#
# TODO: H1. 某人所在城市的预测模型：给定一个人的所有信息（除了所在城市），预测这个人所在的城市。 分析该模型Top1到 Top
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# 增加一列labels， 保留address_city的onehot编码
def H1():
    print('=' * 20, 'problem H1', '=' * 20)
    feature_col = ['first', 'last', 'mother_first', 'father_first', 'gender', 'birth_city',
                   'month_of_birth', 'year_of_birth', 'id_registration_city', 'id_registration_district',
                   'address_district', 'address_neighborhood', 'street_address'
                   ]

    # All the feature columns
    oheOutputCols = [x + '_OHE' for x in feature_col]

    # assemble all the feature columns
    vecAssembler = VectorAssembler(inputCols=oheOutputCols, outputCol='features')
    df_h1 = vecAssembler.transform(trainingData)

    lr = LogisticRegression(featuresCol='features', labelCol='address_city_Index',
                            maxIter=100, regParam=0.3, elasticNetParam=0)
    lrPipeline = Pipeline(stages=[vecAssembler, lr])
    lrModel = lrPipeline.fit(trainingData)

    def evaluate_h1(data, model):
        print(model)
        vecData = vecAssembler.transform(data)
        predictions = model.transform(vecData)
        predictions. \
            select('national_identifier', 'probability', 'address_city_Index', 'prediction'). \
            orderBy('probability', ascending=False). \
            show(n=5, truncate=30)

        evaluator = MulticlassClassificationEvaluator(labelCol='address_city_Index', predictionCol='prediction')
        lrAcc = evaluator.evaluate(predictions)
        print('test accuracy = ', lrAcc)

    evaluate_h1(validData, lrModel)

    # 设置不同超参数
    lr.setRegParam(0.001)
    lrPipeline = Pipeline(stages=[vecAssembler, lr])
    lrModel = lrPipeline.fit(trainingData)
    evaluate_h1(validData, lrModel)

    lr.setRegParam(0.01)
    lrPipeline = Pipeline(stages=[vecAssembler, lr])
    lrModel = lrPipeline.fit(trainingData)
    evaluate_h1(validData, lrModel)

    evaluate_h1(testData, lrModel)


H1()

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### TODO: H2. Given all the information about one person, predict his/her gender.
def H2():
    print('=' * 20, 'problem H2', '=' * 20)
    feature_col = ['first', 'last', 'mother_first', 'father_first', 'birth_city', 'year_of_birth', 'month_of_birth',
                   'id_registration_city', 'id_registration_district', 'address_city',
                   'address_district', 'address_neighborhood', 'street_address'
                   ]

    # All the feature columns
    oheOutputCols = [x + '_OHE' for x in feature_col]
    vecAssembler = VectorAssembler(inputCols=oheOutputCols, outputCol='features')
    lr_h2 = LogisticRegression(featuresCol='features', labelCol='gender_Index',
                               maxIter=100, regParam=0.01, elasticNetParam=0)
    lrPipeline_h2 = Pipeline(stages=[vecAssembler, lr_h2])
    lrModel_h2 = lrPipeline_h2.fit(trainingData)

    def evaluate_h2(data, model):
        predictions = model.transform(data)
        predictions. \
            select('national_identifier', 'probability', 'gender', 'gender_Index', 'prediction'). \
            orderBy('probability', ascending=False). \
            show(n=10, truncate=30)
        evaluator = MulticlassClassificationEvaluator(labelCol='gender_Index', predictionCol='prediction')
        lrAcc = evaluator.evaluate(predictions)
        print('test accuracy = ', lrAcc)

    evaluate_h2(validData, lrModel_h2)
    lrPipeline_h2 = Pipeline(stages=[vecAssembler, lr_h2])
    lrModel_h2 = lrPipeline_h2.fit(trainingData)
    evaluate_h2(testData, lrModel_h2)

H2()


# H3. 姓名预测模型：假设给定一个人的所有信息（除了姓名），预测这个人最可能的姓氏。分析该 模型Top1到 Top 5的预测准确度；
def H3():
    print('=' * 20, 'problem H3', '=' * 20)
    feature_col = ['mother_first', 'father_first', 'birth_city', 'gender', 'year_of_birth', 'month_of_birth',
                   'id_registration_city', 'id_registration_district', 'address_city',
                   'address_district', 'address_neighborhood', 'street_address'
                   ]

    # 所有的特征列列名
    oheOutputCols = [x + '_OHE' for x in feature_col]

    # assemble all the feature columns
    vecAssembler = VectorAssembler(inputCols=oheOutputCols, outputCol='features')
    vecTrainDF_h3 = vecAssembler.transform(trainingData)
    trainingData.show(3)
    lr_h3 = LogisticRegression(featuresCol='features', labelCol='first_Index',
                               maxIter=100, regParam=0.01, elasticNetParam=0)
    # lrPipeline_h3 = Pipeline(stages = [vecAssembler,lr_h3])
    lrModel_h3 = lr_h3.fit(vecTrainDF_h3)

    def evaluate_h3(data):
        print(lrModel_h3)
        vecData = vecAssembler.transform(data)
        predictions = lrModel_h3.transform(vecData)
        predictions.select('national_identifier', 'probability', 'first', 'first_Index', 'prediction').orderBy(
            'probability', ascending=False).show(n=10, truncate=30)

        evaluator = MulticlassClassificationEvaluator(labelCol='first_Index', predictionCol='prediction')
        lrAcc = evaluator.evaluate(predictions)
        print('test accuracy = ', lrAcc)

    evaluate_h3(validData)
    evaluate_h3(testData)

# H3()

# TODO: H4. 人口预测模型：统计每一年出生的人数，预测下一年新增人口数。
from pyspark.sql.types import FloatType
from math import log
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

def H4():
    print('='*2,'problem H4','='*20)
    df_h4 = df_format_date.withColumn(
        'year_of_birth', year('date_of_birth'))
    df_population = df_h4.select("year_of_birth").groupBy('year_of_birth').agg(count('*').alias('total'))

    df_population = df_population.withColumn('year', df_population['year_of_birth'].cast('int')).drop('year_of_birth')
    df_population = df_population.filter(df_population['year'] > 1700)
    df_population.orderBy('total').show(10)

    def to_index(year):
        return year - 1888
    to_index_udf = udf(to_index, returnType=IntegerType())

    min_year = df_population.select(min('year').alias('year')).collect()[0]
    print(min_year)
    new_df = df_population.withColumn('index', to_index_udf(df_population['year']))
    new_df.show()

    (trianing, test) = new_df.randomSplit([0.8, 0.2], seed=2020)
    trianing.persist()
    test.persist()
    
    ### linear regression
    vecAssembler = VectorAssembler(inputCols=['index'],outputCol='features')
    vecTrainDF = vecAssembler.transform(trianing)
    lr_h4 = LinearRegression(featuresCol='features',labelCol='total')

    lrModel_h4 = lr_h4.fit(vecTrainDF)
    m = lrModel_h4.coefficients[0]
    b = lrModel_h4.intercept
    print(f"""The formula for the linear regression lines is num = {m:.2f}*index{b:.2f}""")

    vecTestDF = vecAssembler.transform(test)
    predictions = lrModel_h4.transform(vecTestDF)
    predictions.orderBy('prediction', ascending=False).show(5)

    regresssionEvaluator = RegressionEvaluator(predictionCol='prediction', labelCol='total', metricName='r2')
    r2 = regresssionEvaluator.evaluate(predictions)
    print(f"r2 is {r2}")
    
    ### LR with Malthus model
    def log_num(num):
        if num:
            return log(num)
        else:
            return 0

    log_num_udf = udf(log_num, returnType=FloatType())
    log_df = new_df.withColumn('logTotal', log_num_udf(new_df['total']))
    log_df.show()

    vecAssembler = VectorAssembler(inputCols=['index'], outputCol='features')
    lr_h4_log = LinearRegression(featuresCol='features', labelCol='logTotal')

    training_log = trianing.withColumn('logTotal', log_num_udf('total'))
    vecTrainDF_log = vecAssembler.transform(training_log)
    lrModel_h4_log = lr_h4_log.fit(vecTrainDF_log)
    m_log = lrModel_h4_log.coefficients[0]
    b_log = lrModel_h4_log.intercept
    print(f"""The formula for the linear regression lines is log(total) = {m_log:.3f}*index+{b_log:.3f}""")

    # test
    test_log = test.withColumn('logTotal', log_num_udf('total'))
    vecTestDF_log = vecAssembler.transform(test_log)
    predictions_log = lrModel_h4_log.transform(vecTestDF_log)
    predictions_log.orderBy('prediction', ascending=False).show(10)

    regresssionEvaluator = RegressionEvaluator(predictionCol='prediction', labelCol='logTotal', metricName='r2')
    r2_log = regresssionEvaluator.evaluate(predictions_log)
    print(f"r2 is {r2_log}")
H4()
