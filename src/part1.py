# -*- coding: utf-8 -*-
# @Time    : 2021/5/15 12:06
# @Author  : Xiao Lulu

import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import re
from pyspark.sql.types import StringType, StructField, StructType, IntegerType
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.linalg import Vectors

spark = (SparkSession
         .builder
         .appName("mernis")
         .getOrCreate())
# 加载数据
file_path = '/root/myfile/mernis/data_dump.sql'

data = spark.sparkContext.textFile(file_path). \
    filter((lambda line: re.findall('^\d{6}', line))). \
    map(lambda line: line.split('\t')[:-1])

schema = "uid STRING, national_identifier STRING, first STRING, last STRING, mother_first STRING, " \
         "father_first STRING, gender STRING, birth_city STRING, date_of_birth STRING, " \
         "id_registration_city STRING, id_registration_district STRING, address_city STRING, " \
         "address_district STRING, address_neighborhood STRING,street_address STRING, " \
         "door_or_entrance_number STRING"

df = spark.createDataFrame(data, schema)
total_count = df.count()  # total_count = 49611709
print("The total number:", total_count)
# 展示10条数据
df.show(10)


# E1 统计土耳其所有公民中年龄最大的男性；

# Convert the date to a standard format, dd/mm/yyyy and add 0s in front of the strings.
def format_date(line):
    li = line.split('/')
    if len(li[2]) == 4 and 0 < len(li[1]) <= 2 and 0 < len(li[1]) <= 2:
        return li[2] + '-' + li[1].zfill(2) + '-' + li[0].zfill(2)
    else:
        return 'null'


format_date_udf = udf(format_date, returnType=StringType())

df.createOrReplaceTempView('citizens')
res1 = df.withColumn("date_of_birth", format_date_udf(df["date_of_birth"]))
res2 = res1.filter(expr("""date_of_birth != 'null'"""))
df_format_date = res2.withColumn('date_of_birth', to_date('date_of_birth'))
df_format_date.show(10)
# 筛选出男性
def oldest_men():
    male = df_format_date.filter(expr(""" gender='E' """))
    male.createOrReplaceTempView('male')
    oldest_men = spark.sql('SELECT * FROM male WHERE date_of_birth = '
                           '(SELECT min(date_of_birth) FROM male)' )
    print("The oldest men:", oldest_men.collect())
oldest_men()


# E2 统计所有姓名中最常出现的字母；
def letter_most_name():
    full_name_rdd = df.select(lower('first'), lower('last')).rdd
    res = full_name_rdd.\
        flatMap(lambda line: list((line[0] + line[1]))). \
        map(lambda letter: (letter, 1)). \
        reduceByKey(lambda a, b: a + b). \
        filter(lambda x: x[0].isalpha()). \
        repartition(1). \
        max(lambda x: x[1])
    print('出现最多的字母是：')
    print(res)

letter_most_name()

# E3 统计该国人口的年龄分布，年龄段分（0-18、19-28、29-38、39-48、49-59、>60）
def age_group(age):
    if not isinstance(age, float):
        return 'NULL'
    if 0 < age <= 18:
        return '0-18'
    elif 18 < age <= 28:
        return '19-28'
    elif 28 < age <= 38:
        return '29-39'
    elif 38 < age <= 48:
        return '39-48'
    elif 48 < age <= 59:
        return '49-59'
    elif 59 < age < 200:
        return '>60'
    else:
        return 'NULL'


age_udf = udf(age_group, returnType=StringType())


def age():
    df_age = df_format_date.withColumn('age', (round(months_between(
        to_date(lit('2009-12-31')), col('date_of_birth')) / 12, 2)).cast('float'))
    print(df_age.schema)
    df_age.show()
    df_age_group = df_age.withColumn('age_group', age_udf(df_age['age']))
    dist = df_age_group.groupBy('age_group'). \
        agg(count('*').alias('total_number'), round((count('*') / total_count), 3).alias('proportion'))
    dist.show()


age()

# E4. 分别统计该国的男女人数，并计算男女比例；
def male_female():
    num_male_female = df.select('gender').groupBy('gender').agg(count('*').alias('total'))
    num_male_female.show()
    num_male_female.createOrReplaceTempView('tc')
    male_num = spark.sql("""SELECT t.total FROM tc t WHERE t.gender='E'""").collect()
    female_num = spark.sql("""SELECT t.total FROM tc t WHERE t.gender='K'""").collect()
    print('Number of males is {}, number of females is {}'.format(male_num[0][0], female_num[0][0]))
    ratio = male_num[0][0] / female_num[0][0]
    print('male to female ratio', ratio)


male_female()

# E5. 统计该国男性出生率最高的月份和女性出生率最高的月份；

def birth_rate():
    df1 = df_format_date.select('gender', 'date_of_birth')
    df_month = df1.select('gender', month('date_of_birth').alias('birth_month'))
    male = df_month.filter(expr("""gender = 'E'""")). \
        groupBy('birth_month'). \
        agg(count('*').alias('count')). \
        rdd. \
        max(lambda line: line[-1])

    female = df_month.filter(expr("""gender = 'K'""")). \
        groupBy('birth_month'). \
        agg(count('*').alias('count')). \
        rdd. \
        max(lambda line: line[-1])
    print("The month with the highest male birth rate:", male)
    print("The month with the highest female birth rate：", female)

birth_rate()

# E6.  统计哪个街道居住人口最多。

def street():
    df_street = df.select('street_address').groupBy('street_address'). \
        agg(count('*').alias('total'))
    df_street.createOrReplaceTempView('df_street')
    res = spark.sql('select street_address, total as total from df_street t \
    where total = (select MAX(total) from df_street)')
    res.show()


street()

# N1 分别统计男性和女性中最常见的10个姓
def last_name():
    df_n1 = df.select('gender', 'last')
    male = df_n1.filter("""gender = 'E'""")
    male.groupBy('last'). \
        agg(count('*').alias('total')). \
        orderBy('total', ascending=False). \
        show(10)

    female = df_n1.filter("""gender = 'K'""")
    female.groupBy('last'). \
        agg(count('*').alias('total')). \
        orderBy('total', ascending=False). \
        show(10)


last_name()

# N2. 统计每个城市市民的平均年龄，统计分析每个城市的人口老龄化程度，判断当前城市是否处于 老龄化社会
# （当一个国家或地区60岁以上老年人口占人口总数的10%，或65岁以上老年人口占人 口总数的7%，即意味着这个国家或地区的人口处于老龄化社会）；
def ave_age():
    df3 = df_format_date.select('address_city', 'date_of_birth')
    df_age = df3.withColumn('age', (round(months_between(to_date(lit('2009-12-31')),
                                                         col('date_of_birth')) / 12, 2)).cast('float')). \
        drop('date_of_birth')

    # df_age.groupBy('address_city').agg(avg('age').alias('ave_age'))

    # 增加两列，分别判断当前城市60岁以上老年人口占比，以及65岁以上老年人口占比。
    df_age.withColumn('gt_60', (col('age') >= 60).cast('int')). \
        withColumn('gt_65', (col('age') >= 65).cast('int'))
    # df3.select('address_city','age').\
    #     where(col('age') >= 60).\
    #     groupBy('address_city').\
    #     count().\
    #     orderBy('count',ascending=False).\
    #     show()
    # df3.show()

    df_age.createOrReplaceTempView('df_age')
    res1 = spark.sql("""select address_city, count(*) as count, avg(age) as avg_age, 
    sum(gt_60)/count(*) as gt_60, sum(gt_65) / count(*) as gt_65 
    from df_age group by address_city""")
    res2 = res1.selectExpr('address_city', 'count', 'round(avg_age,3) as avg_age',
                           'round(gt_60,3) as gt_60', 'round(gt_65,3) as gt_65')
    res3 = res2.withColumn('aging', (col('gt_60') > 0.1) | (col('gt_65') > 0.07))
    res3.show()

ave_age()

# N3. 计算一下该国前10大人口城市中，每个城市的人口生日最集中分布的是哪2个月；
def city10():
    df4 = df_format_date.select('address_city', month('date_of_birth').alias('birth_month'))
    df4.createOrReplaceTempView('df4')
    # 该国10大人口城市
    top10 = spark.sql(""" select address_city, count(*) as total
     from df4 group by address_city order by total desc limit 10""")
    top10.createOrReplaceTempView('top10')
    # 10大人口城市生日各月份总数
    res1 = spark.sql(""" select t1.address_city as address_city, t1.birth_month, count(*) as total
    from df4 t1 where t1.address_city in (select address_city from top10) 
    group by t1.address_city, t1.birth_month
    """)
    res1.show()
    # res1.createOrReplaceTempView('res1')
    # # 展示每个城市人口生日最集中分布的2个月

    window = Window.partitionBy(res1['address_city']).orderBy(res1['total'].desc())
    res1.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 2).show()


city10()

# N4. 统计该国前10大人口城市中，每个城市的前3大姓氏，并分析姓氏与所在城市是否具有相关性（相关性分析利用top10的数据分析即可）；
def last_top3():
    df5 = df_format_date.select('address_city', 'last')
    df5.createOrReplaceTempView('df5')
    # 该国10大人口城市
    top10 = spark.sql(""" select address_city, count(*) as total
         from df5 group by address_city order by total desc limit 10""")
    top10.createOrReplaceTempView('top10')
    # 10大人口城市各个姓氏的数量
    res1 = spark.sql(""" select t1.address_city as address_city, t1.last as last, 
    count(*) as total from df5 t1
    where t1.address_city in (select address_city from top10) group by address_city, last""")
    # res1.show()
    res1.createOrReplaceTempView('res1')
    # 1
    window = Window.partitionBy(res1['address_city']).orderBy(res1['total'].desc())
    res2 = res1.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 3)
    res2.show()

    # 卡方检验
    # 十个城市中所有姓
    # all_last_name = res1.select('last').distinct().collect()
    # all_last_name = [row['last'] for row in all_last_name]
    # 窄表变宽表
    # res = res1.groupBy('address_city'). \
    #     pivot('last', all_last_name). \
    #     agg(F.first('total', ignorenulls=True)). \
    #     fillna(0)
    # # res.show()
    # df_res = spark.createDataFrame(res, ['address_city', 'all_last_name'])
    # last_col_name = df_res.columns.remove('address_city')
    # #
    # assembler = VectorAssembler(
    #     inputCols=last_col_name,
    #     outputCol="vector_last_name")
    # vectorized_df = assembler.transform(df_res).select('address_city', 'vector_last_name')
    # r = ChiSquareTest.test(vectorized_df, "vector_last_name ", "address_city").head()
    # print(r)
    # print("pValues: " + str(r.pValues))
    # print("degreesOfFreedom: " + str(r.degreesOfFreedom))
    # print("statistics: " + str(r.statistics))


    city10 = top10.select('address_city').take(10)
    all_last_name = res1.select('last').distinct()
    li = []
    for idx, city in enumerate(city10):

        res2 = res1.\
            where(col('address_city') == city['address_city']).\
            select('*').\
            join(all_last_name,'last','right_outer').\
            fillna(0).\
            orderBy('last')

        vector_last_name = res2.select('total').rdd.map(lambda x: x[0]).collect()
        print(idx)
        vector_last_name = Vectors.dense(vector_last_name)
        res = [idx, vector_last_name]
        li.append(res)

    vectorized_df = spark.createDataFrame(li,['address_city','vector_last_name'])
    r = ChiSquareTest.test(vectorized_df, "vector_last_name", "address_city")
    # print("pValues: " + str(r.pValues))
    # print("degreesOfFreedom: " + str(r.degreesOfFreedom))
    # print("statistics: " + str(r.statistics))
    r.show()
last_top3()
