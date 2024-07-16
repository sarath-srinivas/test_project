import pandera.pyspark as pa
import pyspark.sql.types as T
import json

from decimal import Decimal
from pyspark.sql import SparkSession
from pandera.pyspark import DataFrameModel

spark = SparkSession.builder.getOrCreate()


class PysparkPanderSchema(DataFrameModel):
    color: T.StringType() = pa.Field(isin=["red", "green", "blue"])
    length: T.IntegerType() = pa.Field(gt=10)


data = [("red", 4), ("blue", 11), ("purple", 15), ("green", 39)]

spark_schema := T.StructType(
    [
        T.StructField("color", T.StringType(), False),
        T.StructField("length", T.IntegerType(), False),
    ],
)

df = spark.createDataFrame(data, spark_schema)
df_out = PysparkPanderSchema.validate(check_obj=df)

print(df_out.pandera.errors)
