```python
# Module 5: Big Data and Data Engineering

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pymongo import MongoClient

class BigDataDataEngineering:
    def __init__(self, data):
        self.spark = SparkSession.builder.appName('BigDataDataEngineering').getOrCreate()
        self.data = self.spark.read.csv(data, header=True, inferSchema=True)

    def data_preprocessing(self):
        # String Indexing for categorical columns
        indexer = StringIndexer(inputCol='category', outputCol='categoryIndex')
        self.data = indexer.fit(self.data).transform(self.data)

        # Vector Assembling for numerical columns
        assembler = VectorAssembler(inputCols=['feature1', 'feature2', 'feature3'], outputCol='features')
        self.data = assembler.transform(self.data)

        return self.data

    def model_training(self):
        # Split the data into training and testing sets
        train_data, test_data = self.data.randomSplit([0.7, 0.3])

        # Define the model
        lr = LogisticRegression(featuresCol='features', labelCol='categoryIndex')

        # Define the pipeline
        pipeline = Pipeline(stages=[lr])

        # Train the model
        model = pipeline.fit(train_data)

        return model

    def data_streaming(self, topic):
        # Create a StreamingContext
        ssc = StreamingContext(self.spark.sparkContext, 1)

        # Create a Kafka Stream
        kafkaStream = KafkaUtils.createStream(ssc, 'localhost:2181', 'spark-streaming', {topic:1})

        # Process the stream
        kafkaStream.pprint()

        # Start the streaming context
        ssc.start()
        ssc.awaitTermination()

    def data_storage(self, database, collection):
        # Connect to MongoDB
        client = MongoClient('localhost', 27017)
        db = client[database]

        # Convert Spark DataFrame to Pandas
        pandas_df = self.data.toPandas()

        # Store the data in MongoDB
        db[collection].insert_many(pandas_df.to_dict('records'))

if __name__ == "__main__":
    bdde = BigDataDataEngineering('data.csv')
    processed_data = bdde.data_preprocessing()
    model = bdde.model_training()
    bdde.data_streaming('topic')
    bdde.data_storage('database', 'collection')
```
