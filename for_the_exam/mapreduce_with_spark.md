```python
from pyspark.sql.functions import col, avg

# 1. Filter the dataset for the specific course
filtered_df = df.filter(col("course_name") == "Big Data Management")

# 2. Group by department and calculate the average grade
result_df = filtered_df.groupBy("department") \
                       .agg(avg("grade").alias("average_grade"))

# Display the results
result_df.show()
```

.filter(): Does exactly what our if statement did in the Mapper, keeping only the rows where the course name matches.

.groupBy("department"): Replaces the entire "Shuffle and Sort" phase of MapReduce, gathering all records for each department together.

.agg(avg("grade")): Replaces the Reducer logic. Spark has built-in aggregation functions like avg() that handle the math (including the sum and count logic we discussed earlier) automatically under the hood.

- Benefits of using spark api:
In-Memory Processing (Speed): This is Spark's biggest advantage. MapReduce writes intermediate data to physical hard disks after every map and reduce step. Spark processes data in RAM (in-memory) whenever possible, only spilling to disk if it runs out of memory. This makes Spark up to 100x faster for certain workloads, especially iterative algorithms (like machine learning).

Ease of Use and Richer APIs: As you can see in the code above, Spark's DataFrame API is much more concise and intuitive. You don't have to force your logic into strict Map and Reduce functions. Spark supports Java, Scala, Python, and R, and provides high-level operators for filtering, joining, and grouping.

Built-in Optimization (Catalyst Optimizer): When you write MapReduce code, the framework executes it exactly as written, even if it's inefficient. When you write Spark DataFrame code, Spark's "Catalyst Optimizer" analyzes your query, rearanges steps, and figures out the most efficient physical execution plan before it even runs.

Unified Ecosystem: MapReduce is strictly for batch processing. If you want to do stream processing, machine learning, or graph processing in the Hadoop ecosystem, you have to use separate tools. Spark has built-in libraries for all of these (Spark Streaming, MLlib, GraphX) that work seamlessly together.