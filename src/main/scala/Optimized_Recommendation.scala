import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{StringIndexer, Tokenizer, HashingTF, IDF, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import java.lang.System.nanoTime
import org.apache.spark.sql.{SparkSession, DataFrame}

object Optimized_NLPEnhancedRecommendationSystem {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Random Forest + NLP Recommendation System")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")

    // Define a schema that matches your CSV file
    val schema = StructType(Array(
      StructField("review_id", StringType, true),
      StructField("user_id", StringType, true),
      StructField("business_id", StringType, true),
      StructField("stars", FloatType, true),
      StructField("useful", IntegerType, true),
      StructField("funny", IntegerType, true),
      StructField("cool", IntegerType, true),
      StructField("text", StringType, true)
    ))

    // Load Data with schema
    val reviewDf = spark.read
      .option("header", "true")
      .schema(schema)
      .csv("C:/Users/User/Downloads/Scala-Project/Scala_NLP_Recommendation-RF/processed_review.csv")

    // Filter out rows with null values in essential columns
    val filteredDf = reviewDf.filter($"user_id".isNotNull && $"business_id".isNotNull && $"stars".isNotNull)

    // Indexing user_id and business_id
    val userIndexer = new StringIndexer().setInputCol("user_id").setOutputCol("userId").fit(filteredDf)
    val businessIndexer = new StringIndexer().setInputCol("business_id").setOutputCol("businessId").fit(filteredDf)
    val indexedDf = userIndexer.transform(filteredDf).transform(businessIndexer.transform(_))

    // NLP Feature Extraction from Text
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val wordsData = tokenizer.transform(indexedDf)
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(1000)
    val featurizedData = hashingTF.transform(wordsData)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("textFeatures")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    // Assemble all features into a single vector
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("userId", "businessId", "textFeatures"))
      .setOutputCol("features")

    val finalDf = vectorAssembler.transform(rescaledData)
      .withColumn("rating", col("stars").cast("float"))
      .select("features", "rating")

    // Random Forest Regressor
    val randomForest = new RandomForestRegressor()
      .setFeaturesCol("features")
      .setLabelCol("rating")
      .setNumTrees(10)  // Set the number of trees
      .setMaxBins(1000) // Increase maxBins to accommodate the categorical features

    // Train the model
    val startTime = nanoTime()
    val model = randomForest.fit(finalDf)
    val endTime = nanoTime()
    val duration = (endTime - startTime) / 1e9d
    val predictions = model.transform(finalDf)
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    // List to store results
    var results = List[(Int, Double, Double)]() // (NumTrees, RMSE, Training Time)

    // Grid search over numTrees parameter
    val numTreesSeq = 3 to 50 by 2 

    for (numTrees <- numTreesSeq) {
      val randomForest = new RandomForestRegressor()
        .setFeaturesCol("features")
        .setLabelCol("rating")
        .setNumTrees(numTrees)
        .setMaxBins(1000)

      val startTime = nanoTime()
      val model = randomForest.fit(finalDf)
      val endTime = nanoTime()
      val duration = (endTime - startTime) / 1e9d

      val predictions = model.transform(finalDf)
      val rmse = evaluator.evaluate(predictions)

      results = results :+ (numTrees, rmse, duration)
    }

    // Convert the results list to a DataFrame
    val resultsDf = results.toDF("NumTrees", "RMSE", "TrainingTime")

    // Show sorted results
    resultsDf.sort($"RMSE".asc).show()
  }
}