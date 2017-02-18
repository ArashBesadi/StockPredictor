import java.util

import com.typesafe.scalalogging.Logger
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._

object StockPredictor {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("StockPredictor")
      .getOrCreate()

    val stockTestErrorRate = calculateStockTestErrorRate(spark)
    var stockPredictionAccuracy = 1 - stockTestErrorRate

    stockPredictionAccuracy = BigDecimal(stockPredictionAccuracy).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
    val logger = Logger[StockPredictor.type]
    logger.info("Stock Prediction Accuracy with Logistic Regression: " + BigDecimal(stockPredictionAccuracy) * 100 + "%")
  }

  private def calculateStockTestErrorRate(spark: SparkSession): Double = {

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array[String]("Lag1", "Lag2"))
      .setOutputCol("features")

    val indexer = new StringIndexer()
      .setInputCol("Direction")
      .setOutputCol("label")

    val stockSchema = createStockScheme()
    val stockSet = readStockDataset(spark, stockSchema)

    val stockTrainingSet = stockSet.select("*").where("Year < 2005")
    val stockTestSet = stockSet.select("*").where("Year == 2005")

    val stockTrainingFeatures = createStockFeatures(vectorAssembler, indexer, stockTrainingSet)
    val stockTestFeatures = createStockFeatures(vectorAssembler, indexer, stockTestSet)

    val logisticRegressionModel = trainLogisticRegressionModel(stockTrainingFeatures)

    val stockProbabilityVector = testLogisticRegressionModel(logisticRegressionModel, stockTestFeatures)

    val stockDirectionPrediction = evaluateStockProbability(stockProbabilityVector)

    evaluateStockDirectionPrediction(stockSet, stockDirectionPrediction)
  }

  private def createStockScheme(): StructType = {

    StructType(
      StructField("Year", IntegerType) ::
        StructField("Lag1", DoubleType) ::
        StructField("Lag2", DoubleType) ::
        StructField("Lag3", DoubleType) ::
        StructField("Lag4", DoubleType) ::
        StructField("Lag5", DoubleType) ::
        StructField("Volume", DoubleType) ::
        StructField("Today", DoubleType) ::
        StructField("Direction", StringType) :: Nil)
  }

  private def readStockDataset(spark: SparkSession, stockSchema: StructType): DataFrame = {

    val stockDatasetPath = "Smarket.csv"
    spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .schema(stockSchema)
      .load(stockDatasetPath)
  }

  private def createStockFeatures(vectorAssembler: VectorAssembler, indexer: StringIndexer, stockDataset: DataFrame): DataFrame = {

    val stockVector = vectorAssembler.transform(stockDataset)
    indexer.fit(stockVector).transform(stockVector).select("features", "label")
  }

  private def trainLogisticRegressionModel(stockTrainingFeatures: DataFrame): LogisticRegressionModel = {

    val logisticRegression = new LogisticRegression().setMaxIter(10)
    logisticRegression.fit(stockTrainingFeatures)
  }

  private def testLogisticRegressionModel(logisticRegressionModel: LogisticRegressionModel, stockTestFeatures: DataFrame): DataFrame = {

    val logisticRegressionSummary = logisticRegressionModel.evaluate(stockTestFeatures)
    val predictions = logisticRegressionSummary.predictions
    predictions.select(logisticRegressionSummary.probabilityCol)
  }

  private def evaluateStockProbability(probabilityVector: DataFrame): util.ArrayList[String] = {

    val predictedStockDirection = new util.ArrayList[String]
    for (stockProbabilities <- probabilityVector.collect()) {

      val stockProbability = stockProbabilities.apply(0).asInstanceOf[DenseVector].apply(0)

      if (stockProbability > 0.5) {
        predictedStockDirection.add("Up")
      } else {
        predictedStockDirection.add("Down")
      }
    }
    return predictedStockDirection
  }

  private def evaluateStockDirectionPrediction(stockSet: DataFrame, predictedStockDirection: util.ArrayList[String]): Double = {

    var stockTestErrorSum = 0.0
    val actualStockDirections = stockSet.select("Direction").where("Year == 2005").collect()

    for (i <- actualStockDirections.indices) {

      val actualStockDirection = actualStockDirections(i).apply(0)

      if (!predictedStockDirection.get(i).equals(actualStockDirection)) {
        stockTestErrorSum += 1
      }
    }
    return stockTestErrorSum / actualStockDirections.length
  }
}
