import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

spark = SparkSession.builder.appName("Pipeline data").getOrCreate()

def pipeline_data_consolidate(vib_AB_source, vib_C_source, vib_D_source, prog_mode_source, output_dest):
    """
    Processes milling modes vibrations data and consolidate the data.

    :param vib_AB_source: The URI where you can fin the vib_AB data.
    :param vib_C_source: The URI where you can fin the vib_C data.
    :param vib_D_source: The URI where you can fin the vib_D data.
    :param prog_mode_source: The URI where you can fin the prog_mode data.
    :param output_dest: The dest where output is written, such as 's3://datasciencerecrutement/consolidate_pyspark'.
    """

    # Load the data CSV 
    if (vib_AB_source is not None) and (vib_C_source) and (vib_D_source is not None) and (prog_mode_source is not None):
        vib_A_B_df = spark.read.option("header", "true").csv(vib_AB_source).toDF('date_AB', 'value_A', 'value_B').orderBy(asc('date_AB'))
        vib_C_df = spark.read.option("header", "true").csv(vib_C_source).toDF('date_C', 'value_C').orderBy(asc('date_C'))
        vib_D_df = spark.read.option("header", "true").csv(vib_D_source).toDF('date_D', 'value_D').orderBy(asc('date_D'))
        prog_df = spark.read.option("header", "true").csv(prog_mode_source).toDF('debut_prog_mode', 'programme', 'mode').orderBy(asc('debut_prog_mode'))


        #Join the vib_D and prog_mode data by date (s).
        vib_D_prog_df =  \
            vib_D_df.join(prog_df, vib_D_df.date_D==prog_df.debut_prog_mode, how = 'full')

        # Fill null with last known
        window_last = Window.orderBy("date_D")
        vib_D_prog_df = vib_D_prog_df.withColumn("debut_prog_mode", last("debut_prog_mode", ignorenulls=True).over(window_last)).\
            withColumn("programme", last("programme", ignorenulls=True).over(window_last)).\
            withColumn("mode", last("mode", ignorenulls=True).over(window_last)).orderBy(asc('debut_prog_mode'))
            
        #Join the vib_D and prog_mode data by date (min).
        vib_D_C_prog_df =  \
            vib_D_prog_df.join(vib_C_df, vib_D_prog_df['date_D']==vib_C_df['date_C'], how = 'full')

        # Fill null with last known               
        window_last = Window.orderBy("date_C")
        vib_D_C_prog_df = vib_D_C_prog_df.withColumn("date_D", last("date_D", ignorenulls=True).over(window_last)).\
            withColumn("value_D", last("value_D", ignorenulls=True).over(window_last)).orderBy(asc('date_C'))

        # Join AB to the df
        vib_ABCD_prog_df = \
            vib_A_B_df.\
            join(vib_D_C_prog_df, vib_A_B_df.date_AB.substr(1,19)==vib_D_C_prog_df['date_D'], how = 'full')

        #Write output consolidate
        vib_ABCD_prog_df.write.option("header", "true").mode("overwrite").csv(output_uri_consolidate)
            

    else :
        print("Missing sources")
        
def pipeline_data_clean(input_source, output_dest):
    """
    Clean consolidate data.

    :param input_source: The URI where you can fin the vib_AB data.
    :param output_dest: The URI where output is written, such as 's3://datasciencerecrutement/clean_pyspark'.
    """

    # Load the data CSV 
    if input_source is not None:
        vib_consol_df = spark.read.option("header", "true").csv(input_source).orderBy(asc('date_C'))

        #Drop null values
        vib_clean_df = vib_consol_df.dropna()
        #Write output clean
        vib_clean_df.write.option("header", "true").mode("overwrite").csv(output_dest)       

    else :
        print("Missing sources")

def pipeline_data_feature(input_source, output_dest):
    """
    Process the clean data to get features.

    :param input_source: The URI where you can fin the cleaned data.
    :param output_dest: The dest where output feature is written, such as 's3://datasciencerecrutement/feature_pyspark'.
    """
    
    # Load the data CSV 
    if input_source is not None:
        #Create the schema and load the data
        schema = StructType([ StructField("date_AB",StringType()), StructField("value_A",DoubleType()), StructField("value_B",DoubleType()), \
                     StructField("date_D",StringType()), StructField("value_D",DoubleType()), StructField("debut_prog_mode",StringType()), \
                     StructField("programme",StringType()), StructField("mode",StringType()), StructField("date_C",StringType()),\
                     StructField("value_C",DoubleType())])
        vib_clean_df = spark.read.option("header", "true").csv(input_source, schema).orderBy(asc('date_D'))

        #Lets hot encode the prog_mode
        #Concat programme and mode
        clean_1_df=vib_clean_df.select('date_AB', \
                    'value_A', 'value_B', 'date_D', 'value_D',\
                    'date_C', 'value_C','debut_prog_mode', \
                    concat(vib_clean_df.programme,vib_clean_df.mode).alias("prog_mode"))
        #Lets begin by create a numeric index for the field prog_mode
        prog_mode_indexer = StringIndexer(inputCol="prog_mode", outputCol="prog_mode_Index") #Fits a model to the input dataset with optional parameters.
        clean_2_df = prog_mode_indexer.fit(clean_1_df).transform(clean_1_df)
        
        #onehotencoder to prog_mode_Index
        onehotencoder_prog_mode_vector = OneHotEncoder(inputCol="prog_mode_Index", outputCol="prog_mode_vec")
        clean_3_df = onehotencoder_prog_mode_vector.fit(clean_2_df).transform(clean_2_df)

        #Lets assemble the data in a vector
        assembler = VectorAssembler(
            inputCols=["value_A", "value_B", "value_C", "value_D"], 
            outputCol="features_vector"
            )
        clean_4_df = assembler.transform(clean_3_df)
        clean_5_df = clean_4_df.select("date_AB","features_vector", "prog_mode_vec")
        
        #Normalisation des values
        scaler = StandardScaler(inputCol="features_vector", outputCol="norm_features",
                        withStd=True, withMean=True)
        # Compute summary statistics by fitting the StandardScaler
        clean_6_df = scaler.fit(clean_5_df).transform(clean_5_df).select("date_AB", "norm_features", "prog_mode_vec", "prog_mode")

        #From vector values to column values
        feature_df = clean_6_df.withColumn("values", vector_to_array("norm_features")).select(["date_AB"] + [col("values")[i] for i in range(4)] +  ["prog_mode_vec", "prog_mode"])
        
        #Write output feature
        feature_df.write.option("header", "true").mode("overwrite").csv(output_dest)       

    else :
        print("Missing sources")

          
if __name__ == "__main__":
    #get the args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket_name', help="The name of the used bucket.")
    args = parser.parse_args()
    
    #Definition des sources
    bucket_path = 's3://{args.bucket_name}'
    vib_AB_source = f'{bucket_path}/vibration_axis_A_axis_B.csv'   
    vib_C_source = f'{bucket_path}/vibration_axis_C.csv'    
    vib_D_source = f'{bucket_path}/vibration_axis_D.csv'
    prog_mode_source = f'{bucket_path}/vibration_axis_A_axis_B.csv'
    
    output_consolidate = f'{bucket_path}/pyspark_consolidate'
    
    """
    First STAGE :
                CONSOLIDATE DATA
    """
    pipeline_data_consolidate(vib_AB_source, vib_C_source, vib_D_source, prog_mode_source, output_consolidate)

    
    output_clean = f'{bucket_path}/pyspark_clean'
    """
    Second STAGE :
                CLEAN DATA
    """    
    pipeline_data_clean(output_consolidate, output_clean)
    
    output_feature = f'{bucket_path}/pyspark_feature'
    
    """
    Third STAGE :
                PREPARE FEATURE
    """  
    pipeline_data_feature(output_clean, output_feature)
    