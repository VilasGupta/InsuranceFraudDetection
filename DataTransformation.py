from datetime import datetime
from os import listdir
import pandas
from application_logging.logger import App_Logger


class dataTransform:


     def __init__(self):
          self.goodDataPath = "Training_Raw_files_validated/Good_Raw"
          self.logger = App_Logger()


     def replaceMissingWithNull(self):

          log_file = open("Training_Logs/dataTransformLog.txt", 'a+')
          try:
               onlyfiles = [f for f in listdir(self.goodDataPath)]
               for file in onlyfiles:
                    data = pandas.read_csv(self.goodDataPath + "/" + file)
                    # list of columns with string datatype variables
                    columns = ["policy_bind_date","policy_state","policy_csl","insured_sex","insured_education_level","insured_occupation","insured_hobbies","insured_relationship","incident_state","incident_date","incident_type","collision_type","incident_severity","authorities_contacted","incident_city","incident_location","property_damage","police_report_available","auto_make","auto_model","fraud_reported"]

                    for col in columns:
                         data[col] = data[col].apply(lambda x: "'" + str(x) + "'")

                    data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                    self.logger.log(log_file, " %s: Quotes added successfully!!" % file)
          except Exception as e:
               self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
               #log_file.write("Current Date :: %s" %date +"\t" +"Current time:: %s" % current_time + "\t \t" + "Data Transformation failed because:: %s" % e + "\n")
               log_file.close()
          log_file.close()
