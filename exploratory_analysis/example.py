import expdataanalysis as eda
import pandas as pd
submission = eda.read_file('sample_submission.csv', type_file="csv")
#submission = eda.optimize_mem_usage(submission, True)
print(eda.get_mem_usage(submission, print_inf = False))
json_df = eda.read_file("Config", "json")
#json_df = eda.optimize_mem_usage(json_df)
#json_df
