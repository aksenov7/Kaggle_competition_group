import expdataanalysis as eda
import pandas as pd
submission = eda.read_file('sample_submission.csv')
submission = eda.optimize_mem_usage(submission, True)
sales_df = eda.read_file('sell_prices.csv')
sales_df = eda.optimize_mem_usage(sales_df, True)

json_df = eda.read_file("Config.json")
json_df
