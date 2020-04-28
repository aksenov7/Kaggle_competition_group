import expdataanalysis as eda
import pandas as pd
submission = eda.read_file('sample_submission.csv', type_file="csv")
#submission = eda.optimize_mem_usage(submission, True)
print(eda.get_mem_usage(submission, print_inf = False))
json_df = eda.read_file("Config", "json")

#json_df = eda.optimize_mem_usage(json_df)
#json_df


titanic_df = eda.read_file("train_titanic.csv")
print(eda.get_corr_matrix(titanic_df))
#eda.plot_corr_matrix(titanic_df)
#eda.plot_corr_matrix(titanic_df.drop(['PassengerId',  'Name', 'Ticket', 'Cabin'], axis=1))

eda.missing_data(titanic_df)
eda.count_types(titanic_df)

eda.plot_dependency_chart(titanic_df.loc[:, ['Age','PassengerId', 'Cabin','Fare']], 'Age', chart_in_str=2)
eda.plot_scatter_matrix(titanic_df.loc[:, ~titanic_df.columns.isin(['Name', 'PassengerId', 'Pclass','Embarked'])])
