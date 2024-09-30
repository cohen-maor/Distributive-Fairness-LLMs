from components import OutputAnalyzer, PromptGenerator
import os
import statistics
import itertools
from openpyxl import Workbook
import pandas as pd
import matplotlib.pyplot as plt



# dict_list = []
# new_questions = [11,12,13,14,15,16,17,21,22,23,24,31,41,42,51,61,91]
# old_questions = [i+1 for i in range(10)]
# for q in new_questions:
#     pg = PromptGenerator(q)

#     pg.set_valuations()

#     if not pg.valuations:
#         continue

#     valuations = pg.valuations['valuation']
#     # print(valuations)

#     table_dict = {
#         '\\textbf{Agent}': []
#     }
#     for i in range(len(valuations[1])):
#         table_dict['$\\boldsymbol{g_' + str(i+1) + '}$'] = []

#     for person in valuations:
#         if person >= 0:
#             person_string = '$\\boldsymbol{a_' + str(person+1) + '}$'
#             table_dict['\\textbf{Agent}'].append(person_string)
#         else:
#             table_dict['\\textbf{Agent}'].append("\\textbf{You}")
#         for i in range(len(valuations[1])):
#             table_dict['$\\boldsymbol{g_' + str(i+1) + '}$'].append(valuations[person][i])

#     # print(table_dict)
#     df = pd.DataFrame(table_dict)
#     # size = round(0.45*((len(valuations[1])+1)/7),1)
#     size=1
#     tabular = df.to_latex(index=False)
#     latex_string = '\\begin{table}[!htp]\n\centering\n'
#     # latex_string += '\\resizebox{'+str(size)+'\\textwidth}{!}{\n'
#     latex_string += tabular 
#     # latex_string += '}'
#     if 'money' in pg.valuations:
#         latex_string += '\caption{Valuation table for $I_{'+str(q)+'}$ (Money = '+str(pg.valuations['money'])+')}\n\label{tab:sc-'+str(q)+'}\n\end{table}'
#     else:
#         latex_string += '\caption{Valuation table for $I_{'+str(q)+'}$}\n\label{tab:sc-'+str(q)+'}\n\end{table}'
#     print(latex_string)
#     print()
#     dict_list.append(table_dict)

# fig, axes = plt.subplots(1,4,sharey=True, figsize=(8,3))

# notion_problems = {
#     'EF': [1,2,3,4,5,6,7,9,10],
#     'EQ': [2,4,5,6,7,9,10],
#     'USW': [1,2,3,4,5,6,9],
#     'RMM': [1,2,3,4,5,6,9],
# }

# for i, notion in enumerate(notion_problems):
#     oa = OutputAnalyzer(
#         model_paths=[
#             ('gemini/gemini-1.0-pro-latest', 'word_best_goods_centric'),
#             ('gemini/gemini-1.0-pro-latest', f'word_best_goods_centric_{notion.lower()}'), 
#             ('chatgpt/gpt-35-turbo-1106', 'word_mandatory'), 
#             ('chatgpt/gpt-35-turbo-1106', f'word_mandatory_{notion.lower()}')
#         ],
#         index=[
#             'Gemini-1.0',
#             f'Gemini-1.0 {notion}',
#             'GPT-3.5',
#             f'GPT-3.5 {notion}'
#         ]
#     )
#     gemini, gpt = oa.notion_detection(notion_problems[notion], notion=notion)

#     res = pd.DataFrame(zip(notion_problems[notion], gemini, gpt), columns = ['SC', 'Gemini', 'ChatGPT'])
#     res = res.set_index('SC')

#     ax = res.plot(kind = 'bar', ax=axes[i], legend=False)
#     ax.hlines(y=10, xmin=-1, xmax=len(notion_problems[notion]), linewidth=1, color='g')
#     ax.hlines(y=-10, xmin=-1, xmax=len(notion_problems[notion]), linewidth=1, color='r')
#     ax.set_xticklabels(notion_problems[notion], rotation = 0)
#     ax.set_xlabel(notion)
#     ax.set_ylabel('Number of Responses')

#     handles, labels = ax.get_legend_handles_labels()

#     print('Notion - ', notion)
#     print(res)

# fig.legend(handles, labels, loc='upper right')
# fig.savefig('notion_detection.pdf')
# fig.savefig('notion_detection.pdf')
# fig.suptitle('Increase in Allocations Satisfying the Required Notion', fontsize=20)

# oa = OutputAnalyzer()

# results = oa.get_rows()

# print(results.head())

# results.to_csv('strategy_comparison.csv')
# expressions = {
#     'nash bargaining': {},
#     'adjusted winner': {},
#     'maximizes the total value': {},
#     'envy-free': {},
#     'maximizes the total utility': {},
#     "maximin": {},
#     "max-min": {}
# }
# families = os.listdir('results')
# if '.DS_Store' in families: families.remove('.DS_Store')
# for family in families:
#     models = os.listdir(f'results/{family}')
#     if '.DS_Store' in models: models.remove('.DS_Store')
#     for model in models:
#         if model not in ('gpt-35-turbo-1106', 'gemini-1.0-pro-latest'): continue
#         questions = list(sorted(os.listdir(f'results/{family}/{model}')))
#         for question in questions:
#             if 'question' not in question: continue
#             q = int(question.split('_')[1])
#             if q > 10: continue
#             filename = f'results/{family}/{model}/{question}/word.csv'
#             data = pd.read_csv(filename)
#             for just in data['Text']:
#                 for exp in expressions:
#                     if question not in expressions[exp]:
#                         expressions[exp][question] = {}
#                     if model not in expressions[exp][question]:
#                         expressions[exp][question][model] = 0

#                     if exp in just.lower():
#                         expressions[exp][question][model] += 1

# model_paths = []
# question_nums = []
# expression_texts = []
# frequencies = []

# for exp in expressions:
#     for ques in expressions[exp]:
#         for model in expressions[exp][ques]:
#             model_paths.append(model)
#             question_nums.append(ques)
#             expression_texts.append(exp)
#             frequencies.append(expressions[exp][ques][model])

# data = pd.DataFrame(zip(model_paths,question_nums,expression_texts,frequencies), columns=['Model', 'Question', 'Expressions', 'Frequency'])
# data.to_csv('explanation_word_count.csv')
# print(data)

# oa = OutputAnalyzer(
#     model_paths=[
#         ('gemini/gemini-1.0-pro-latest', 'word'), 
#         ('chatgpt/gpt-35-turbo-1106', 'word')
#     ],
#     index=[
#         'Gemini-1.0',
#         'GPT-3.5'
#     ]
# )
# oa.plot_tweak_comparison([14,15,16,17], title='Variations of SC-1 having Greater Inequality')

# data = oa.get_data('gemini/gemini-1.0-pro-latest',14)
# data = data[data['Person 1 money']=='25']
# print(data)

# data = oa.get_data(model_path='gemini/gemini-1.0-pro-latest', question=1, variation='word_best_goods_centric')
# columns = list(data.columns[1:])
# if 'Tokens' in columns: columns.remove('Tokens')
# grouped = data.groupby(columns).size().sort_values(ascending=False)
# vals = [int(val) for val in grouped.values]
# l = [1,2,3,4,5]
# print(vals)
# print(statistics.stdev(vals))



# data = oa.get_data(variation='word_best')

# columns = list(data.columns[1:])
# grouped = data.groupby(columns).size().sort_values(ascending=False)[:5]
# print(grouped)
# variance = 0
# for val in grouped:
#     variance += val**2
# print(variance)


# data = oa.get_data(variation='word_best_goods_centric')

# columns = list(data.columns[1:])
# grouped = data.groupby(columns).size().sort_values(ascending=False)[:5]
# print(grouped)
# variance = 0
# for val in grouped:
#     variance += val**2
# print(variance)