import json
import os
from tqdm import tqdm
import pandas as pd
import time

from components import PromptGenerator, LLM

variation_dict = {
    'valuation_type': ['word', 'table', 'pipes'],
    'restriction': ['vanilla', 'best', 'rank'],
    'template': ['vanilla', 'person_centric', 'goods_centric'],
    'example': [False, True],
    'justification': ['vanilla', 'optional', 'mandatory']
}

# llm = LLM(auth_file='../Auth_keys/gemini_auth_key.txt', family='gemini', model='gemini-1.5-pro-latest')
# pg = PromptGenerator(question=1,valuation_type='word', restriction='best', justification='mandatory', llm_type='gemini')

llm = LLM(auth_file='../Auth_keys/claude_auth_key.txt', family='claude', model='claude-3-5-sonnet-20240620')
pg = PromptGenerator(question=2,valuation_type='word', instruction='USWp', llm_type='claude')
num_agents, num_goods = pg.set_text_values()
prompt1 = pg.generate_prompt1()

print(prompt1)

# df_dict = {
#     'Text': [],
# }

# for g in range(num_goods):
#     df_dict[f'Good {chr(65+g)}'] = []

# df_dict['Text'] = llm.makeLLMRequest(prompt1)
# prompt2 = pg.generate_prompt2(df_dict['Text'])
# print(prompt2)

# try:
#     alloc = llm.makeLLMRequest(prompt2)
# except:
#     print('LLM request failed.')
#     time.sleep(5)

# bopen = alloc.find('{')
# bclose = alloc.find('}')

# json_obj = json.loads(alloc[bopen:bclose+1])
# print(json_obj)
# for g in range(num_goods):
#     df_dict[f'Good {chr(65+g)}'].append(json_obj[f'Good {chr(65+g)}'].split()[-1])
# if 'money' in pg.valuations:
#     for a in list(pg.valuations['valuation'].keys()):
#         df_dict[f'{pg.person_names[a]} money'] = json_obj[f'{pg.person_names[a]} money']

# print(df_dict)

# variation = 'justification'

# auth_key_files = ['../Auth_keys/gemini_auth_key.txt', '../Auth_keys/gemini_auth_key_2.txt']
# auth_file_num = 0

# llm_types = ['gemini']
# type_to_model = {
#     'gemini': ['gemini-1.0-pro', 'gemini-1.0-pro-latest']
# }
# for llm_type in llm_types:
#     if not os.path.exists(f'results/{llm_type}'):
#         os.mkdir(f'results/{llm_type}')
#     for model in type_to_model[llm_type][1:]:
#         if not os.path.exists(f'results/{llm_type}/{model}'):
#             os.mkdir(f'results/{llm_type}/{model}')
#         for question in range(1,11):
#             if not os.path.exists(f'results/{llm_type}/{model}/question_{question}'):
#                 os.mkdir(f'results/{llm_type}/{model}/question_{question}')
#             else:
#                 variations = variation_dict[variation]
#                 for option in variations[1:]:
#                     filename = f'results/{llm_type}/{model}/question_{question}/word_best_goods_centric_{option}.csv'
#                     if os.path.exists(filename):
#                         print(f'The file \"results/{llm_type}/{model}/question_{question}/word_best_goods_centric_{option}.csv\" already exists!')
#                         continue
#                     else:
#                         print(f'Generating file \"results/{llm_type}/{model}/question_{question}/word_best_goods_centric_{option}.csv\"!')
#                     pg = PromptGenerator(question=question, valuation_type="word", restriction="best", template="goods_centric", justification=option)
#                     num_agents, num_goods = pg.set_text_values()
#                     prompt1 = pg.generate_prompt1()
#                     prompt2 = pg.generate_prompt2()

#                     print(prompt1)
#                     print(prompt2)

#                     df_dict = {
#                         'Text': [],
#                     }

#                     for g in range(num_goods):
#                         df_dict[f'Good {chr(65+g)}'] = []
#                     if 'money' in pg.valuations:
#                         for a in list(pg.valuations['valuation'].keys()):
#                             df_dict[f'{pg.person_names[a]} money'] = []

#                     j = 0
#                     for i in tqdm(range(150)):
#                         llm = LLM(auth_file=auth_key_files[auth_file_num], family=llm_type)

#                         try:
#                             text = llm.makeLLMRequest(prompt1)
#                         except:
#                             auth_file_num += 1
#                             auth_file_num %= len(auth_key_files)
#                             print(f'LLM request failed. Auth key file is now {auth_key_files[auth_file_num]}')
#                             time.sleep(5)
#                             continue

#                         try:
#                             alloc = llm.makeLLMRequest(prompt2)
#                         except:
#                             auth_file_num += 1
#                             auth_file_num %= len(auth_key_files)
#                             print(f'LLM request failed. Auth key file is now {auth_key_files[auth_file_num]}')
#                             time.sleep(5)
#                             continue

#                         bopen = alloc.find('{')
#                         bclose = alloc.find('}')

#                         try:
#                             # print(alloc[bopen:bclose+1])
#                             json_obj = json.loads(alloc[bopen:bclose+1])
#                         except:
#                             print('JSON read failed.')
#                             print(alloc[bopen:bclose+1])
#                             continue
                        
#                         try:
#                             if 'money' in pg.valuations:
#                                 for a in list(pg.valuations['valuation'].keys()):
#                                     df_dict[f'{pg.person_names[a]} money'].append(json_obj[f'{pg.person_names[a]} money'])

#                             for g in range(num_goods):
#                                 df_dict[f'Good {chr(65+g)}'].append(json_obj[f'Good {chr(65+g)}'].split()[-1])
                            
#                             df_dict['Text'].append(text)
#                         except:
#                             print('JSON format incomatible.')
#                             print(json_obj)
#                             # print(len(df_dict['Text']))
#                             # print(len(df_dict['Good A']))
#                             # print(len(df_dict['You money']))
#                             # print(df_dict['Good A'][-1])
#                             continue
                
#                         j += 1

#                         if j == 100: 
#                             # print(len(df_dict['Text']))
#                             # print(len(df_dict['Good A']))
#                             # print(len(df_dict['You money']))
#                             data = pd.DataFrame.from_dict(df_dict)
#                             data.to_csv(filename)
#                             break
            



