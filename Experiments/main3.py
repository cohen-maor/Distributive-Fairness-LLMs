import json
import os
from tqdm import tqdm
import pandas as pd
import time

from components import PromptGenerator, LLM

variation_dict = {
    'valuation': ['word', 'table', 'pipes'],
    'restriction': ['vanilla', 'best', 'rank'],
    'template': ['vanilla', 'person_centric', 'goods_centric'],
    'example': [False, True],
    'justification': ['vanilla', 'optional', 'mandatory']
}

# llm = LLM(family='mixtral')
# pg = PromptGenerator(question=1,valuation_type='word')
# num_agents, num_goods = pg.set_text_values()
# prompt1 = pg.generate_prompt1()
# prompt2 = pg.generate_prompt2()

# print(prompt1)
# print(prompt2)

# df_dict = {
#     'Text': [],
# }

# for g in range(num_goods):
#     df_dict[f'Good {chr(65+g)}'] = []

# try:
#     df_dict['Text'] = llm.makeLLMRequest(prompt1)
# except:
#     print('LLM request failed.')
#     time.sleep(5)

# print(df_dict['Text'])

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

variation = 'valuation'

auth_key_files = {
    'gemini': ['../Auth_keys/gemini_auth_key_3.txt'],
    'llama': ['../Auth_keys/llama_auth_key.txt'],
    'chatgpt': ['../Auth_keys/gpt_auth_key_2.txt'],
    'claude': ['../Auth_keys/claude_auth_key.txt'],
}

auth_file_num = 0

new_questions = [11,12,13,14,15,16,17,18,21,22,23,24,31,41,42,51,52,61,62,91]
old_questions = [i+1 for i in range(10)]
instruction_questions = [7,3,19,2]
new_expts = [32,43]
version_tests = [2,21]

llm_types = ['gemini', 'llama', 'chatgpt', 'claude']
type_to_model = {
    'gemini': ['gemini-1.5-pro'],
    'llama': ['llama3-70b-8192'],
    'chatgpt': ["gpt-4o"],
    'claude': ['claude-3-5-sonnet-20240620']
}
for llm_type in llm_types[2:]:
    if not os.path.exists(f'results/{llm_type}'):
        os.mkdir(f'results/{llm_type}')
    for model in type_to_model[llm_type][:1]:
        print(f"Model {model}")
        model_file = model.replace('.', '') if llm_type == 'chatgpt' else model
        if not os.path.exists(f'results/{llm_type}/{model_file}'):
            os.mkdir(f'results/{llm_type}/{model_file}')
        for question in [2,7]:
            if not os.path.exists(f'results/{llm_type}/{model_file}/question_{question}'):
                os.mkdir(f'results/{llm_type}/{model_file}/question_{question}')
            else:
                filename = f'results/{llm_type}/{model_file}/question_{question}/word_bad5.csv'
                pg = PromptGenerator(question=question, valuation_type='word', instruction='bad5', llm_type=llm_type)
                if os.path.exists(filename):
                    print(f'The file \"{filename}\" already exists!')
                    continue
                else:
                    print(f'Generating file \"{filename}\"!')
                num_agents, num_goods = pg.set_text_values()
                prompt1 = pg.generate_prompt1()

                print(prompt1)

                df_dict = {
                    'Text': [], 'Tokens': []
                }

                for g in range(num_goods):
                    df_dict[f'Good {chr(65+g)}'] = []
                if 'money' in pg.valuations:
                    for a in list(pg.valuations['valuation'].keys()):
                        df_dict[f'{pg.person_names[a]} money'] = []

                j = 0
                for i in tqdm(range(200)):
                    llm = LLM(auth_file=auth_key_files[llm_type][auth_file_num], family=llm_type, model=model)

                    try:
                        text = llm.makeLLMRequest(prompt1)
                    except:
                        auth_file_num += 1
                        auth_file_num %= len(auth_key_files[llm_type])
                        print(f'LLM request failed. Auth key file is now {auth_key_files[llm_type][auth_file_num]}')
                        time.sleep(5)
                        continue

                    try:
                        prompt2 = pg.generate_prompt2(text)
                        alloc = llm.makeLLMRequest(prompt2)
                    except:
                        auth_file_num += 1
                        auth_file_num %= len(auth_key_files[llm_type])
                        print(f'LLM request failed. Auth key file is now {auth_key_files[llm_type][auth_file_num]}')
                        time.sleep(5)
                        continue

                    bopen = alloc.find('{')
                    bclose = alloc.find('}')

                    try:
                        # print(alloc[bopen:bclose+1])
                        json_obj = json.loads(alloc[bopen:bclose+1])
                    except:
                        print('JSON read failed.')
                        print(alloc[bopen:bclose+1])
                        continue
                    
                    try:
                        flag = 0
                        for g in range(num_goods):
                            if type(json_obj[f'Good {chr(65+g)}']) != str:
                                flag = 1
                                break

                        if flag:
                            continue

                        money_dict = {}

                        if 'money' in pg.valuations:
                            for a in list(pg.valuations['valuation'].keys()):
                                money_dict[f'{pg.person_names[a]} money'] = json_obj[f'{pg.person_names[a]} money']

                        for key in money_dict:
                            df_dict[key].append(money_dict[key])

                        for g in range(num_goods):
                            df_dict[f'Good {chr(65+g)}'].append(json_obj[f'Good {chr(65+g)}'].split()[-1])
                        
                        df_dict['Text'].append(text)
                        df_dict['Tokens'].append(llm.get_tokens_used())

                    except:
                        print('JSON format incomatible.')
                        print(json_obj)
                        continue
            
                    j += 1

                    if j == 100: 
                        data = pd.DataFrame.from_dict(df_dict)
                        data.to_csv(filename)
                        break
            



