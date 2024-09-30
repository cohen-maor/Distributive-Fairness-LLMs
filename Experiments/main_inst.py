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

variation = 'valuation'

auth_key_files = {
    'gemini': ['../Auth_keys/gemini_auth_key_3.txt'],
    'llama': ['../Auth_keys/llama_auth_key_2.txt'],
    'chatgpt': ['../Auth_keys/gpt_auth_key_2.txt'],
    'claude': ['../Auth_keys/claude_auth_key.txt']
}

auth_file_num = 0

llm_types = ['gemini', 'llama', 'chatgpt', 'claude']
type_to_model = {
    'gemini': ['gemini-1.5-pro', 'gemini-1.0-pro-latest', 'gemini-1.5-pro-latest'],
    'llama': ['llama3-70b-8192', 'llama-13b-chat', 'llama-70b-chat', 'mixtral-8x7b-instruct'],
    'chatgpt': ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
    'claude': ['claude-3-5-sonnet-20240620']
}
for llm_type in ['claude']:
    if not os.path.exists(f'results/{llm_type}'):
        os.mkdir(f'results/{llm_type}')
    for model in type_to_model[llm_type][:1]:
        print(f"Model {model}")
        model_file = model.replace('.', '') if llm_type == 'chatgpt' else model
        if not os.path.exists(f'results/{llm_type}/{model_file}'):
            os.mkdir(f'results/{llm_type}/{model_file}')
        for question in [2, 3, 19]:
            if not os.path.exists(f'results/{llm_type}/{model_file}/question_{question}'):
                os.mkdir(f'results/{llm_type}/{model_file}/question_{question}')
            else:
                for instruction in ['RMM', 'RMMp', 'EF', 'EFp', 'USW', 'USWp']:
                    ins_string = instruction if not instruction else f'_{instruction}'
                    filename = f'results/{llm_type}/{model_file}/question_{question}/word{ins_string}.csv'
                    pg = PromptGenerator(question=question, valuation_type='word', llm_type=llm_type, instruction=instruction)
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
                    for i in tqdm(range(120)):
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

                            if 'money' in pg.valuations:
                                for a in list(pg.valuations['valuation'].keys()):
                                    df_dict[f'{pg.person_names[a]} money'].append(json_obj[f'{pg.person_names[a]} money'])

                            for g in range(num_goods):
                                df_dict[f'Good {chr(65+g)}'].append(json_obj[f'Good {chr(65+g)}'].split()[-1])
                            
                            df_dict['Text'].append(text)
                            df_dict['Tokens'].append(llm.get_tokens_used())
                        except:
                            print('JSON format incomatible.')
                            print(json_obj)
                            continue
                
                        j += 1
                        # if llm_type == "llama" and j == 5: 
                        #     data = pd.DataFrame.from_dict(df_dict)
                        #     data.to_csv(filename)
                        #     break

                        if j == 100: 
                            data = pd.DataFrame.from_dict(df_dict)
                            data.to_csv(filename)
                            break
            



