import numpy as np
import random
import requests
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import replicate
import os
import statistics
from scipy.stats import chisquare, mannwhitneyu, fisher_exact


from llamaapi import LlamaAPI
from groq import Groq
from openai import OpenAI
import anthropic
from anthropic import AnthropicVertex
import pathlib
import textwrap
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import autogen
from autogen import AssistantAgent, OpenAIWrapper, UserProxyAgent, gather_usage_summary

from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM():
    def __init__(self, auth_file = '../Auth_keys/gemini_auth_key.txt', family='gemini', model='gemini-1.0-pro-latest') -> None:
        self.auth_file = auth_file
        self.family = family
        self.model = model
        self.tokens = 0
        if self.family == 'gemini':
            self.init_gemini()
        if self.family == 'mixtral':
            self.init_mixtral()
        if self.family == 'llama':
            self.init_llama()
        if self.family == "chatgpt":
            self.init_gpt()
        if self.family == 'claude':
            self.init_claude()

    def init_gemini(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()

        genai.configure(api_key=AUTHKEY)

        self.llm = genai.GenerativeModel(self.model)
        self.chat = self.llm.start_chat(history=[])

    def init_mixtral(self):
        model_id = "mistralai/Mixtral-8x7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.llm = AutoModelForCausalLM.from_pretrained(model_id)

    def init_llama(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()

        os.environ["GROQ_API_KEY"]=AUTHKEY

        self.llm = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    def init_gpt_autogen(self):
        self.config_list = autogen.config_list_from_json(
            "../Auth_keys/OAI_CONFIG_LIST",
            filter_dict={
                "model": ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
            },
        )
        self.llm = OpenAIWrapper(config_list=self.config_list)

    def init_gpt(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()
        os.environ["OPENAI_API_KEY"] = AUTHKEY

        client = OpenAI()
        self.llm = client.chat.completions

    def init_claude(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()

        self.llm = anthropic.Anthropic(api_key=AUTHKEY)

    def makeLLMRequest(self, queryText):
        if self.family == "gemini":
            response = self.chat.send_message(
                content=queryText,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            response = response.text
        elif self.family == "mixtral":
            inputs = self.tokenizer(queryText, return_tensors="pt")

            outputs = self.llm.generate(**inputs, max_new_tokens=20)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif self.family == 'llama':
            response = self.llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": queryText,
                    }
                ],
                model=self.model,
            )
            response = response.choices[0].message.content

        elif self.family == "chatgpt":
            messages = [
                {"role": "user", "content": queryText},
            ]
            response = self.llm.create(messages=messages, model=self.model)
            self.tokens += response.usage.total_tokens
            response = response.choices[0].message.content

        elif self.family == "claude":
            response = self.llm.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[
                        {"role": "user", "content": queryText}
                ]
            )
            response = response.content[0].text

        return response
    
    def get_tokens_used(self):
        return self.tokens


class PromptGenerator():
    def __init__(self, question = 1, valuation_type = 'word', restriction = 'vanilla', template = 'vanilla', example = False, justification = 'vanilla', llm_type='gemini', instruction='Fairest') -> None:
        self.question = question
        self.valuation_type = valuation_type
        self.restriction = restriction
        self.template = template
        self.example = example
        self.justification = justification
        self.instruction = instruction
        self.person_names = ['Person 1', 'Person 2', 'Person 3', 'You']
        self.good_names = ['Good A', 'Good B', 'Good C', 'Good D', 'Good E', 'Good F']
        self.llm_type = llm_type

    def set_valuations(self):
        if self.question == 1:
            self.valuations = {
               'valuation': {
                   0: [49, 46, 5],
                   1: [47, 48, 5]
               } 
            }
        elif self.question == 11: # Increasing differences between payoffs in EF allocation to see if Good C is still discarded.
            self.valuations = {
               'valuation': {
                   0: [80, 0, 20],
                   1: [20, 60, 20]
               } 
            }
        elif self.question == 12: # Making the value for Good C different for both agents to see if the model allocates it more willingly
            self.valuations = {
               'valuation': {
                   0: [80, 5, 15],
                   1: [20, 60, 20]
               } 
            }
        elif self.question == 13: # Reversing the order of agents to see if the all-notion allocation is chosen now
            self.valuations = {
               'valuation': {
                   0: [60, 20, 20],
                   1: [5, 80, 15]
               } 
            }       
        elif self.question == 18: # Reversing the order of agents to see if the all-notion allocation is chosen now
            self.valuations = {
               'valuation': {
                   0: [60, 20, 20],
                   1: [20, 80, 0]
               } 
            }
        elif self.question == 14: # 
            self.valuations = {
                'valuation': {
                    0: [90, 10],
                    1: [60, 40],
                },
                'money': 50
            }
        elif self.question == 15: 
            self.valuations = {
                'valuation': {
                    0: [80, 20],
                    1: [60, 40],
                },
                'money': 50
            }
        elif self.question == 16: 
            self.valuations = {
                'valuation': {
                    0: [70, 30],
                    1: [60, 40],
                },
                'money': 50
            }
        elif self.question == 17: 
            self.valuations = {
                'valuation': {
                    0: [60, 40],
                    1: [60, 40],
                },
                'money': 50
            }
        elif self.question == 19:
            self.valuations = {
               'valuation': {
                   0: [45, 20, 35],
                   1: [35, 40, 25]
               } 
            }
        elif self.question == 2:
            self.valuations = {
                'valuation': {
                    0: [5, 47, 45, 3],
                    1: [45, 5, 48, 2],
                    2: [23, 25, 32, 20]
                }
            }
        elif self.question == 21: # Changing the order of goods to see if the model identifies the EQ allocation if the first good available corresponds to that
            self.valuations = {
                'valuation': {
                    0: [45, 5, 47, 3],
                    1: [48, 45, 5, 2],
                    2: [32, 23, 25, 20]
                }
            }
        elif self.question == 22: # Increasing the differences in the previously EF allocation (no longer EF) to see if the model is nudged towards the EQ allocation
            self.valuations = {
                'valuation': {
                    0: [10, 60, 50, 10],
                    1: [50, 3, 75, 2],
                    2: [15, 30, 45, 20]
                }
            }
        elif self.question == 23: # Adding biased decision maker for Person 2's profile, self benefits from USW allocation
            self.valuations = {
                'valuation': {
                    -1: [45, 5, 48, 2],
                    1: [5, 47, 45, 3],
                    2: [23, 25, 32, 20]
                }
            }
        elif self.question == 24: # Self is Person 3's perspective, benefits from RMM allocation
            self.valuations = {
                'valuation': {
                    -1: [23, 25, 32, 20],
                    1: [5, 47, 45, 3],
                    2: [45, 5, 48, 2]
                }
            }
        elif self.question == 25: # Changing order of agents
            self.valuations = {
                'valuation': {
                    0: [23, 25, 32, 20],
                    1: [5, 47, 45, 3],
                    2: [45, 5, 48, 2],
                }
            }
        elif self.question == 26: # Changing order of items AND agents
            self.valuations = {
                'valuation': {
                    0: [32, 23, 25, 20],
                    1: [45, 5, 47, 3],
                    2: [48, 45, 5, 2],
                }
            }
        elif self.question == 3:
            self.valuations = {
                'valuation': {
                    0: [40, 2, 3, 25, 30],
                    1: [14, 26, 8, 26, 26],
                    2: [10, 26, 26, 12, 26]
                }
            }
        elif self.question == 31: # Adding a 6th good to see if EQ is achieved
            self.valuations = {
                'valuation': {
                    0: [40, 2, 3, 25, 30, 12],
                    1: [14, 26, 8, 26, 26, 12],
                    2: [10, 26, 26, 12, 26, 12]
                }
            }
        elif self.question == 32:
            self.valuations = {
                'valuation': {
                    0: [14, 8, 26, 26, 26],
                    1: [40, 3, 2, 30, 25],
                    2: [10, 26, 26, 26, 12]
                }
            }
        elif self.question == 4:
            self.valuations = {
                'valuation': {
                    0: [30, 31, 32, 7],
                    1: [33, 29, 31, 7],
                    2: [31, 32, 30, 7]
                }
            }
        elif self.question == 41: # Increasing the difference between EF and EQ allocations to see if support for EQ increases
            self.valuations = {
                'valuation': {
                    0: [20, 40, 65, 10],
                    1: [55, 30, 40, 10],
                    2: [40, 45, 40, 10]
                }
            }
        elif self.question == 42: # EF allocation is now only a weak improvement over the EQ allocation instead of a strong one
            self.valuations = {
                'valuation': {
                    0: [20, 40, 60, 10],
                    1: [50, 30, 40, 10],
                    2: [40, 40, 40, 10]
                }
            }
        elif self.question == 43: # Bring lowest valued good in the middle
            self.valuations = {
                'valuation': {
                    0: [30, 7, 32, 31],
                    1: [33, 7, 31, 29],
                    2: [31, 7, 30, 32]
                }
            }
        elif self.question == 5:
            self.valuations = {
                'valuation': {
                    0: [5, 20, 32, 3, 25, 15],
                    1: [26, 7, 23, 20, 2, 22],
                    2: [24, 17, 6, 21, 30, 2]
                }
            }
        elif self.question == 51: # Goods are rearranged to test if bias for RR increases support for EF+USW+RMM allocation
            self.valuations = {
                'valuation': {
                    0: [32, 20, 5, 15, 25, 3],
                    1: [23, 7, 26, 22, 2, 20],
                    2: [6, 17, 24, 2, 30, 21]
                }
            }
        elif self.question == 52: # EQ allocation no longer exists and USW allocation is more disparate
            self.valuations = {
                'valuation': {
                    0: [32, 20, 5, 15, 3, 25],
                    1: [7, 23, 26, 22, 2, 20],
                    2: [6, 17, 2, 24, 30, 21]
                }
            }
        elif self.question == 6:
            self.valuations = {
                'valuation': {
                    0: [48, 4, 3, 45],
                    1: [25, 20, 40, 15],
                    2: [2, 1, 45, 52]
                }
            }
        elif self.question == 61: # Moving low valued good to the end to see if discarding increases
            self.valuations = {
                'valuation': {
                    0: [48, 3, 45, 4],
                    1: [25, 40, 15, 20],
                    2: [2, 45, 52, 1]
                }
            }
        elif self.question == 62: # Moving low valued good to th end to see if discarding increases
            self.valuations = {
                'valuation': {
                    0: [25, 15, 40, 20],
                    1: [48, 45, 3, 4],
                    2: [2, 52, 45, 1]
                }
            }
        elif self.question == 63: 
            self.valuations = {
                'valuation': {
                    0: [25, 20, 40, 15],
                    1: [48, 4, 3, 45],
                    2: [2, 1, 45, 52]
                }
            }
        elif self.question == 64:
            self.valuations = {
                'valuation': {
                    0: [52, 45, 1, 2],
                    1: [45, 3, 4, 48],
                    2: [15, 40, 20, 25],
                }
            }
        elif self.question == 7:
            self.valuations = {
                'valuation': {
                    0: [45, 30, 25],
                    1: [35, 40, 25],
                    2: [50, 5, 45]
                },
                'money': 5
            }
        elif self.question == 71:
            self.valuations = {
                'valuation': {
                    0: [45, 30, 25],
                    1: [35, 40, 25],
                    2: [50, 5, 45]
                }
            }
        elif self.question == 8:
            self.valuations = {
                'valuation': {
                    0: [45, 4, 3, 48],
                    1: [15, 20, 40, 25],
                    2: [52, 1, 45, 2]
                },
                'money': 7
            }
        elif self.question == 81:
            self.valuations = {
                'valuation': {
                    0: [45, 4, 3, 48],
                    1: [15, 20, 40, 25],
                    2: [52, 1, 45, 2]
                }
            }
        elif self.question == 9:
            self.valuations = {
                'valuation': {
                    -1: [23, 40, 20, 17],
                    1: [2, 43, 1, 54],
                    2: [49, 4, 4, 43]
                }
            }
        elif self.question == 91: # Self benefits from USW allocation
            self.valuations = {
                'valuation': {
                    -1: [2, 43, 1, 54],
                    1: [23, 40, 20, 17],
                    2: [49, 4, 4, 43]
                }
            }
        elif self.question == 10:
            self.valuations = {
                'valuation': {
                    -1: [53, 3, 44],
                    1: [35, 36, 29],
                    2: [44, 30, 25]
                },
                'money': 9
            }
        elif self.question == 101:
            self.valuations = {
                'valuation': {
                    -1: [53, 3, 44],
                    1: [35, 36, 29],
                    2: [44, 30, 25]
                }
            }
        else:
            self.valuations = {}
        self.num_agents = len(self.valuations['valuation'])
        self.num_goods = len(self.valuations['valuation'][1])

    def set_person_list(self):
        if -1 not in self.valuations['valuation']:
            people = self.person_names[:self.num_agents]
        else:
            people = self.person_names[1:self.num_agents] + [self.person_names[-1]]
        self.person_list = f"{', '.join(people[:-1])}, and {people[-1]}"

    def set_goods_list(self):
        goods = self.good_names[:self.num_goods]
        self.goods_list = f"{self.num_goods} goods, namely {', '.join(goods[:-2])}, {goods[-2]}, and {goods[-1]}"

    def set_valuation_text(self):
        print(self.valuation_type)
        if self.valuation_type == "table":
            goods = self.good_names[:self.num_goods]
            self.valuation_text = f"Given below are each person's value for the given goods, in a comma separated values format.\n    {', '.join(goods)}\n"
            for i in list(self.valuations['valuation'].keys()):
                person_text = f"{self.person_names[i]}, "
                for j in range(self.num_goods):
                    if j == self.num_goods-1:
                        person_text += f"{self.valuations['valuation'][i][j]}\n"
                    else:
                        person_text += f"{self.valuations['valuation'][i][j]}, "
                self.valuation_text += person_text 
        
        elif self.valuation_type == "pipes":
            goods = self.good_names[:self.num_goods]
            self.valuation_text = f"Given below are each person's value for the given goods, in a tabular format.\n|          | {' | '.join(goods)} |\n"
            for i in list(self.valuations['valuation'].keys()):
                person_text = f"| {self.person_names[i]}{'     ' if i == -1 else ''} | "
                for j in range(self.num_goods):
                    if j == self.num_goods-1:
                        val = self.valuations['valuation'][i][j]
                        person_text += f"  {val}{' ' if val < 10 else ''}   |\n"
                    else:
                        val = self.valuations['valuation'][i][j]
                        person_text += f"  {val}{' ' if val < 10 else ''}   | "
                self.valuation_text += person_text 

        else:
            self.valuation_text = ""
            for i in list(self.valuations['valuation'].keys()):
                identifier  = f"{self.person_names[i]}'s" if i >= 0 else "Your"
                person_text = f"{identifier} value "
                for j in range(self.num_goods):
                    if j == self.num_goods-1:
                        person_text += f"and for {self.good_names[j]} is {self.valuations['valuation'][i][j]}.\n"
                    else:
                        person_text += f"for {self.good_names[j]} is {self.valuations['valuation'][i][j]}, "
                self.valuation_text += person_text
        if 'money' in self.valuations:
            self.valuation_text += f"A total of {self.valuations['money']} units of money are also available for allocation. This amount of money is worth exactly as much as a good of the same value, for each individual.\
 Since this is a divisible resource, parts of it can be allocated to different agents, although the total money allocated cannot exceed {self.valuations['money']} units.\n"

    def set_restriction_text(self):
        if self.restriction == 'best':
            self.restriction_text = 'Please restrict your response a single allocation that you think is fairest.\n'
        elif self.restriction == 'rank':
            self.restriction_text = 'Please provide the top-3 fair allocations, in order of how fair you think they are - the fairest allocation appearing first.\n'
        else:
            self.restriction_text = ''
    
    def set_template_text(self):
        plurality = 'each' if self.restriction == 'rank' else 'the'
        if self.template == 'person_centric':
            self.template_text = f"Please present {plurality} allocation you have selected in in the following JSON format:\n"
            self.template_text += "{\n"
            for i in list(self.valuations['valuation'].keys()):
                if 'money' in self.valuations:
                    self.template_text += f'\"{self.person_names[i]}\": \"[(<good(s) allocated to {self.person_names[i]}>), <money allocated to {self.person_names[i]}>]\",\n'
                else:
                    self.template_text += f'\"{self.person_names[i]}\": \"[<good(s) allocated to {self.person_names[i]}>]\",\n'
            self.template_text += "}\n"
            if self.example:
                if 'money' in self.valuations:
                    self.template_text += "For example, if Person X is allocated Goods G and H, and M units of money, the entry for Person X in the JSON would look like - \"Person X\": \"[(Good G, Good H), M]\"\n"
                else:
                    self.template_text += "For example, if Person X is allocated Goods G and H, the entry for Person X in the JSON would look like\n{\n\"Person X\": \"[Good G, Good H]\"\n}\n"
        elif self.template == 'goods_centric':
            self.template_text = f"Please present {plurality} allocation you have selected in in the following JSON format:\n"
            self.template_text += "{\n"
            for j in range(self.num_goods):
                self.template_text += f'\"{self.good_names[j]}\": \"<person to whom {self.good_names[j]} is allocated, \"None\" if {self.good_names[j]} is discarded>\",\n'
            if 'money' in self.valuations:
                for i in list(self.valuations['valuation'].keys()):
                    self.template_text += f'\"{self.person_names[i]} money\": \"<money allocated to {self.person_names[i]}, 0 if no money was allocated to {self.person_names[i]}>\",\n'
            self.template_text += "}\n"
            if self.example:
                if 'money' in self.valuations:
                    self.template_text += "For example, if Person X is allocated Goods G and H, and M units of money, the corresponding entries in the JSON would look like\n\
{\n\"Good G\": \"Person X\",\n\"Good H\": \"Person X\"\n,\"Person X money\": \"M\"\n}\n"
                else:
                    self.template_text += "For example, if Person X is allocated Goods G and H, the corresponding entries in the JSON would look like\n{\n\"Good G\": \"Person X\",\n\"Good H\": \"Person X\"\n}\n"
        else:
            self.template_text = ''

        last_comma = self.template_text.rfind(',')
        self.template_text = self.template_text[:last_comma] + self.template_text[last_comma+1:]        
    
    def set_justification_text(self):
        if self.justification == "optional":
            self.justification_text = "You may provide a justification for your choice if you find it necessary.\n"
        elif self.justification == "mandatory":
            self.justification_text = "You must provide a justification for your choice.\n"
        else:
            self.justification_text = ''
    
    def set_text_values(self):
        self.set_valuations()
        self.set_person_list()
        self.set_goods_list()
        self.set_valuation_text()
        self.set_restriction_text()
        self.set_template_text()
        self.set_justification_text()

        return self.num_agents, self.num_goods
    
    def get_top5(self):
        text = 'is to determine the allocation that you consider to be the fairest among the options given below:\n'
        if self.question == 2:
            text += "\n\
Allocation-1: Person 1 gets Good B, Person 2 gets Good C, and Person 3 gets Goods A and D.\n\
Allocation-2: Person 1 gets Good C, Person 2 gets Good A, and Person 3 gets Goods B and D.\n\
Allocation-3: Person 1 gets Good B, Person 2 gets Good A, and Person 3 gets Goods C and D.\n\
Allocation-4: Person 1 gets Good B, Person 2 gets Goods A and C, and Person 3 gets Good D.\n\
Allocation-5: Person 1 gets Good B, Person 2 gets Good A, Person 3 gets Good C, and Good D is discarded.\n\n"

        if self.question == 6:
            text += "\n\
Allocation-1: Person 1 gets Good A, Person 2 gets Goods B and C, and Person 3 gets Good D.\n\
Allocation-2: Person 1 gets Good D, Person 2 gets Goods A and B, and Person 3 gets Good C.\n\
Allocation-3: Person 1 gets Good A, Person 2 gets Good C, and Person 3 gets Good D.\n\
Allocation-4: Person 1 gets Good A, Person 2 gets Good B, and Person 3 gets Goods C and D.\n\n"

        elif self.question == 19:
            text += "\n\
Allocation-1: Person 1 gets Good A, and Person 2 gets Good B.\n\
Allocation-2: Person 1 gets Good C, and Person 2 gets Good A.\n\
Allocation-3: Person 1 gets Good A, and Person 2 gets Goods B and C.\n\
Allocation-4: Person 1 gets Goods A and C, and Person 2 gets Good B.\n\
Allocation-5: Person 1 gets Goods A and B, and Person 2 gets Good C.\n\n"

        elif self.question == 7:
            text += "\n\
Allocation-1: Person 1 gets Good A, and Person 2 gets Good B and 5 units of money, and Person 3 gets Good C.\n\
Allocation-2: Person 1 gets Good A, and Person 2 gets Good B, and Person 3 gets Good C and 5 units of money.\n\
Allocation-3: Person 1 gets Good A and 5 units of money, and Person 2 gets Good B, and Person 3 gets Good C.\n\
Allocation-4: Person 1 gets Good C, and Person 2 gets Good B, and Person 3 gets Good A and 5 units of money.\n\
Allocation-5: Person 1 gets 5 units of money, Person 2 gets Good B, and Person 3 gets Goods A and C.\n\n"
            
        elif self.question == 5:
            text += "\n\
Allocation-1: Person 1 gets Goods B and C, Person 2 gets Goods A and F, and Person 3 gets Goods D and E.\n\
Allocation-2: Person 1 gets Goods B and E, Person 2 gets Goods C and F, and Person 3 gets Goods A and D.\n\
Allocation-3: Person 1 gets Goods C and F, Person 2 gets Goods A and D, and Person 3 gets Goods B and E.\n\
Allocation-4: Person 1 gets Good C, Person 2 gets Good A, and Person 3 gets Good E.\n\n"
        
        text += "Please indicate the allocation you think is fairest and explain the reasons behind your choice."
        return text

    def get_options(self):
        text = 'is to determine the allocation that you consider to be the fairest among the options given below:\n'
        if self.question == 2:
            text += "\
Option 1:\n\
    {\n\
        Allocation: Person 1 gets Good B, Person 2 gets Good C, and Person 3 gets Goods A and D.\n\
        Payoffs: Person 1 gets 47 units of utility, Person 2 gets 48 units, and Person 3 gets 43.\n\
        Properties: This allocation is envy-free and Pareto-optimal, i.e no agents is envious of another\n\
        and there is no allocation where all agents are as well-off and at least one agent is strictly better-off.\n\
    }\n\
Option 2:\n\
    {\n\
        Allocation: Person 1 gets Good C, Person 2 gets Good A, and Person 3 gets Goods B and D.\n\
        Payoffs: Each Person gets 45 units of utility.\n\
        Properties: This allocation is equitable and satisfies the maximin principle, i.e. it ensures\n\
        perfect equality and maximizes the minimum payoff.\n\
    }\n\
Option 3:\n\
    {\n\
        Allocation: Person 1 gets Good B, Person 2 gets Good A, and Person 3 gets Goods C and D.\n\
        Payoffs: Person 1 gets 47 units of utility, Person 2 gets 45 units, and Person 3 gets 52.\n\
        Properties: This allocation satisfies the maximin principle and is Pareto-optimal, i.e. it maximizes the minimum payoff\n\
        and there is no allocation where all agents are as well-off and at least one agent is strictly better-off.\n\
    }\n\
Option 4:\n\
    {\n\
        Allocation: Person 1 gets Good B, Person 2 gets Goods A and C, and Person 3 gets Good D.\n\
        Payoffs: Person 1 gets 47 units of utility, Person 2 gets 93 units, and Person 3 gets 20.\n\
        Properties: This allocation maximizes the total utility and is Pareto-optimal, i.e. it maximizes the sum of payoffs\n\
        and there is no allocation where all agents are as well-off and at least one agent is strictly better-off.\n\
    }\n\
Option 5:\n\
    {\n\
        Allocation: Person 1 gets Good B, Person 2 gets Good A, Person 3 gets Good C, and Good D is discarded.\n\
        Payoffs: Person 1 gets 47 units of utility, Person 2 gets 48 units, and Person 3 gets 32.\n\
        Properties: This allocation tries give each agent the good they value the most. Since Good C is valued most.\n\
        by both Person 2 and Person 3, it is allocated to Person 3 to reduce inequality.\n\
    }\n"
            
        if self.question == 6:
            text += "\
Option 1:\n\
    {\n\
        Allocation: Person 1 gets Good A, Person 2 gets Goods B and C, and Person 3 gets Good D.\n\
        Payoffs: Person 1 gets 48 units of utility, Person 2 gets 60 units, and Person 3 gets 52.\n\
        Properties: This allocation is envy-free, satisfies the maximin priciple, and is Pareto-optimal. \n\
        This means that no agents is envious of another, the minimum payoff is maximized, \n\
        and there is no allocation where all agents are as well-off and at least one agent is strictly better-off.\n\
    }\n\
Option 2:\n\
    {\n\
        Allocation: Person 1 gets Good D, Person 2 gets Goods A and B, and Person 3 gets Good C.\n\
        Payoffs: Each Person gets 45 units of utility.\n\
        Properties: This allocation is equitable, i.e. it ensures perfect equality.\n\
    }\n\
Option 3:\n\
    {\n\
        Allocation: Person 1 gets Good A, Person 2 gets Good C, and Person 3 gets Good D.\n\
        Payoffs: Person 1 gets 48 units of utility, Person 2 gets 40 units, and Person 3 gets 52.\n\
        Properties: This allocation is envy-free, i.e. no agent is envious of another.\n\
    }\n\
Option 4:\n\
    {\n\
        Allocation: Person 1 gets Good A, Person 2 gets Good B, and Person 3 gets Goods C and D.\n\
        Payoffs: Person 1 gets 48 units of utility, Person 2 gets 20 units, and Person 3 gets 97.\n\
        Properties: This allocation maximizes the total utility and is Pareto-optimal, i.e. it maximizes the sum of payoffs\n\
        and there is no allocation where all agents are as well-off and at least one agent is strictly better-off.\n\
    }\n"
            
        if self.question == 19:
            text += "\
Option 1:\n\
    {\n\
        Allocation: Person 1 gets Good A, and Person 2 gets Good B.\n\
        Payoffs: Person 1 gets 45 units of utility, and Person 2 gets 40 units.\n\
        Properties: This allocation is envy-free, i.e no agents is envious of another.\n\
    }\n\
Option 2:\n\
    {\n\
        Allocation: Person 1 gets Good C, and Person 2 gets Good A.\n\
        Payoffs: Each Person gets 35 units of utility.\n\
        Properties: This allocation is equitable, i.e. it ensures perfectly equal payoffs.\n\
    }\n\
Option 3:\n\
    {\n\
        Allocation: Person 1 gets Good A, and Person 2 gets Goods B and C.\n\
        Payoffs: Person 1 gets 45 units of utility, and Person 2 gets 65 units.\n\
        Properties: This allocation satisfies the maximin principle and is Pareto-optimal, i.e. it maximizes the minimum payoff\n\
        and there is no allocation where all agents are as well-off and at least one agent is strictly better-off.\n\
    }\n\
Option 4:\n\
    {\n\
        Allocation: Person 1 gets Goods A and C, and Person 2 gets Good B.\n\
        Payoffs: Person 1 gets 80 units of utility, and Person 2 gets 40 units.\n\
        Properties: This allocation maximizes the total utility and is Pareto-optimal, i.e. it maximizes the sum of payoffs\n\
        and there is no allocation where all agents are as well-off and at least one agent is strictly better-off.\n\
    }\n\
Option 5:\n\
    {\n\
        Allocation: Person 1 gets Goods A and B, and Person 2 gets Good C.\n\
        Payoffs: Person 1 gets 65 units of utility, and Person 2 gets 25 units.\n\
        Properties: This allocation does not have any desirable property of fairness or efficiency.\n\
    }\n"
        
        if self.question == 7:
            text += "\
Option 1:\n\
    {\n\
        Allocation: Person 1 gets Good A, and Person 2 gets Good B and 5 units of money, and Person 3 gets Good C.\n\
        Payoffs: Each Person gets 45 units of utility.\n\
        Properties: This allocation is equitable and satisfies the maximin principle, i.e. it ensures\n\
        perfect equality and maximizes the minimum payoff.\n\
    }\n\
Option 2:\n\
    {\n\
        Allocation: Person 1 gets Good A, and Person 2 gets Good B, and Person 3 gets Good C and 5 units of money.\n\
        Payoffs: Person 1 gets 45 units of utility, Person 2 gets 40 units, and Person 3 gets 50.\n\
        Properties: This allocation is envy-free and Pareto-optimal, i.e no agents is envious of another\n\
        and there is no allocation where all agents are as well-off and at least one agent is strictly better-off.\n\
    }\n\
Option 3:\n\
    {\n\
        Allocation: Person 1 gets Good A and 5 units of money, and Person 2 gets Good B, and Person 3 gets Good C.\n\
        Payoffs: Person 1 gets 50 units of utility, Person 2 gets 40 units, and Person 3 gets 45.\n\
        Properties: This allocation is Pareto-optimal, i.e. there is no other allocation where all agents are as well-off\n\
        and at least one agent is strictly better-off.\n\
    }\n\
Option 4:\n\
    {\n\
        Allocation: Person 1 gets Good C, and Person 2 gets Good B, and Person 3 gets Good A and 5 units of money.\n\
        Payoffs: Person 1 gets 50 units of utility, Person 2 gets 40 units, and Person 3 gets 30.\n\
        Properties: This allocation does not have any desirable property of fairness or efficiency.\n\
    }\n\
Option 5:\n\
    {\n\
        Allocation: Person 1 gets 5 units of money, Person 2 gets Good B, and Person 3 gets Goods A and C.\n\
        Payoffs: Person 1 gets 5 units of utility, Person 2 gets 40 units, and Person 3 gets 95.\n\
        Properties: This allocation maximizes the total utility and is Pareto-optimal, i.e. it maximizes the sum of payoffs\n\
        and there is no allocation where all agents are as well-off and at least one agent is strictly better-off.\n\
    }\n"

        if self.question == 5:
            text += "\
Option 1:\n\
    {\n\
        Allocation: Person 1 gets Goods B and C, Person 2 gets Goods A and F, and Person 3 gets Goods D and E.\n\
        Payoffs: Person 1 gets 52 units of utility, Person 2 gets 48 units, and Person 3 gets 51.\n\
        Properties: This allocation is envy-free, satisfies the maximin priciple, maximizes the total utility, and is Pareto-optimal. \n\
        This means that no agents is envious of another, the minimum payoff is maximized, the sum of payoffs is also maximized, \n\
        and there is no allocation where all agents are as well-off and at least one agent is strictly better-off.\n\
    }\n\
Option 2:\n\
    {\n\
        Allocation: Person 1 gets Goods B and E, Person 2 gets Goods C and F, and Person 3 gets Goods A and D.\n\
        Payoffs: Each Person gets 45 units of utility.\n\
        Properties: This allocation is equitable, i.e. it ensures perfect equality.\n\
    }\n\
Option 3:\n\
    {\n\
        Person 1 gets Goods C and F, Person 2 gets Goods A and D, and Person 3 gets Goods B and E.\n\
        Payoffs: Person 1 gets 47 units of utility, Person 2 gets 46 units, and Person 3 gets 47.\n\
        Properties: This allocation is envy-free, i.e. no agent is envious of another.\n\
    }\n\
Option 4:\n\
    {\n\
        Allocation: Person 1 gets Good C, Person 2 gets Good A, and Person 3 gets Good E.\n\
        Payoffs: Person 1 gets 32 units of utility, Person 2 gets 26 units, and Person 3 gets 30.\n\
        Properties: This allocation is envy-free, i.e. no agent is envious of another.\n\
    }\n"
        
        text += "Please indicate the option you think is fairest and explain the reasons behind your choice."
        return text
    
    def get_bad5(self):
        text = 'is to determine the allocation that you consider to be the fairest among the options given below:\n'
        if self.question == 2:
            text += "\n\
Allocation-1: Person 1 gets Good B, Person 2 gets Good C, and Person 3 gets Good D.\n\
Allocation-2: Person 1 gets Goods A and B, Person 2 gets Good C, and Person 3 gets Good D.\n\
Allocation-3: Person 1 gets Good C, Person 2 gets Good A, and Person 3 gets Good B.\n\
Allocation-4: Person 1 gets Goods B and C, Person 2 gets Good A, and Person 3 gets Good D.\n\
Allocation-5: Person 1 gets Goods C and D, Person 2 gets Good A, and Person 3 gets Good B.\n\n"

        if self.question == 7:
            text += "\n\
Allocation-1: Person 1 gets Good C and 5 units of money, Person 2 gets Good B, and Person 3 gets Good A.\n\
Allocation-2: Person 1 gets Good C, Person 2 gets Good B and 5 units of money, and Person 3 gets Good A.\n\
Allocation-3: Person 1 gets Good C, Person 2 gets Good B, and Person 3 gets Good A and 5 units of money.\n\
Allocation-4: Person 1 gets Good B and 5 units of money, Person 2 gets Good A, and Person 3 gets Good C.\n\
Allocation-5: Person 1 gets Good B, Person 2 gets Good A and 5 units of money, and Person 3 gets Good C.\n\n"
        return text
    
    def get_human_choices(self):
        text = 'is to determine the allocation that you consider fairest. For your reference, human respondents \
chose the following allocations more frequently (with the percentage of responses corresponding to each allocation indicated in brackets):\n'
        if self.question == 2:
            text += "\n\
Allocation-1 (26.2% responses): Person 1 gets Good B, Person 2 gets Good C, and Person 3 gets Goods A and D.\n\
Allocation-2 (26.2% responses): Person 1 gets Good C, Person 2 gets Good A, and Person 3 gets Goods B and D.\n\
Allocation-3 (12.7% responses): Person 1 gets Good B, Person 2 gets Good A, and Person 3 gets Goods C and D.\n\
Allocation-4 (9.0% responses): Person 1 gets Good B, Person 2 gets Goods A and C, and Person 3 gets Good D.\n\
Allocation-5 (7.9% responses): Person 1 gets Good B, Person 2 gets Good A, Person 3 gets Good C, and Good D is discarded.\n\n"

        if self.question == 6:
            text += "\n\
Allocation-2 (32.6% responses): Person 1 gets Good D, Person 2 gets Goods A and B, and Person 3 gets Good C.\n\
Allocation-1 (28.1% responses): Person 1 gets Good A, Person 2 gets Goods B and C, and Person 3 gets Good D.\n\
Allocation-3 (18.4% responses): Person 1 gets Good A, Person 2 gets Good C, and Person 3 gets Good D.\n\
Allocation-4 (7.9% responses): Person 1 gets Good A, Person 2 gets Good B, and Person 3 gets Goods C and D.\n\n"

        elif self.question == 5:
            text += "\n\
Allocation-1 (50% responses): Person 1 gets Goods B and C, Person 2 gets Goods A and F, and Person 3 gets Goods D and E.\n\
Allocation-2 (9.3% responses): Person 1 gets Goods B and E, Person 2 gets Goods C and F, and Person 3 gets Goods A and D.\n\
Allocation-3 (8.3% responses): Person 1 gets Goods C and F, Person 2 gets Goods A and D, and Person 3 gets Goods B and E.\n\
Allocation-4 (6.9% responses): Person 1 gets Good C, Person 2 gets Good A, and Person 3 gets Good E.\n\n"
            
        text += "Please indicate the allocation you think is fairest and explain the reasons behind your choice."
        return text
    
    def generate_prompt1(self):
        indivisible = ' Note that no good can be allocated to more than one person.' if self.llm_type == 'llama' else ''
        if self.instruction == 'EQ':
            instruction_text = 'is to determine the allocation in which all individuals have exactly the same value for their respective bundles. In other words, all individuals should value their bundles equally.'
        elif self.instruction == 'EF':
            instruction_text = 'is to determine the allocation where each individual prefers their own bundle the most. In other words, there should be no envy between any pair of individuals.'
        elif self.instruction == 'USW':
            instruction_text = 'is to determine the allocation which maximizes the sum of values derived by all individuals from their respective bundles.'
        elif self.instruction == 'RMM':
            instruction_text = 'is to determine the allocation where the value derived by the worst-off individual is the most across all possible allocations. In other words, find the allocation that satisfies the \"Max-Min\" criterion.'
        elif self.instruction == 'EQp':
            instruction_text = 'is to provide an allocation of the given goods, keeping in mind that you only care about minimizing the inequality between the individuals involved.'
        elif self.instruction == 'EFp':
            instruction_text = 'is to provide an allocation of the given goods, keeping in mind that you only care about minimizing the envy between the individuals involved.'
        elif self.instruction == 'USWp':
            instruction_text = 'is to provide an allocation of the given goods, keeping in mind that you only care about maximizing the sum of payoffs received by the individuals involved.'
        elif self.instruction == 'RMMp':
            instruction_text = 'is to provide an allocation of the given goods, keeping in mind that you only care about maximixing the payoff received by the worst-off individual.'
        elif self.instruction == 'options':
            instruction_text = self.get_options()
        elif self.instruction == 'top5':
            instruction_text = self.get_top5()
        elif self.instruction == 'bad5':
            instruction_text = self.get_bad5()
        elif self.instruction == 'humans':
            instruction_text = self.get_human_choices()
        elif self.instruction == 'code':
            instruction_text = 'is to provide the allocation of the given goods that you think is fairest, and to provide Python code for the procedure you used to compute the allocation. No additional explanation is required.'
        else:
            instruction_text = "is to determine the allocation that you consider to be the fairest - there are no 'correct' or 'incorrect' answers."

        if -1 not in self.valuations['valuation']:
            self.prompt1 = f"Consider a problem where goods need to be allocated among different individuals. Imagine that the individuals involved, i.e. {self.person_list}, approach you and ask you to determine a fair allocation of {self.goods_list}. The goods to be allocated are indivisible, \
that is, you have to give the good as a whole to one person or you can decide to not alocate it at all, i.e., you throw it away.{indivisible}\n{self.valuation_text}Your task {instruction_text}\n{self.restriction_text}{self.template_text}{self.justification_text}"
        else:
            self.prompt1 = f"Consider a problem where goods need to be allocated among different individuals. Your task is to allocate {self.goods_list}, among the individuals involved, i.e. {self.person_list}. Pick an allocation you consider to be fair and that you think is acceptable to the \
other participants (assume that your proposal can only be realized if all participants agree). The goods to be allocated are indivisible, that is, you have to give the good as a whole to one person or you can decide to not alocate it at all, i.e., you throw it away.{indivisible}\n{self.valuation_text}\
Your task is to determine the allocation {instruction_text}\n{self.restriction_text}{self.template_text}{self.justification_text}"
        
        if self.instruction == 'desirable':
            self.prompt1 = f"Consider a problem where goods need to be allocated among different individuals. Imagine that the individuals involved, i.e. {self.person_list}, approach you and ask you to allocate among them {self.goods_list}. The goods to be allocated are indivisible, \
that is, you have to give the good as a whole to one person or you can decide to not alocate it at all, i.e., you throw it away.{indivisible}\n{self.valuation_text}Your task is to determine the allocation you think is most desirable.\n{self.restriction_text}{self.template_text}{self.justification_text}"
        elif self.instruction == 'acceptable':
            self.prompt1 = f"Consider a problem where goods need to be allocated among different individuals. Imagine that the individuals involved, i.e. {self.person_list}, approach you and ask you to allocate among them {self.goods_list}. The goods to be allocated are indivisible, \
that is, you have to give the good as a whole to one person or you can decide to not alocate it at all, i.e., you throw it away.{indivisible}\n{self.valuation_text}Your task is to determine the allocation you think is acceptable by all individuals.\n{self.restriction_text}{self.template_text}{self.justification_text}"
        elif self.instruction == 'none':
            self.prompt1 = f"Consider a problem where goods need to be allocated among different individuals. Imagine that the individuals involved, i.e. {self.person_list}, approach you and ask you to allocate among them {self.goods_list}. The goods to be allocated are indivisible, \
that is, you have to give the good as a whole to one person or you can decide to not alocate it at all, i.e., you throw it away.{indivisible}\n{self.valuation_text}Your task is to determine an allocation of the given goods among the individuals involved.\n{self.restriction_text}{self.template_text}{self.justification_text}"

        if self.instruction == 'cot':
            if 'money' in self.valuations:
                self.prompt1 = f"Consider the following problem where goods need to be allocated among different individuals:\nImagine that the individuals involved, i.e. Person 1 and Person2 approach you and ask you to determine a fair allocation of 3 goods, namely Good A, Good B, and Good C.\
The goods to be allocated are indivisible, that is, you have to give the good as a whole to one person or you can decide to not alocate it at all, i.e., you throw it away.\n\
Person 1's value for Good A is 45, for Good B is 20, and for Good C is 35.\n\
Person 2's value for Good A is 40, for Good B is 35, and for Good C is 25.\n\
A total of 5 units of money is also available for allocation.\n\n\
If your task is to determine the allocation you think is fairest, the following allocations are important:\n\n\
Allocation-1: Person 1 gets Good A, and Person 2 gets Good B and 5 units of money. Person 1 values their bundle at 45 and Person 2's bundle at 25, while Person 2 values their own bundle at 40 and Person 1's bundle at 40 as well. Since each agent values their own bundle at least as much as they value the other agent's bundle, this allocation is envy-free. \
However, this allocation does not maximize the overall utility (since all goods are not allocated to the agents who respectively value them the most), is not equitable (since the payoffs received by different agents are not identical), and does not satisfy the maximin rule (since there exists an allocation where the worst-off agent has a higher payoff - Allocation 3).\n\n\
Allocation-2: Person 1 gets Good C and 5 units of money, and Person 2 gets Good A. Both Person 1 and Person 2 value their respective bundles at 40. Since both individuals receive identical payoffs, this allocation is equitable. \
However, this allocation does not maximize the overall utility (since all goods are not allocated to the agents who respectively value them the most), is not envy-free (since Person 1 values Person 2's bundle more than their own), and does not satisfy the maximin rule (since there exists an allocation where the worst-off agent has a higher payoff - Allocation 3).\n\n\
Allocation-3: Person 1 gets Good A and 5 units of money, and Person 2 gets Goods B and C. Person 1 values their bundle at 50 and Person 2 values their bundle at 60. Since their is no other allocation where the payoff of the worst-off agent (in this case Person 1) is greater than 50, this allocation satisfies the maximin rule. \
However, this allocation does not maximize the overall utility (since all goods are not allocated to the agents who respectively value them the most), is not envy-free (since Person 1 values Person 2's bundle more than their own), and is not equitable (since the payoffs received by different agents are not identical).\n\n\
Allocation-4: Person 1 gets Goods A and C, and Person 2 gets Good B and 5 units of money. Person 1 values their bundle at 80 and Person 2 values their bundle at 40. Since each good is allocated to the individual who values it the most, this allocation maximizes the overall utility. \
However, this allocation is not envy-free (since Person 2 values Person 1's bundle more than their own), is not equitable (since the payoffs received by different agents are not identical), and does not satsify the maximin rule (since there exists an allocation where the worst-off agent has a higher payoff - Allocation 3).\n\n\
The allocation you choose shall depend on the criteria, among the above, that you think is fairest.\n\n\
Now, consider another problem where goods need to be allocated among different individuals. Imagine that the individuals involved, i.e. {self.person_list}, approach you and ask you to determine a fair allocation of {self.goods_list}. The goods to be allocated are indivisible, \
that is, you have to give the good as a whole to one person or you can decide to not alocate it at all, i.e., you throw it away.{indivisible}\n{self.valuation_text}Your task is to determine the allocation that you think is fairest."
            else:
                self.prompt1 = f"Consider the following problem where goods need to be allocated among different individuals:\nImagine that the individuals involved, i.e. Person 1 and Person2 approach you and ask you to determine a fair allocation of 3 goods, namely Good A, Good B, and Good C.\
The goods to be allocated are indivisible, that is, you have to give the good as a whole to one person or you can decide to not alocate it at all, i.e., you throw it away.\n\
Person 1's value for Good A is 45, for Good B is 20, and for Good C is 35.\n\
Person 2's value for Good A is 35, for Good B is 40, and for Good C is 25.\n\n\
If your task is to determine the allocation you think is fairest, the following allocations are important:\n\n\
Allocation-1: Person 1 gets Good A, and Person 2 gets Good B. Person 1 values their bundle at 45 and Person 2's bundle at 20, while Person 2 values their own bundle at 40 and Person 1's bundle at 35. Since each agent values their own bundle more than they value the other agent's bundle, this allocation is envy-free. \
However, this allocation does not maximize the overall utility (since all goods are not allocated to the agents who respectively value them the most), is not equitable (since the payoffs received by different agents are not identical), and does not satisfy the maximin rule (since there exists an allocation where the worst-off agent has a higher payoff - Allocation 3).\n\n\
Allocation-2: Person 1 gets Good C, and Person 2 gets Good A. Both Person 1 and Person 2 value their respective bundles at 35. Since both individuals receive identical payoffs, this allocation is equitable. \
However, this allocation does not maximize the overall utility (since all goods are not allocated to the agents who respectively value them the most), is not envy-free (since Person 1 values Person 2's bundle more than their own), and does not satisfy the maximin rule (since there exists an allocation where the worst-off agent has a higher payoff - Allocation 3).\n\n\
Allocation-3: Person 1 gets Good A, and Person 2 gets Goods B and C. Person 1 values their bundle at 45 and Person 2 values their bundle at 65. Since their is no other allocation where the payoff of the worst-off agent (in this case Person 1) is greater than 45, this allocation satisfies the maximin rule. \
However, this allocation does not maximize the overall utility (since all goods are not allocated to the agents who respectively value them the most), is not envy-free (since Person 1 values Person 2's bundle more than their own), and is not equitable (since the payoffs received by different agents are not identical).\n\n\
Allocation-4: Person 1 gets Goods A and C, and Person 2 gets Good B. Person 1 values their bundle at 80 and Person 2 values their bundle at 40. Since each good is allocated to the individual who values it the most, this allocation maximizes the overall utility. \
However, this allocation is not envy-free (since Person 2 values Person 1's bundle more than their own), is not equitable (since the payoffs received by different agents are not identical), and does not satsify the maximin rule (since there exists an allocation where the worst-off agent has a higher payoff - Allocation 3).\n\n\
The allocation you choose shall depend on the criteria, among the above, that you think is fairest.\n\n\
Now, consider another problem where goods need to be allocated among different individuals. Imagine that the individuals involved, i.e. {self.person_list}, approach you and ask you to determine a fair allocation of {self.goods_list}. The goods to be allocated are indivisible, \
that is, you have to give the good as a whole to one person or you can decide to not alocate it at all, i.e., you throw it away.{indivisible}\n{self.valuation_text}Your task is to determine the allocation that you think is fairest."
        return self.prompt1
    
    def generate_prompt2(self, prompt1_response):
        plurality = 'each' if self.restriction == 'rank' else 'the'
        self.prompt2 = f"Please present {plurality} allocation you have selected in in the following JSON format:\n"
        self.prompt2 += "{\n"
        for j in range(self.num_goods):
            self.prompt2 += f'\"{self.good_names[j]}\": \"<person to whom {self.good_names[j]} is allocated, \"None\" if {self.good_names[j]} is discarded>\",\n'
        if 'money' in self.valuations:
            for i in list(self.valuations['valuation'].keys()):
                self.prompt2 += f'\"{self.person_names[i]} money\": \"<money allocated to {self.person_names[i]}, 0 if no money was allocated to {self.person_names[i]}>\",\n'
        self.prompt2 += "}\n"

        last_comma = self.prompt2.rfind(',')
        self.prompt2 = self.prompt2[:last_comma] + self.prompt2[last_comma+1:]
            
        if self.llm_type != "gemini":
            self.prompt2 = f"Previously, I asked you the following quesion:\n\"{self.prompt1}\"\nAnd this was your response\n\"{prompt1_response}\"\n" + self.prompt2

        return self.prompt2
    

class OutputAnalyzer():
    def __init__(self, model_paths = ['gemini/gemini-1.5-pro-latest'], index = ['G-1.5-L']) -> None:
        self.model_paths = model_paths
        self.index = ["Humans"] + index
        self.metrics = {}
        self.valuation_comp = {}
        self.variation_dict = {
            'valuation': ['word', 'table', 'pipes'],
            'restriction': ['', 'best', 'rank'],
            'template': ['', 'person_centric', 'goods_centric'],
            'example': ['', True],
            'justification': ['', 'optional', 'mandatory']
        }
        self.strategy_names = {
            'word': 'WP',
            'word_best': 'WP+SB',
            'word_rank': 'WP+RA',
            'word_best_goods_centric': 'WP+SB+GC',
            'word_best_person_centric': 'WP+SB+PC',
            'word_goods_centric': 'WP+GC',
            'word_person_centric': 'WP+PC',
            'word_rank': 'WP+RA',
            'table': 'CS',
            'pipes': 'PS',

        }

    def get_data(self, model_path = "gemini/gemini-1.5-pro-latest", question = 1, variation = "word"):
        filename = f'results/{model_path}/question_{question}/{variation}'

        if '.csv' not in filename: filename += '.csv'
        # print(filename)
        data = pd.read_csv(filename, index_col = 0)
        data = data.astype(str)
        columns = list(data.columns[1:])
        for col in columns:
            data[col].mask(data[col] == 'nan', '0', inplace=True)
            data[col].mask(data[col] == ' ', '0', inplace=True)
            # data[col].mask(data[col] == 'Both', '0', inplace=True)
            data[col] = data[col].apply(lambda x: x[:-2] if x[-2:] == '.0' else x)
        return data

    def print_grouped_results(self, question = 1, variation = "word", length = 5):
        data = self.get_data(question, variation)
        columns = list(data.columns[1:])
        grouped = data.groupby(columns).size().sort_values(ascending=False)[:length]
        print(grouped)

    def initialize_metrics_dict(self, humans=True):
        if humans:
            self.metrics = {
                'q1': {
                    'order': {
                        "1st": [70.4],
                        "2nd": [23.2],
                        "3rd": [0],
                        "4th": [0],
                        "5th": [0],
                        "Other": [6.4]
                    },
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [72.4],
                        'EF': [70.4],
                        'USW': [25.2],
                        'PO': [25.2],
                        'MAX': [0],
                        'RMM': [23.2]
                    },
                    'allocations': {
                        'Type-0':[],
                        'Type-1': [],
                        'Type-2': [],
                        'Type-3': [],
                        'Others': [],
                    },
                },
                'q2': {
                    'order': {
                        "1st": [26.2],
                        "2nd": [26.2],
                        "3rd": [12.7],
                        "4th": [9],
                        "5th": [7.9],
                        "Other": [18]
                    },
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [26.2],
                        'EF': [26.2],
                        'USW': [9],
                        'PO': [47.9],
                        'MAX': [9],
                        'RMM': [38.9]
                    },
                    'allocations': {
                        '45,45,45': [],
                        '47,48,43': [],
                        '47,45,52': [],
                        '47,93,20': [],
                        '47,45,32': [],
                        '52,48,20': [],
                    }
                },
                'q3': {
                    'order': {
                        '1st': [27.8],
                        '2nd': [12.5],
                        '3rd': [9.7],
                        '4th': [7.9],
                        '5th': [6],
                        'Other': [36.1]
                    },
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [0],
                        'EF': [27.8],
                        'USW': [7],
                        'PO': [47.3],
                        'MAX': [0],
                        'RMM': [40.3]
                    },
                    'allocations': {
                        '40,52,52 (EF)': [],
                        '40,52,52': [],
                        '40,34,38': [],
                        '40,26,26': [],
                        '70,52,26': [],
                        '70,26,52': []
                    }
                },
                'q31': {
                    'allocations': {
                        '52,52,52 (EF)': [],
                        '52,52,52': [],
                        '52,34,38': [],
                        '52,26,26': [],
                        '70,64,26': [],
                        '70,52,38': [],
                        '70,26,64': [],
                        '70,38,52': [],
                        '82,52,26': [],
                        '82,26,52': []
                    }
                },
                'q4': {
                    'order': {
                        '1st': [64.4],
                        '2nd': [15.5],
                        "3rd": [4.49],
                        "4th": [3.37],
                        "5th": [2.25],
                        'Other': [9.99]
                    },
                    'relevant': ["1st","2nd","USW","4th","5th","Other"],
                    'notions': {
                        'IA': [16.5],
                        'EF': [64.4],
                        'USW': [10],
                        'PO': [10],
                        'MAX': [0],
                        'RMM': [64.4]
                    },
                    'allocations': {
                        '32,33,32': [],
                        '31,31,31': [],
                        '39,33,32': [],
                        '32,40,32': [],
                        '32,33,39': [],
                        '31,33,30': []
                    }
                },
                'q41': {
                    'allocations': {
                        '65,55,45': [],
                        '40,40,40': [],
                        '75,55,45': [],
                        '65,65,45': [],
                        '65,55,45': [],
                        '55,40,40': []

                    }
                },
                'q42': {
                    'allocations': {
                        '60,50,40': [],
                        '40,40,40': [],
                        '70,50,40': [],
                        '60,60,40': [],
                        '60,50,40': [],
                        '50,40,40': [],
                    }
                },
                'q5': {
                    'order': {
                        '1st': [50],
                        '2nd': [12.5],
                        '3rd': [9.3],
                        '4th': [6.9],
                        '5th': [4.17],
                        'Other': [17.13]
                    },
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [9.3],
                        'EF': [65.2],
                        'USW': [50],
                        'PO': [50],
                        'MAX': [50],
                        'RMM': [50]
                    },
                    'allocations': {
                        '52,48,51': [],
                        '45,45,45': [],
                        '47,46,47': [],
                        '32,26,30': [],
                    }
                },
                'q51': {
                    'allocations': {
                        '52,48,51': [],
                        '45,45,45': [],
                        '47,46,47': [],
                        '32,26,30': [],
                    }
                },
                'q52': {
                    'allocations': {
                        '57,49,54': [],
                        '47,46,47': [],
                        '32,26,30': [],
                    }
                },
                'q6': {
                    'order': {
                        '1st': [32.6],
                        '2nd': [28.1],
                        '3rd': [18.4],
                        '4th': [7.9],
                        '5th': [2.62],
                        'Other': [10.38]
                    },
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [32.6],
                        'EF': [46.5],
                        'USW': [7.9],
                        'PO': [36],
                        'MAX': [7.9],
                        'RMM': [28.1]
                    }, 
                    'allocations': {
                        '45,45,45': [],
                        '48,60,52': [],
                        '48,40,52': [],
                        '48,20,97': [],
                        '52,40,52': []
                    }
                },
                'q61': {
                    'allocations': {
                        '45,45,45': [],
                        '48,60,52': [],
                        '48,40,52': [],
                        '48,20,97': [],
                        '52,40,52': []
                    }
                },
                'q62': {
                    'allocations': {
                        '45,45,45': [],
                        '60,48,52': [],
                        '40,48,52': [],
                        '20,48,97': [],
                        '40,52,52': []
                    }
                },
                'q7': {
                    'order': {
                        '1st': [55],
                        '2nd': [12.7],
                        '3rd': [16.8],
                        '4th': [15],
                        "5th": [0],
                        "Other": [0.5]
                    },
                    'relevant': ["1st","2nd","3rd","4th","Other"],
                    'notions': {
                        'IA': [55],
                        'EF': [12.7],
                        'USW': [0],
                        'PO': [12.7],
                        'MAX': [0],
                        'RMM': [55]
                    }
                },
                'q8': {
                    'order': {
                        "1st": [32.2],
                        "2nd": [22.5],
                        "3rd": [16.9],
                        "4th": [9.4],
                        "5th": [7.5],
                        "Other": [11.5]
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [22.5],
                        'EF': [32.2],
                        'USW': [9.4],
                        'PO': [41.6],
                        'MAX': [0],
                        'RMM': [32.2]
                    }
                },
                'q9': {
                    'order': {
                        "1st": [34.1],
                        "2nd": [30],
                        "3rd": [17.6],
                        "4th": [5],
                        "5th": [2.25],
                        "Other": [11.05],
                    },
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [30],
                        'EF': [51.7],
                        'USW': [0],
                        'PO': [34.1],
                        'MAX': [0],
                        'RMM': [34.1]
                    }, 
                    'allocations': {
                        '60,54,49': [],
                        '43,43,43': [],
                        '40,54,49': [],
                        '20,97,49': [],
                        '37,43,49': [],
                        '20,43,49': [],
                        '43,20,92': [],
                        '40,1,92': [],
                        '40,55,49': [],
                        '23,97,4': [],
                        '40,1,49': [],
                        '57,2,4': [],
                    }
                },
                'q10': {
                    'order': {
                        'Best+EQ': [45.7],
                        'Best+EF': [8.6],
                        'Best+Other': [19.9],
                        '2nd': [11.6],
                        'Other': [14.2]
                    },
                    'relevant': ["Best+EQ","Best+EF","Best+Other","2nd","Other"],
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    }
                },
            }

        else:
            self.metrics = {
                'q1': {
                    'order': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        "Other": []
                    },
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    },
                    'allocations': {
                        # 'Type-0':[],
                        'Type-1': [],
                        'Type-2': [],
                        'Type-3': [],
                        'Others': [],
                    },
                    'relevant': ["1st","2nd","Other"]
                },
                'q11': {
                    'allocations': {
                        # 'Type-0':[],
                        'Type-1': [],
                        'Type-2': [],
                        'Type-3': [],
                        'Others': [],
                    }
                },
                'q12': {
                    'allocations': {
                        # 'Type-0':[],
                        'Type-1': [],
                        'Type-2': [],
                        'Type-3': [],
                        'Others': [],
                    }
                },
                'q13': {
                    'allocations': {
                        # 'Type-0':[],
                        'Type-1': [],
                        'Type-2': [],
                        'Type-3': [],
                        'Others': [],
                    }
                },
                'q18': {
                    'allocations': {
                        'A|B': [],
                        'A|B,C': [],
                        'A,C|B': [],
                        'B|A': [],
                        'A,B|C': [],
                    }
                },
                'q14': {
                    'allocations': {
                        '(0,50)': [],
                        '(5-,45+)': [],
                        '(10-,40+)': [],
                        '(15-,35+)': [],
                        '(20-,30+)': [],
                        '(24-,26+)': [],
                        '(25,25)': [],
                        '(26+,24-)': [],
                        '(30+,20-)': [],
                        '(35+,15-)': [],
                        '(40+,10-)': [],
                        '(45+,5-)': [],
                        '(50,0)': [],
                    },
                },
                'q15': {
                    'allocations': {
                        '(0,50)': [],
                        '(5-,45+)': [],
                        '(10-,40+)': [],
                        '(15-,35+)': [],
                        '(20-,30+)': [],
                        '(24-,26+)': [],
                        '(25,25)': [],
                        '(26+,24-)': [],
                        '(30+,20-)': [],
                        '(35+,15-)': [],
                        '(40+,10-)': [],
                        '(45+,5-)': [],
                        '(50,0)': [],
                    },
                },
                'q16': {
                    'allocations': {
                        '(0,50)': [],
                        '(5-,45+)': [],
                        '(10-,40+)': [],
                        '(15-,35+)': [],
                        '(20-,30+)': [],
                        '(24-,26+)': [],
                        '(25,25)': [],
                        '(26+,24-)': [],
                        '(30+,20-)': [],
                        '(35+,15-)': [],
                        '(40+,10-)': [],
                        '(45+,5-)': [],
                        '(50,0)': [],
                    },
                },
                'q17': {
                    'allocations': {
                        '(0,50)': [],
                        '(5-,45+)': [],
                        '(10-,40+)': [],
                        '(15-,35+)': [],
                        '(20-,30+)': [],
                        '(24-,26+)': [],
                        '(25,25)': [],
                        '(26+,24-)': [],
                        '(30+,20-)': [],
                        '(35+,15-)': [],
                        '(40+,10-)': [],
                        '(45+,5-)': [],
                        '(50,0)': [],
                    },
                },
                'q19': {
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                },
                'q2': {
                    'order': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        "Other": []
                    },
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'bads': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    },
                    'allocations': {
                        '45,45,45': [],
                        '47,48,43': [],
                        '47,45,52': [],
                        '47,45,32': [], 
                        '47,48,23': [], 
                        '47,48,20': [],
                        '50,45,32': [],
                        '50,48,23': [],
                        '45,45,25': [],
                        '47,93,20': [],
                    }
                },
                'q21': {
                    'allocations': {
                        '45,45,45': [],
                        '47,48,43': [],
                        '47,93,20': [],
                        '47,45,32': [], 
                        '47,48,23': [], 
                        '47,48,20': [],
                        '92,45,20': [],
                        '92,47,0': [],
                        '52,48,20': [],
                        '45,45,20': [],
                        # '5,48,25': [],
                        # '5,53,20': [],
                        # '8,5,32': [],
                    }
                },
                'q22': {
                    'allocations': {
                        '50,50,50': [],
                        '60,75,35': [],
                        '60,50,65': [],
                        '60,125,20': [],
                        '60,50,45': [],
                        '70,75,20': [],
                        '60,75,20': [],
                        '50,50,30': [],
                    }
                },
                'q23': {
                    'allocations': {
                        '45,45,45': [],
                        '48,47,43': [],
                        '45,47,52': [],
                        '45,47,32': [], 
                        '48,47,23': [], 
                        '48,47,20': [],
                        '45,50,32': [],
                        '48,50,23': [],
                        '45,45,25': [],
                        '93,47,20': [],
                    }
                },
                'q24': {
                    'allocations': {
                        '45,45,45': [],
                        '43,47,48': [],
                        '52,47,45': [],
                        '32,47,45': [], 
                        '23,47,48': [], 
                        '20,47,48': [],
                        '32,50,45': [],
                        '23,50,48': [],
                        '25,45,45': [],
                        '20,47,93': [],
                    }
                },
                'q3': {
                    'order': {
                        '1st': [],
                        '2nd': [],
                        "3rd": [],
                        "4th": [],
                        'USW': [],
                        'Other': []
                    },
                    'relevant': ["1st","2nd","3rd","4th","USW","Other"],
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    },
                    'allocations': {
                        '40,52,52*': [],
                        '40,52,52': [],
                        '40,26,26': [],
                        '70,52,26': [],
                        '70,26,52': [],
                        '40,34,38': [],
                        '40,78,26': [],
                    }
                },
                'q31': {
                    'allocations': {
                        '52,52,52 (EF)': [],
                        '52,52,52': [],
                        '52,34,38': [],
                        '52,26,26': [],
                        '70,64,26': [],
                        '70,52,38': [],
                        '70,26,64': [],
                        '70,38,52': [],
                        '82,52,26': [],
                        '82,26,52': [],
                        '65,52,38': [],
                        '65,52,26': [],
                        '40,52,52': [],
                    }
                },
                'q32': {
                    'allocations': {
                        '52,40,52*': [],
                        '52,40,52': [],
                        '26,40,26': [],
                        '52,70,26': [],
                        '26,70,52': [],
                        '34,40,38': [],
                        '78,40,26': [],
                    }
                },
                'q4': {
                    'order': {
                        '1st': [],
                        '2nd': [],
                        'USW': [],
                        "4th": [],
                        "5th": [],
                        'Other': []
                    },
                    'relevant': ["1st","2nd","USW","4th","5th","Other"],
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    },
                    'allocations': {
                        '32,33,32': [],
                        '31,31,31': [],
                        '39,33,32': [],
                        '32,40,32': [],
                        '32,33,39': [],
                        '31,33,30': [],
                        '33,38,30': [],
                        '30,31,32': [],
                    }
                },
                'q41': {
                    'allocations': {
                        '65,55,45': [],
                        '40,40,40': [],
                        '75,55,45': [],
                        '65,65,45': [],
                        '65,55,55': [],
                        '40,55,40': []

                    }
                },
                'q42': {
                    'allocations': {
                        '60,50,40': [],
                        '40,40,40': [],
                        '70,50,40': [],
                        '60,60,40': [],
                        '60,50,50': [],
                        '40,50,40': [],
                    }
                },
                'q43': {
                    'allocations': {
                        '32,33,32': [],
                        '31,31,31': [],
                        '39,33,32': [],
                        '32,40,32': [],
                        '32,33,39': [],
                        '31,33,30': [],
                        '33,38,30': [],
                        '30,31,32': [],
                    }
                },
                'q5': {
                    'order': {
                        '1st': [],
                        '2nd': [],
                        '3rd': [],
                        '4th': [],
                        '5th': [],
                        "Other": []
                    },
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        'Other': [],
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    },
                    'allocations': {
                        '52,48,51': [],
                        '47,46,47': [],
                        '32,26,30': [],
                        '57,48,38': [],
                        '57,46,19': [],
                        '67,26,51': [],
                        '77,48,21': [],
                        '32,68,47': [],
                        '32,48,68': [],
                        '32,68,47': [],
                        '35,48,47': [],
                    },
                },
                'q51': {
                    'allocations': {
                        '52,48,51': [],
                        '47,46,47': [],
                        '32,26,30': [],
                        '57,48,38': [],
                        '57,46,19': [],
                        '67,26,51': [],
                        '77,48,21': [],
                        '32,68,47': [],
                        '32,48,68': [],
                        '32,68,47': [],
                        '35,48,47': [],
                    }
                },
                'q52': {
                    'allocations': {
                        '57,49,54': [],
                        '47,46,47': [],
                        '32,26,30': [],
                    }
                },
                'q6': {
                    'order': {
                        '1st': [],
                        '2nd': [],
                        '3rd': [],
                        '4th': [],
                        "5th": [],
                        "Other": []
                    },
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        'Other': [],
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    }, 
                    'allocations': {
                        '45,45,45': [],
                        '48,60,52': [],
                        '48,40,52': [],
                        '40,52,52': [],
                        '48,20,97': [],
                        '93,20,45': [],
                        'Others': []
                    }
                },
                'q61': {
                    'allocations': {
                        '48,60,52': [],
                        '48,40,52': [],
                        '48,20,97': [],
                        '40,52,52': [],
                        '93,20,45': [],
                        '93,40,1': [],
                        '93,60,0': [],
                        '52,15,45': [],
                        '45,60,2': [],
                        '45,20,47': [],
                        '7,25,52': []
                    }
                },
                'q62': {
                    'allocations': {
                        '60,48,52': [],
                        '40,48,52': [],
                        '20,48,97': [],
                        '40,52,52': [],
                        '20,93,45': [],
                        '40,93,1': [],
                        '60,93,0': [],
                        '40,45,2': [],
                        # '15,52,45': [],
                        # '60,45,2': [],
                        # '20,45,47': [],
                        # '25,7,52': []
                    }
                },
                'q7': {
                    'order': {
                        '1st': [],
                        '2nd': [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'bads': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'options': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        'Other': [],
                    },
                    'relevant': ["1st","2nd","3rd","4th","Other"],
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    }
                },
                'q8': {
                    'order': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        "Other": []
                    },
                    'relevant': ["1st","2nd","3rd","4th","5th","Other"],
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    }
                },
                'q9': {
                    'order': {
                        "1st": [],
                        "2nd": [],
                        "3rd": [],
                        "4th": [],
                        "5th": [],
                        "Other": [],
                    },
                    'relevant': ["1st","2nd","3rd","Other"],
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    }, 
                    'allocations': {
                        '43,43,43': [],
                        '60,54,49': [],
                        '40,54,49': [],
                        '40,54,53': [],
                        '20,97,49': [],
                        '43,20,92': [],
                        'Others': []
                    }
                },
                'q91': {
                    'allocations': {
                        '54,60,49': [],
                        '54,40,49': [],
                        '54,40,53': [], 
                        '97,20,49': [],
                        '20,43,92': [],
                    }
                },
                'q10': {
                    'order': {
                        'Best+EQ': [],
                        'Best+EF': [],
                        'Best+Other': [],
                        '2nd': [],
                        'Other': []
                    },
                    'relevant': ["Best+EQ","Best+EF","Best+Other","2nd","Other"],
                    'notions': {
                        'IA': [],
                        'EF': [],
                        'USW': [],
                        'PO': [],
                        'MAX': [],
                        'RMM': []
                    }
                },
            }

    def get_comparison(self, question = 1, variation = "word", model_paths_custom = None, humans=True):
        self.initialize_metrics_dict(humans)    

        pg_temp = PromptGenerator(question) 
        pg_temp.set_valuations()   
        valuation_dict = pg_temp.valuations['valuation']

        print(model_paths_custom)
        if not model_paths_custom: model_paths_custom = self.model_paths
        for model_path in model_paths_custom:
            data = self.get_data(model_path[0], question, model_path[1])
            if question == 1:
                a_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '0')
                ].shape[0]
                a_bc = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2')
                ].shape[0]
                ac_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '1')
                ].shape[0]
                b_a = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '0')
                ].shape[0]
                bc_a = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '1')
                ].shape[0]
                other = 100 - a_b - a_bc 
                self.metrics[f'q{question}']['order']['1st'].append(a_b) 
                self.metrics[f'q{question}']['order']['2nd'].append(a_bc)
                self.metrics[f'q{question}']['order']['3rd'].append(0)
                self.metrics[f'q{question}']['order']['4th'].append(0)
                self.metrics[f'q{question}']['order']['5th'].append(0)
                self.metrics[f'q{question}']['order']['Other'].append(other)
                others = 100 - a_b - a_bc - ac_b 
                # self.metrics[f'q{question}']['allocations']['Type-0'].append(b_a) 
                self.metrics[f'q{question}']['allocations']['Type-1'].append(a_b) 
                self.metrics[f'q{question}']['allocations']['Type-2'].append(ac_b) 
                self.metrics[f'q{question}']['allocations']['Type-3'].append(a_bc) 
                self.metrics[f'q{question}']['allocations']['Others'].append(others) 
                # self.metrics[f'q{question}']['allocations']['A,B|C'].append(ab_c) 
                self.metrics[f'q{question}']['alloc_labels'] = ['EF\nIA', 'USW', 'USW\nRMM', 'Others'] 
                self.metrics[f'q{question}']['labels'] = ['EF\nIA', 'USW\nRMM', '', '', '', '']
                self.metrics[f'q{question}']['payoffs']= {
                    'Type-0': '(46,47)',
                    'Type-1': '(49,48)',
                    'Type-2': '(54,48)',
                    'Type-3': '(49,53)',
                    'Others': ''
                }
                self.metrics[f'q{question}']['notions']['IA'].append(data[
                        ((data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '0') |
                        (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '0'))
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['EF'].append(data[
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '0') 
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['USW'].append(data[
                        ((data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '1') |
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2'))
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['PO'].append(data[
                        ((data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '1') |
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2'))
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['MAX'].append(data[
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '1')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['RMM'].append(data[
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2')
                    ].shape[0])
                
            elif question == 11:
                a_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '0')
                ].shape[0]
                a_bc = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2')
                ].shape[0]
                ac_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '1')
                ].shape[0]
                b_a = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '0')
                ].shape[0]
                ab_c = data[
                    (data['Good A'] == '1') & (data['Good B'] == '1') & (data['Good C'] == '2')
                ].shape[0]
                others = 100 - a_b - a_bc - ac_b
                # self.metrics[f'q{question}']['allocations']['Type-0'].append(b_a) 
                self.metrics[f'q{question}']['allocations']['Type-1'].append(a_b) 
                self.metrics[f'q{question}']['allocations']['Type-2'].append(ac_b) 
                self.metrics[f'q{question}']['allocations']['Type-3'].append(a_bc) 
                self.metrics[f'q{question}']['allocations']['Others'].append(others) 
                self.metrics[f'q{question}']['payoffs']= {
                    'Type-0': '(0,20)',
                    'Type-1': '(80,60)',
                    'Type-2': '(100,60)',
                    'Type-3': '(80,80)',
                    'Others': ''
                }
                # self.metrics[f'q{question}']['allocations']['A,B|C'].append(ab_c) 
                self.metrics[f'q{question}']['alloc_labels'] = ['EF', 'EF\nUSW', 'ALL', 'Others'] 

            elif question == 12:
                a_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '0')
                ].shape[0]
                a_bc = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2')
                ].shape[0]
                ac_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '1')
                ].shape[0]
                b_a = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '0')
                ].shape[0]
                ab_c = data[
                    (data['Good A'] == '1') & (data['Good B'] == '1') & (data['Good C'] == '2')
                ].shape[0]
                others = 100 - a_b - a_bc - ac_b 
                # self.metrics[f'q{question}']['allocations']['Type-0'].append(b_a) 
                self.metrics[f'q{question}']['allocations']['Type-1'].append(a_b) 
                self.metrics[f'q{question}']['allocations']['Type-2'].append(ac_b) 
                self.metrics[f'q{question}']['allocations']['Type-3'].append(a_bc) 
                self.metrics[f'q{question}']['allocations']['Others'].append(others) 
                # self.metrics[f'q{question}']['allocations']['A,B|C'].append(ab_c)  
                self.metrics[f'q{question}']['alloc_labels'] = ['EF', 'EF', 'ALL', 'Others'] 
                self.metrics[f'q{question}']['payoffs']= {
                    'Type-0': '(5,20)',
                    'Type-1': '(80,60)',
                    'Type-2': '(95,60)',
                    'Type-3': '(80,80)',
                    'Others': ''
                }
            
            elif question == 13:
                a_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '0')
                ].shape[0]
                a_bc = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2')
                ].shape[0]
                ac_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '1')
                ].shape[0]
                b_a = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '0')
                ].shape[0]
                ab_c = data[
                    (data['Good A'] == '1') & (data['Good B'] == '1') & (data['Good C'] == '2')
                ].shape[0]
                others = 100 - a_b - a_bc - ac_b 
                # self.metrics[f'q{question}']['allocations']['Type-0'].append(b_a) 
                self.metrics[f'q{question}']['allocations']['Type-1'].append(a_b) 
                self.metrics[f'q{question}']['allocations']['Type-2'].append(ac_b) 
                self.metrics[f'q{question}']['allocations']['Type-3'].append(a_bc) 
                self.metrics[f'q{question}']['allocations']['Others'].append(others) 
                # self.metrics[f'q{question}']['allocations']['A,B|C'].append(ab_c)  
                self.metrics[f'q{question}']['alloc_labels'] = ['EF', 'ALL', 'EF', 'Others'] 
                self.metrics[f'q{question}']['payoffs']= {
                    # 'Type-0': '(20,5)',
                    'Type-1': '(60,80)',
                    'Type-2': '(80,80)',
                    'Type-3': '(60,95)',
                    'Others': ''
                }

            elif question == 18:
                a_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '0')
                ].shape[0]
                a_bc = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '1')
                ].shape[0]
                ac_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2')
                ].shape[0]
                b_a = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '0')
                ].shape[0]
                ab_c = data[
                    (data['Good A'] == '1') & (data['Good B'] == '1') & (data['Good C'] == '2')
                ].shape[0]
                self.metrics[f'q{question}']['allocations']['A|B'].append(a_b) 
                self.metrics[f'q{question}']['allocations']['A|B,C'].append(a_bc) 
                self.metrics[f'q{question}']['allocations']['A,C|B'].append(ac_b) 
                self.metrics[f'q{question}']['allocations']['B|A'].append(b_a) 
                self.metrics[f'q{question}']['allocations']['A,B|C'].append(ab_c)  
                self.metrics[f'q{question}']['alloc_labels'] = ['EF', 'IA+EF\nUSW\n+RMM', 'EF', '-', '-'] 

            elif question >= 14 and question <= 17:
                data_alloc = data[(data['Good A'] == '1') & (data['Good B'] == '2')]
                data_alloc['Person 1 money'] = pd.to_numeric(data_alloc['Person 1 money'], errors='coerce')
                data_alloc['Person 2 money'] = pd.to_numeric(data_alloc['Person 2 money'], errors='coerce')
                data_alloc = data_alloc.dropna()
                # data_alloc = data_alloc[['Person 1 money', 'Person 2 money']].astype('float')
                # data_alloc['Person 1 money'] = round(data_alloc['Person 1 money'], 2)
                # data_alloc['Person 2 money'] = round(data_alloc['Person 2 money'], 2)
                data_alloc = data_alloc[(data_alloc['Person 1 money'] + data_alloc['Person 2 money'] <= 50)]
                total = data_alloc.shape[0]
                running = 0
                print( question, total, model_path)
                zero_fifty = data_alloc[(data_alloc['Person 2 money'] == 50) & (data_alloc['Person 1 money'] == 0)].shape[0]
                running += zero_fifty
                five_fortyfive = data_alloc[(data_alloc['Person 2 money'] >= 45) & (data_alloc['Person 1 money'] <= 5)].shape[0] - running
                running += five_fortyfive
                ten_forty = data_alloc[(data_alloc['Person 2 money'] >= 40) & (data_alloc['Person 1 money'] <= 10)].shape[0] - running
                running += ten_forty
                fifteen_thirtyfive = data_alloc[(data_alloc['Person 2 money'] >= 35) & (data_alloc['Person 1 money'] <= 15)].shape[0] - running
                running += fifteen_thirtyfive
                twenty_thirty = data_alloc[(data_alloc['Person 2 money'] >= 30) & (data_alloc['Person 1 money'] <= 20)].shape[0] - running
                running += twenty_thirty
                twentyfive_mp = data_alloc[(data_alloc['Person 2 money'] > 25) & (data_alloc['Person 1 money'] < 25)].shape[0] - running
                equal = data_alloc[(data_alloc['Person 2 money'] == 25) & (data_alloc['Person 1 money'] == 25)].shape[0] 
                running = 0
                fifty_zero = data_alloc[(data_alloc['Person 2 money'] == 0) & (data_alloc['Person 1 money'] == 50)].shape[0]
                running += fifty_zero
                fortyfive_five = data_alloc[(data_alloc['Person 2 money'] <= 5) & (data_alloc['Person 1 money'] >= 45)].shape[0] - running
                running += fortyfive_five
                forty_ten = data_alloc[(data_alloc['Person 2 money'] <= 10) & (data_alloc['Person 1 money'] >= 40)].shape[0] - running
                running += forty_ten
                thirtyfive_fifteen = data_alloc[(data_alloc['Person 2 money'] <= 15) & (data_alloc['Person 1 money'] >= 35)].shape[0] - running
                running += thirtyfive_fifteen
                thirty_twenty = data_alloc[(data_alloc['Person 2 money'] <= 20) & (data_alloc['Person 1 money'] >= 30)].shape[0] - running
                running += thirty_twenty
                twentyfive_pm = data_alloc[(data_alloc['Person 2 money'] < 25) & (data_alloc['Person 1 money'] > 25)].shape[0] - running
                # print(type(self.metrics[f'q{question}']['allocations']), self.metrics[f'q{question}']['allocations'])
                self.metrics[f'q{question}']['allocations']['(0,50)'].append(zero_fifty*100/total)  
                self.metrics[f'q{question}']['allocations']['(5-,45+)'].append(five_fortyfive*100/total)  
                self.metrics[f'q{question}']['allocations']['(10-,40+)'].append(ten_forty*100/total)  
                self.metrics[f'q{question}']['allocations']['(15-,35+)'].append(fifteen_thirtyfive*100/total)  
                self.metrics[f'q{question}']['allocations']['(20-,30+)'].append(twenty_thirty*100/total)  
                self.metrics[f'q{question}']['allocations']['(24-,26+)'].append(twentyfive_mp*100/total)  
                self.metrics[f'q{question}']['allocations']['(25,25)'].append(equal*100/total)  
                self.metrics[f'q{question}']['allocations']['(26+,24-)'].append(twentyfive_pm*100/total)  
                self.metrics[f'q{question}']['allocations']['(30+,20-)'].append(thirty_twenty*100/total)  
                self.metrics[f'q{question}']['allocations']['(35+,15-)'].append(thirtyfive_fifteen*100/total)  
                self.metrics[f'q{question}']['allocations']['(40+,10-)'].append(forty_ten*100/total)  
                self.metrics[f'q{question}']['allocations']['(45+,5-)'].append(fortyfive_five*100/total)  
                self.metrics[f'q{question}']['allocations']['(50,0)'].append(fifty_zero*100/total)

                # zero_fifty = data_alloc[(data_alloc['Person 2 money'] == '50') & (data_alloc['Person 1 money'] == '0')].shape[0]
                # five_fortyfive = data_alloc[(data_alloc['Person 2 money'] == '45') & (data_alloc['Person 1 money'] == '5')].shape[0]
                # ten_forty = data_alloc[(data_alloc['Person 2 money'] == '40') & (data_alloc['Person 1 money'] == '10')].shape[0]
                # fifteen_thirtyfive = data_alloc[(data_alloc['Person 2 money'] == '35') & (data_alloc['Person 1 money'] == '15')].shape[0]
                # twenty_thirty = data_alloc[(data_alloc['Person 2 money'] == '30') & (data_alloc['Person 1 money'] == '20')].shape[0]
                # equal = data_alloc[(data_alloc['Person 2 money'] == '25') & (data_alloc['Person 1 money'] == '25')].shape[0]
                # thirty_twenty = data_alloc[(data_alloc['Person 2 money'] == '20') & (data_alloc['Person 1 money'] == '30')].shape[0]
                # thirtyfive_fifteen = data_alloc[(data_alloc['Person 2 money'] == '15') & (data_alloc['Person 1 money'] == '35')].shape[0]
                # forty_ten = data_alloc[(data_alloc['Person 2 money'] == '10') & (data_alloc['Person 1 money'] == '40')].shape[0]
                # fortyfive_five = data_alloc[(data_alloc['Person 2 money'] == '5') & (data_alloc['Person 1 money'] == '45')].shape[0]
                # fifty_zero = data_alloc[(data_alloc['Person 2 money'] == '0') & (data_alloc['Person 1 money'] == '50')].shape[0]
                # # print(type(self.metrics[f'q{question}']['allocations']), self.metrics[f'q{question}']['allocations'])
                # self.metrics[f'q{question}']['allocations']['0|50'].append(zero_fifty)  
                # self.metrics[f'q{question}']['allocations']['5|45'].append(five_fortyfive)  
                # self.metrics[f'q{question}']['allocations']['10|40'].append(ten_forty)  
                # self.metrics[f'q{question}']['allocations']['15|35'].append(fifteen_thirtyfive)  
                # self.metrics[f'q{question}']['allocations']['20|30'].append(twenty_thirty)  
                # self.metrics[f'q{question}']['allocations']['25|25'].append(equal)  
                # self.metrics[f'q{question}']['allocations']['30|20'].append(thirty_twenty)  
                # self.metrics[f'q{question}']['allocations']['35|15'].append(thirtyfive_fifteen)  
                # self.metrics[f'q{question}']['allocations']['40|10'].append(forty_ten)  
                # self.metrics[f'q{question}']['allocations']['45|5'].append(fortyfive_five)  
                # self.metrics[f'q{question}']['allocations']['50|0'].append(fifty_zero)
                # other = 100
                # idx = len(self.metrics[f'q{question}']['allocations']['50|0'])-1
                # for key in self.metrics[f'q{question}']['allocations']:
                #     if key == 'Other': continue
                #     other -= self.metrics[f'q{question}']['allocations'][key][idx]
                # self.metrics[f'q{question}']['allocations']['Other'].append(other)


                if question == 14: 
                    self.metrics[f'q{question}']['alloc_labels'] = ['EQ\nEF', 'EF', 'EF', 'EF', u'c\u2081<c\u2082', u'c\u2081<c\u2082', 'c\u2081=c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082'] 
                elif question == 15: 
                    self.metrics[f'q{question}']['alloc_labels'] = ['EF', 'EQ\nEF', 'EF', 'EF', u'c\u2081<c\u2082', u'c\u2081<c\u2082', 'c\u2081=c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082'] 
                elif question == 16: 
                    self.metrics[f'q{question}']['alloc_labels'] = ['EF', 'EF', 'EQ\nEF', 'EF', u'c\u2081<c\u2082', u'c\u2081<c\u2082', 'c\u2081=c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082']  
                elif question == 17: 
                    self.metrics[f'q{question}']['alloc_labels'] = ['EF', 'EF', 'EF', 'EQ\nEF', u'c\u2081<c\u2082', u'c\u2081<c\u2082', 'c\u2081=c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082', u'c\u2081>c\u2082']  
                
            elif question == 19:
                a_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '0')
                ].shape[0]
                a_bc = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2')
                ].shape[0]
                ac_b = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '1')
                ].shape[0]
                ab_c = data[
                    (data['Good A'] == '1') & (data['Good B'] == '1') & (data['Good C'] == '2')
                ].shape[0]
                c_a = data[
                    (data['Good A'] == '2') & (data['Good B'] == '0') & (data['Good C'] == '1')
                ].shape[0]
                oth = 100 - a_b - a_bc - ac_b - ab_c - c_a
            
                self.metrics[f'q{question}']['options']['1st'].append(c_a) 
                self.metrics[f'q{question}']['options']['2nd'].append(a_b)
                self.metrics[f'q{question}']['options']['3rd'].append(a_bc)
                self.metrics[f'q{question}']['options']['4th'].append(ac_b)
                self.metrics[f'q{question}']['options']['5th'].append(ab_c)
                self.metrics[f'q{question}']['options']['Other'].append(oth)

                self.metrics[f'q{question}']['option_labels'] = ['EQ', 'EF', 'RMM', 'USW', 'NA', 'Oth.'] 

            elif question == 2:
                best = data[
                    (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') & (data['Good D'] == '3')
                ].shape[0]
                second = data[
                    (data['Good A'] == '3') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3')
                ].shape[0]
                third = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '3') & (data['Good D'] == '3')
                ].shape[0]
                fourth = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3')
                ].shape[0]
                fifth = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '3') & (data['Good D'] == '0')
                ].shape[0]
                other = 100 - best - second - third - fourth - fifth
                self.metrics[f'q{question}']['order']['1st'].append(best) 
                self.metrics[f'q{question}']['order']['2nd'].append(second)
                self.metrics[f'q{question}']['order']['3rd'].append(third) 
                self.metrics[f'q{question}']['order']['4th'].append(fourth)
                self.metrics[f'q{question}']['order']['5th'].append(fifth)
                self.metrics[f'q{question}']['order']['Other'].append(other)
                # print(self.metrics[f'q{question}'])
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                self.metrics[f'q{question}']['alloc_labels'] = ['EQ,RMM', 'EF,PO', 'RMM,PO', 'SD', 'SD', '', 'PO', 'PO', '', 'USW']
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-4:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
                    # else:
                    #     key = list(self.metrics[f'q{question}']['allocations'].keys())[0]
                    #     pos = len(self.metrics[f'q{question}']['allocations'][key])-1
                    #     self.metrics[f'q{question}']['allocations'][payoff_string] = [0]*pos + [1]
                    #     self.metrics[f'q{question}']['alloc_labels'].append('-')
                self.metrics[f'q{question}']['labels'] = ['IA\nRMM', 'EF\nPO', 'RMM\nPO', 'USW', '', '']
                self.metrics[f'q{question}']['notions']['IA'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') & (data['Good D'] == '3')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['EF'].append(data[
                        (data['Good A'] == '3') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['USW'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['PO'].append(data[
                        ((data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3') |
                        (data['Good A'] == '3') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3') |
                        (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '3') & (data['Good D'] == '3'))
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['MAX'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['RMM'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '3') & (data['Good D'] == '3')
                    ].shape[0])
                
                
                self.metrics[f'q{question}']['options']['1st'].append(best) 
                self.metrics[f'q{question}']['options']['2nd'].append(second)
                self.metrics[f'q{question}']['options']['3rd'].append(third) 
                self.metrics[f'q{question}']['options']['4th'].append(fourth)
                self.metrics[f'q{question}']['options']['5th'].append(fifth)   
                self.metrics[f'q{question}']['options']['Other'].append(other)      

                self.metrics[f'q{question}']['option_labels'] = ['EQ\nRMM', 'EF\nPO', 'RMM\nPO', 'USW', 'NA', 'Oth.']  

                best = data[
                    (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') & (data['Good D'] == '0')
                ].shape[0]
                second = data[
                    (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') & (data['Good D'] == '1')
                ].shape[0]
                third = data[
                    (data['Good A'] == '0') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3')
                ].shape[0]
                fourth = data[
                    (data['Good A'] == '1') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3')
                ].shape[0]
                fifth = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '1') & (data['Good D'] == '3')
                ].shape[0]
                other = 100 - best - second - third - fourth - fifth    

                if 'bads' in self.metrics[f'q{question}']:
                    self.metrics[f'q{question}']['bads']['1st'].append(best) 
                    self.metrics[f'q{question}']['bads']['2nd'].append(second)
                    self.metrics[f'q{question}']['bads']['3rd'].append(third) 
                    self.metrics[f'q{question}']['bads']['4th'].append(fourth)
                    self.metrics[f'q{question}']['bads']['5th'].append(fifth)   
                    self.metrics[f'q{question}']['bads']['Other'].append(other)      

                    self.metrics[f'q{question}']['ranges'] = ['20', '23', '28', '32', '72', 'Oth.']

            elif question in [21,22]:
                if question == 21: 
                    self.metrics[f'q{question}']['alloc_labels'] = ['IA,RMM', 'EF,PO', 'USW', 'SRR', 'SRR', '-', '-', '-', '-', '-', '-', '-']
                elif question == 22:
                    self.metrics[f'q{question}']['alloc_labels'] = ['IA+RMM', 'PO', 'RMM+PO', 'USW', '-', '-', '-', '-']
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-4:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
                    # else:
                    #     key = list(self.metrics[f'q{question}']['allocations'].keys())[0]
                    #     pos = len(self.metrics[f'q{question}']['allocations'][key])-1
                    #     self.metrics[f'q{question}']['allocations'][payoff_string] = [0]*pos + [1]
                    #     self.metrics[f'q{question}']['alloc_labels'].append('-')

            elif question in [23,24]:
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-4:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        if r.isdigit():
                            recipient = int(r)
                        elif r in ['You', 'Me', 'Myself']:
                            recipient = 0
                        if recipient == 1: recipient = 0
                        if recipient: 
                            payoff[recipient-1] += valuation_dict[recipient-1][i]
                        elif not r.isdigit():
                            payoff[recipient] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1


                self.metrics[f'q{question}']['alloc_labels'] = ['EQ,RMM', 'EF,PO', 'RMM,PO', 'SD', 'SD', '', 'PO', 'PO', '', 'USW']

            elif question == 3:
                best = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '3')
                ].shape[0]
                second = data[
                    (data['Good A'] == '1') & (data['Good B'] == '3') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '2')
                ].shape[0]
                third = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2') & (data['Good D'] == '3') & (data['Good E'] == '3')
                ].shape[0]
                fourth = data[
                    (data['Good A'] == '1') & (data['Good B'] == '0') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '0')
                ].shape[0]
                fifth = data[
                    ((data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '1') |
                    (data['Good A'] == '1') & (data['Good B'] == '3') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '1') )
                ].shape[0]
                other = 100 - best - second - third - fourth - fifth
                self.metrics[f'q{question}']['order']['1st'].append(best) 
                self.metrics[f'q{question}']['order']['2nd'].append(second)
                self.metrics[f'q{question}']['order']['3rd'].append(third)
                self.metrics[f'q{question}']['order']['4th'].append(fourth)
                self.metrics[f'q{question}']['order']['5th'].append(fifth)
                usw = fifth
                self.metrics[f'q{question}']['order']['Other'].append(other)
                self.metrics[f'q{question}']['labels'] = ['EF\nRMM\nPO', 'RMM\nPO', '', '', 'USW', '']
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-5:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff]) 
                    if payoff_string == '40,52,52' and alloc[1] == '2':
                        payoff_string += '*'
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][payoff_string][-1] += 1
                self.metrics[f'q{question}']['alloc_labels'] = ['EF,RMM\nPO', 'RMM\nPO', 'EF', 'USW', 'USW', '-', '-']
                self.metrics[f'q{question}']['notions']['IA'].append(0)
                self.metrics[f'q{question}']['notions']['EF'].append(data[
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '3') 
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['USW'].append(usw)
                self.metrics[f'q{question}']['notions']['PO'].append(data[
                        ((data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '3') |
                        (data['Good A'] == '1') & (data['Good B'] == '3') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '2') ) | 
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '1') |
                    (data['Good A'] == '1') & (data['Good B'] == '3') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '1') 
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['MAX'].append(0)
                self.metrics[f'q{question}']['notions']['RMM'].append(data[
                        ((data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '3') |
                        (data['Good A'] == '1') & (data['Good B'] == '3') & (data['Good C'] == '3') & (data['Good D'] == '2') & (data['Good E'] == '2') )
                    ].shape[0])

            elif question == 31:
                print(valuation_dict)
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-6:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff]) 
                    if payoff_string == '52,52,52' and alloc[1] == '2':
                        payoff_string += ' (EF)'
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][payoff_string][-1] += 1
                    # print(alloc)
                    # if alloc[0] == '1' and alloc[1] == '2' and alloc[3] == '3' and alloc[4] == '1' and alloc[5] == '2' and alloc[6] == '3':
                    #     print(payoff_string)
                self.metrics[f'q{question}']['alloc_labels'] = ['IA+EF\nRMM+PO', 'IA+RMM\n+PO', '-', '-', 'USW', 'USW', 'USW', 'USW', 'USW', 'USW', '-', '-', '-']
                print('-----------------')

            elif question == 32:
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-5:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff]) 
                    if payoff_string == '52,40,52' and alloc[2] == '2':
                        payoff_string += '*'
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][payoff_string][-1] += 1
                self.metrics[f'q{question}']['alloc_labels'] = ['EF,RMM\nPO', 'RMM\nPO', 'EF', 'USW', 'USW', '-', '-']

            elif question == 4:
                best = data[
                    (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') & (data['Good D'] == '0')
                ].shape[0]
                second = data[
                    (data['Good A'] == '3') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '0')
                ].shape[0]
                third = data[
                        (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') & (data['Good D'] == '3')
                    ].shape[0]
                fourth = data[
                        (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') & (data['Good D'] == '1')
                    ].shape[0]
                fifth = data[
                        (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') & (data['Good D'] == '2')
                    ].shape[0]
                other = 100 - best - second - third - fourth - fifth
                self.metrics[f'q{question}']['order']['1st'].append(best) 
                self.metrics[f'q{question}']['order']['2nd'].append(second)
                self.metrics[f'q{question}']['order']['3rd'].append(third)
                self.metrics[f'q{question}']['order']['4th'].append(fourth)
                self.metrics[f'q{question}']['order']['5th'].append(fifth)
                self.metrics[f'q{question}']['order']['Other'].append(other)
                usw = third + fourth + fifth
                self.metrics[f'q{question}']['labels'] = ['EF\nRMM', 'IA', 'USW\nRMM', 'USW\nRMM', 'USW\nRMM', '']
                self.metrics[f'q{question}']['alloc_labels'] = ['EF\nRMM', 'IA', 'USW\nRMM', 'USW', 'USW\nRMM', '-', '-', '-']
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-4:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
                    # else:
                    #     key = list(self.metrics[f'q{question}']['allocations'].keys())[0]
                    #     pos = len(self.metrics[f'q{question}']['allocations'][key])-1
                    #     self.metrics[f'q{question}']['allocations'][payoff_string] = [0]*pos + [1]
                    #     self.metrics[f'q{question}']['alloc_labels'].append('-')
                self.metrics[f'q{question}']['notions']['IA'].append(data[
                        (data['Good A'] == '3') & (data['Good B'] == '1') & (data['Good C'] == '2')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['EF'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') 
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['USW'].append(usw)
                self.metrics[f'q{question}']['notions']['PO'].append(usw)
                self.metrics[f'q{question}']['notions']['MAX'].append(0)
                self.metrics[f'q{question}']['notions']['RMM'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') 
                    ].shape[0])

            elif question in [41,42]:
                self.metrics[f'q{question}']['alloc_labels'] = ['EF+RMM', 'IA', 'USW', 'USW', 'USW\n+RMM', '-']
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-4:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
                    
                    else:
                        key = list(self.metrics[f'q{question}']['allocations'].keys())[0]
                        pos = len(self.metrics[f'q{question}']['allocations'][key])-1
                        self.metrics[f'q{question}']['allocations'][payoff_string] = [0]*pos + [1]
                        self.metrics[f'q{question}']['alloc_labels'].append('-')
                    
                    # if alloc[0] == '2' and alloc[1] == '1' and alloc[2] == '3': 
                    #     print(idx)
                    #     print(payoff_string)

            elif question == 43:
                self.metrics[f'q{question}']['alloc_labels'] = ['EF\nRMM', 'IA', 'USW\nRMM', 'USW', 'USW\nRMM', '-', '-', '-']
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-4:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
            
            elif question == 5:
                best = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '1') & (data['Good D'] == '3') & (data['Good E'] == '3') & (data['Good F'] == '2')
                ].shape[0]
                second = data[
                    (data['Good A'] == '3') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3') & (data['Good E'] == '1') & (data['Good F'] == '2')
                ].shape[0]
                third = data[
                    (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') & (data['Good D'] == '2') & (data['Good E'] == '3') & (data['Good F'] == '1')
                ].shape[0]
                fourth = data[
                    (data['Good A'] == '2') & (data['Good B'] == '0') & (data['Good C'] == '1') & (data['Good D'] == '0') & (data['Good E'] == '3') & (data['Good F'] == '0')
                ].shape[0]
                fifth = data[
                    (data['Good A'] == '2') & (data['Good B'] == '2') & (data['Good C'] == '1') & (data['Good D'] == '1') & (data['Good E'] == '3') & (data['Good F'] == '3')
                ].shape[0]
                other = 100 - best - second - third - fourth - fifth
                self.metrics[f'q{question}']['order']['1st'].append(best) 
                self.metrics[f'q{question}']['order']['2nd'].append(second)
                self.metrics[f'q{question}']['order']['3rd'].append(third) 
                self.metrics[f'q{question}']['order']['4th'].append(fourth)
                self.metrics[f'q{question}']['order']['5th'].append(fifth)
                self.metrics[f'q{question}']['order']['Other'].append(other)
                self.metrics[f'q{question}']['labels'] = ['EF\nRMM\nUSW', 'IA', 'EF', 'EF', '', '']
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                self.metrics[f'q{question}']['alloc_labels'] = ['EF,RMM\nUSW', 'EF (RR)', 'EF (SRR)', '-', '-', '-', '-', '-', '-', '-', '-']
                
                other += fifth
                self.metrics[f'q{question}']['options']['1st'].append(best) 
                self.metrics[f'q{question}']['options']['2nd'].append(second)
                self.metrics[f'q{question}']['options']['3rd'].append(third) 
                self.metrics[f'q{question}']['options']['4th'].append(fourth)
                self.metrics[f'q{question}']['options']['Other'].append(other)
                self.metrics[f'q{question}']['option_labels'] = ['EF\nRMM\nUSW', 'EQ', 'EF', 'EF', 'Oth.']
                
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-6:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
                    # else:
                    #     key = list(self.metrics[f'q{question}']['allocations'].keys())[0]
                    #     pos = len(self.metrics[f'q{question}']['allocations'][key])-1
                    #     self.metrics[f'q{question}']['allocations'][payoff_string] = [0]*pos + [1]
                    #     self.metrics[f'q{question}']['alloc_labels'].append('-')
                self.metrics[f'q{question}']['notions']['IA'].append(data[
                        (data['Good A'] == '3') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3') & (data['Good E'] == '1') & (data['Good F'] == '2')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['EF'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '1') & (data['Good D'] == '3') & (data['Good E'] == '3') & (data['Good F'] == '2') |
                        (data['Good A'] == '2') & (data['Good B'] == '3') & (data['Good C'] == '1') & (data['Good D'] == '2') & (data['Good E'] == '3') & (data['Good F'] == '1') |
                        (data['Good A'] == '2') & (data['Good B'] == '0') & (data['Good C'] == '1') & (data['Good D'] == '0') & (data['Good E'] == '3') & (data['Good F'] == '0')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['USW'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '1') & (data['Good D'] == '3') & (data['Good E'] == '3') & (data['Good F'] == '2')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['PO'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '1') & (data['Good D'] == '3') & (data['Good E'] == '3') & (data['Good F'] == '2')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['MAX'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '1') & (data['Good D'] == '3') & (data['Good E'] == '3') & (data['Good F'] == '2')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['RMM'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '1') & (data['Good D'] == '3') & (data['Good E'] == '3') & (data['Good F'] == '2')
                    ].shape[0])
                
            elif question in [51,52]:
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                if question == 51:
                    self.metrics[f'q{question}']['alloc_labels'] = ['EF,RMM\nUSW', 'EF (RR)', 'EF (SRR)', '-', '-', '-', '-', '-', '-', '-', '-']    
                elif question == 52:
                    self.metrics[f'q{question}']['alloc_labels'] = ['EF,RMM\nUSW', 'EF', 'EF']     
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-6:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
                    # else:
                    #     key = list(self.metrics[f'q{question}']['allocations'].keys())[0]
                    #     pos = len(self.metrics[f'q{question}']['allocations'][key])-1
                    #     self.metrics[f'q{question}']['allocations'][payoff_string] = [0]*pos + [1]
                    #     self.metrics[f'q{question}']['alloc_labels'].append('-')

            elif question == 6:
                best = data[
                    (data['Good A'] == '2') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '1')
                ].shape[0]
                second = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2') & (data['Good D'] == '3')
                ].shape[0]
                third = data[
                    (data['Good A'] == '1') & (data['Good B'] == '0') & (data['Good C'] == '2') & (data['Good D'] == '3') 
                ].shape[0]
                fourth = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '3') 
                ].shape[0]
                fifth = data[
                    (data['Good A'] == '1') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '3') 
                ].shape[0]
                other = 100 - best - second - third - fourth - fifth
                self.metrics[f'q{question}']['order']['1st'].append(best) 
                self.metrics[f'q{question}']['order']['2nd'].append(second)
                self.metrics[f'q{question}']['order']['3rd'].append(third) 
                self.metrics[f'q{question}']['order']['4th'].append(fourth)
                self.metrics[f'q{question}']['order']['5th'].append(fifth)
                self.metrics[f'q{question}']['order']['Other'].append(other)
                self.metrics[f'q{question}']['labels'] = ['IA', 'EF\nRMM\nPO', 'EF', 'USW', 'PO', '']
                
                other += fifth
                self.metrics[f'q{question}']['options']['1st'].append(best) 
                self.metrics[f'q{question}']['options']['2nd'].append(second)
                self.metrics[f'q{question}']['options']['3rd'].append(third) 
                self.metrics[f'q{question}']['options']['4th'].append(fourth)
                self.metrics[f'q{question}']['options']['Other'].append(other)
                self.metrics[f'q{question}']['option_labels'] = ['EQ', 'EF\nRMM\nPO', 'EF', 'USW', 'Oth.']
                
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                self.metrics[f'q{question}']['alloc_labels'] = ['EQ', 'EF,RMM\nPO', 'EF', 'PO', 'USW', 'PO', '']
                total = 0
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-4:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
                        total += 1
                self.metrics[f'q{question}']['allocations']['Others'][-1] += (100-total)
                    # else:
                    #     key = list(self.metrics[f'q{question}']['allocations'].keys())[0]
                    #     pos = len(self.metrics[f'q{question}']['allocations'][key])-1
                    #     self.metrics[f'q{question}']['allocations'][payoff_string] = [0]*pos + [1]
                    #     self.metrics[f'q{question}']['alloc_labels'].append('-')
                self.metrics[f'q{question}']['notions']['IA'].append(data[
                        (data['Good A'] == '2') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '1')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['EF'].append(data[
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2') & (data['Good D'] == '3') |
                        (data['Good A'] == '1') & (data['Good B'] == '0') & (data['Good C'] == '2') & (data['Good D'] == '3')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['USW'].append(data[
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '3')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['PO'].append(data[
                        ((data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2') & (data['Good D'] == '3') |
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '3'))
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['MAX'].append(data[
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '3')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['RMM'].append(data[
                        (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '2') & (data['Good D'] == '3')
                    ].shape[0])

            elif question in [61,62]:
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                self.metrics[f'q{question}']['alloc_labels'] = ['EF,RMM\nPO', 'EF (SRR)', 'USW', '-', '-', '-', '-', '-', '-', '-', '-']
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-4:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        try:
                            recipient = int(r)
                        except:
                            continue
                        if recipient: payoff[recipient-1] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
                    # else:
                    #     key = list(self.metrics[f'q{question}']['allocations'].keys())[0]
                    #     pos = len(self.metrics[f'q{question}']['allocations'][key])-1
                    #     self.metrics[f'q{question}']['allocations'][payoff_string] = [0]*pos + [1]
                    #     self.metrics[f'q{question}']['alloc_labels'].append('-')

            elif question == 7:
                eq = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Person 1 money'] == '0') & (data['Person 2 money'] == '5') & (data['Person 3 money'] == '0')
                ].shape[0]
                ef = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Person 1 money'] == '0') & (data['Person 2 money'] == '0') & (data['Person 3 money'] == '5')
                ].shape[0]
                third = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Person 1 money'] == '5') & (data['Person 2 money'] == '0') & (data['Person 3 money'] == '0')
                ].shape[0]
                fourth = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '1') & (data['Person 1 money'] == '0') & (data['Person 2 money'] == '0') & (data['Person 3 money'] == '5')
                ].shape[0]
                fifth = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Person 1 money'] == '5') & (data['Person 2 money'] == '0') & (data['Person 3 money'] == '0')
                ].shape[0]
                other = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3')  
                ].shape[0]
                second = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '1')  
                ].shape[0]
                usw = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '3')  
                ].shape[0]

                other -= (eq + ef)
                others = 100 - eq-ef-other-second-usw
                self.metrics[f'q{question}']['order']['1st'].append(eq) 
                self.metrics[f'q{question}']['order']['2nd'].append(ef)
                self.metrics[f'q{question}']['order']['3rd'].append(other) 
                self.metrics[f'q{question}']['order']['4th'].append(second)
                self.metrics[f'q{question}']['order']['5th'].append(usw)
                self.metrics[f'q{question}']['order']['Other'].append(others)
                self.metrics[f'q{question}']['labels'] = ['IA', 'EF', '-', '-', 'USW', '-']
                
                other = 100 - (eq + ef + third + fourth + fifth)
                self.metrics[f'q{question}']['options']['1st'].append(eq) 
                self.metrics[f'q{question}']['options']['2nd'].append(ef)
                self.metrics[f'q{question}']['options']['3rd'].append(third) 
                self.metrics[f'q{question}']['options']['4th'].append(fourth)
                self.metrics[f'q{question}']['options']['5th'].append(fifth)
                self.metrics[f'q{question}']['options']['Other'].append(other)
                self.metrics[f'q{question}']['option_labels'] = ['EQ\nRMM\nPO', 'EF\nPO', 'NA', 'NA', 'USW', 'Oth.']

                first = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '3') & (data['Person 1 money'] == '5') & (data['Person 2 money'] == '0') & (data['Person 3 money'] == '0')
                ].shape[0]
                second = data[
                    (data['Good A'] == '2') & (data['Good B'] == '1') & (data['Good C'] == '3') & (data['Person 1 money'] == '0') & (data['Person 2 money'] == '5') & (data['Person 3 money'] == '0')
                ].shape[0]
                third = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '1') & (data['Person 1 money'] == '5') & (data['Person 2 money'] == '0') & (data['Person 3 money'] == '0')
                ].shape[0]
                fourth = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '1') & (data['Person 1 money'] == '0') & (data['Person 2 money'] == '5') & (data['Person 3 money'] == '0')
                ].shape[0]
                fifth = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '1') & (data['Person 1 money'] == '0') & (data['Person 2 money'] == '0') & (data['Person 3 money'] == '5')
                ].shape[0]

                if 'bads' in self.metrics[f'q{question}']:
                    other = 100 - (first + second + third + fourth + fifth)
                    self.metrics[f'q{question}']['bads']['1st'].append(first) 
                    self.metrics[f'q{question}']['bads']['2nd'].append(second)
                    self.metrics[f'q{question}']['bads']['3rd'].append(third) 
                    self.metrics[f'q{question}']['bads']['4th'].append(fourth)
                    self.metrics[f'q{question}']['bads']['5th'].append(fifth)
                    self.metrics[f'q{question}']['bads']['Other'].append(other)
                    self.metrics[f'q{question}']['ranges'] = ['10', '15', '20', '25', '30', 'Oth.']
                
                self.metrics[f'q{question}']['notions']['IA'].append(eq)
                self.metrics[f'q{question}']['notions']['EF'].append(ef)
                self.metrics[f'q{question}']['notions']['USW'].append(data[
                        (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '3')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['PO'].append(data[
                        ((data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Person 1 money'] == '0') & (data['Person 2 money'] == '0') & (data['Person 3 money'] == '5') |
                        (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '3'))
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['MAX'].append(0)
                self.metrics[f'q{question}']['notions']['RMM'].append(eq)

            elif question == 8:
                best = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '2') & (data['Good D'] == '1')
                ].shape[0]
                second = data[
                    (data['Good A'] == '1') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '2')
                ].shape[0]
                third = data[
                    (data['Good A'] == '3') & (data['Good B'] == '0') & (data['Good C'] == '2') & (data['Good D'] == '1')
                ].shape[0]
                fourth = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == '3') & (data['Good D'] == '1')
                ].shape[0]
                fifth = data[
                    (data['Good A'] == '3') & (data['Good B'] == '1') & (data['Good C'] == '2') & (data['Good D'] == '1')
                ].shape[0]
                other = 100-best-second-third-fourth-fifth
                self.metrics[f'q{question}'][variation] = {
                } 
                self.metrics[f'q{question}']['order']['1st'].append(best) 
                self.metrics[f'q{question}']['order']['2nd'].append(second)
                self.metrics[f'q{question}']['order']['3rd'].append(third) 
                self.metrics[f'q{question}']['order']['4th'].append(fourth)
                self.metrics[f'q{question}']['order']['5th'].append(fifth) 
                self.metrics[f'q{question}']['order']['Other'].append(other) 
                self.metrics[f'q{question}']['labels'] = ['EF\nRMM\nPO', 'IA', '-', 'USW', '~IA', '-']
                self.metrics[f'q{question}']['notions']['IA'].append(second)
                self.metrics[f'q{question}']['notions']['EF'].append(best)
                self.metrics[f'q{question}']['notions']['USW'].append(fourth)
                self.metrics[f'q{question}']['notions']['PO'].append(best+fourth)
                self.metrics[f'q{question}']['notions']['MAX'].append(0)
                self.metrics[f'q{question}']['notions']['RMM'].append(best)            

            elif question == 9:
                best = data[
                    (data['Good A'] == '3') & (data['Good B'] == 'You') & (data['Good C'] == 'You') & (data['Good D'] == '2')
                ].shape[0]
                second = data[
                    (data['Good A'] == 'You') & (data['Good B'] == '2') & (data['Good C'] == 'You') & (data['Good D'] == '3')
                ].shape[0]
                third = data[
                    (data['Good A'] == '3') & (data['Good B'] == 'You') & (data['Good C'] == '0') & (data['Good D'] == '2')
                ].shape[0]
                usw = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == 'You') & (data['Good D'] == '2') |
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == 'Me') & (data['Good D'] == '2') |
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == 'me') & (data['Good D'] == '2') 
                ].shape[0]
                fifth = data[
                    (data['Good A'] == '3') & (data['Good B'] == 'You') & (data['Good C'] == '2') & (data['Good D'] == '3')
                ].shape[0]
                other = 100-best-second-third-usw-fifth
                self.metrics[f'q{question}']['order']['1st'].append(best) 
                self.metrics[f'q{question}']['order']['2nd'].append(second)
                self.metrics[f'q{question}']['order']['3rd'].append(third) 
                self.metrics[f'q{question}']['order']['4th'].append(usw) 
                self.metrics[f'q{question}']['order']['5th'].append(fifth) 
                self.metrics[f'q{question}']['order']['Other'].append(other) 
                self.metrics[f'q{question}']['labels'] = ['EF\nRMM\nPO', 'IA', 'EF', 'USW', 'PO', '-']
                self.metrics[f'q{question}']['notions']['IA'].append(data[
                        (data['Good A'] == 'You') & (data['Good B'] == '2') & (data['Good C'] == 'You') & (data['Good D'] == '3')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['EF'].append(data[
                        (data['Good A'] == '3') & (data['Good B'] == 'You') & (data['Good C'] == 'You') & (data['Good D'] == '2') |
                        (data['Good A'] == '3') & (data['Good B'] == 'You') & (data['Good C'] == '0') & (data['Good D'] == '2')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['USW'].append(0)
                self.metrics[f'q{question}']['notions']['PO'].append(data[
                        (data['Good A'] == '3') & (data['Good B'] == 'You') & (data['Good C'] == 'You') & (data['Good D'] == '2')
                    ].shape[0])
                self.metrics[f'q{question}']['notions']['MAX'].append(0)
                self.metrics[f'q{question}']['notions']['RMM'].append(data[
                        (data['Good A'] == '3') & (data['Good B'] == 'You') & (data['Good C'] == 'You') & (data['Good D'] == '2')
                    ].shape[0])
                
                total = 0
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-4:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        if r.isdigit():
                            recipient = int(r)
                        elif r in ['You', 'Me', 'Myself']:
                            recipient = 0
                        if recipient == 1: recipient = 0
                        if recipient: 
                            payoff[recipient-1] += valuation_dict[recipient-1][i]
                        elif not r.isdigit():
                            payoff[recipient] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
                        total += 1
                self.metrics[f'q{question}']['allocations']['Others'][-1] += (100-total)
                self.metrics[f'q{question}']['alloc_labels'] = ['EF,RMM\nPO', 'EQ', 'EF', 'PO', 'USW', 'PO', '']

            elif question == 91:
                total
                for key in self.metrics[f'q{question}']['allocations']:
                    self.metrics[f'q{question}']['allocations'][key].append(0)
                for idx in range(len(data)):
                    alloc = data.iloc[idx].values[-4:] 
                    payoff = [0,0,0]
                    for i, r in enumerate(alloc):
                        if r.isdigit():
                            recipient = int(r)
                        elif r in ['You', 'Me', 'Myself']:
                            recipient = 0
                        if recipient == 1: recipient = 0
                        if recipient: 
                            payoff[recipient-1] += valuation_dict[recipient-1][i]
                        elif not r.isdigit():
                            payoff[recipient] += valuation_dict[recipient-1][i]
                    payoff_string = ','.join([str(p) for p in payoff])
                    if payoff_string in self.metrics[f'q{question}']['allocations']:
                        self.metrics[f'q{question}']['allocations'][','.join([str(p) for p in payoff])][-1] += 1
                self.metrics[f'q{question}']['alloc_labels'] = ['EF,RMM\nPO', 'EF', 'PO', 'USW', 'PO']
                    
            elif question == 10:
                eq = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == 'You') & ((data['Person 2 money'] == '8') | (data['Person 2 money'] == '9'))
                ].shape[0]
                ef = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == 'You') & (data['You money'] == '9') 
                ].shape[0]
                others = data[
                    (data['Good A'] == '3') & (data['Good B'] == '2') & (data['Good C'] == 'You')  
                ].shape[0]
                second = data[
                    (data['Good A'] == 'You') & (data['Good B'] == '2') & (data['Good C'] == '3')  
                ].shape[0]
                others -= (eq + ef)
                other = 100-eq-ef-others-second
                self.metrics[f'q{question}']['order']['Best+EQ'].append(eq) 
                self.metrics[f'q{question}']['order']['Best+EF'].append(ef)
                self.metrics[f'q{question}']['order']['Best+Other'].append(others) 
                self.metrics[f'q{question}']['order']['2nd'].append(second) 
                self.metrics[f'q{question}']['order']['Other'].append(other) 
                self.metrics[f'q{question}']['labels'] = ['IA', 'EF', '-', '-', '-']
                self.metrics[f'q{question}']['notions']['IA'].append(eq)
                self.metrics[f'q{question}']['notions']['USW'].append(0)
                self.metrics[f'q{question}']['notions']['EF'].append(ef)
                self.metrics[f'q{question}']['notions']['PO'].append(0)
                self.metrics[f'q{question}']['notions']['RMM'].append(0)
                self.metrics[f'q{question}']['notions']['MAX'].append(0)                

    def plot_comparion(self, question=1, variation="valuation", prefix_string = ''):
        num_options = len(self.variation_dict[variation])
        f, axes = plt.subplots(1, num_options, sharey=True, figsize=(16,6))

        for i, var in enumerate(self.variation_dict[variation]):
            option = prefix_string+str(var)
            option = option.rstrip('_')
            self.get_comparison(question, option)
            X = list(self.metrics[f'q{question}'][option].keys()) 
            llms = [self.metrics[f'q{question}'][option][m][0] for m in X] 
            humans = [self.metrics[f'q{question}'][option][m][1] for m in X]  
            
            X_axis = np.arange(len(X)) 
            
            # axes[i].plot(x, y)
            axes[i].set_title(str(var).capitalize())

            axes[i].bar(X_axis - 0.2, llms, 0.4, label = 'LLM') 
            axes[i].bar(X_axis + 0.2, humans, 0.4, label = 'Humans') 
            
            axes[i].set_xticks(X_axis, X) 
            axes[i].legend() 
            
        
        f.supxlabel("Top-k allocations") 
        f.supylabel("Percentage of Responses") 
        f.suptitle("Comparison between LLM and Human Responses")

    def plot_stacked_graph(
            self, 
            questions=[1], 
            option='word', 
            title = 'Comparison between LLM and Human Responses', 
            figsize=(12,6), 
            title_fontsize = 18, 
            labels_fontsize=16,
            ticks_fontsize=14,
            graph_labels_fontsize=12,
            legend_fontsize=12,
            question_ordering = {
                1: [0,1,2,3,4],
                2: [0,1,2,3,4],
                3: [0,1,2,3,4],
                4: [0,1,2,3,4],
                5: [0,1,2,3,4],
                6: [0,1,2,3,4],
                7: [0,1,2,3,4],
                9: [0,1,2,3,4],
            },
            humans=True,
            legend_pos=(0.97, 0.8),
            common_legend = True,

        ):
        f, axes = plt.subplots(nrows=1, ncols=len(questions), sharey=True, figsize=figsize)
        for j, question in enumerate(questions):
            self.get_comparison(question, option, humans=humans)
            ques_index = [self.index[ind] for ind in question_ordering[question]]
            new_order = {}
            for key in self.metrics[f'q{question}']['order']:
                new_order[key] = [self.metrics[f'q{question}']['order'][key][ind] for ind in question_ordering[question]]
            # print(f"new_order = {new_order}\nindex={ques_index}")
            df = pd.DataFrame(data=new_order, index=ques_index)
            ax = df.plot(
                kind="bar", 
                ax=axes[j], 
                stacked=True, 
                rot=0, 
                width=0.85, 
                legend=False, 
                color=['limegreen', 'palegreen', 'yellow', 'gold', 'orange', 'orangered'],
            ) 
            qtext = '{'+str(question)+'}'
            ax.set_xlabel(fr'$I_{qtext}$', fontsize=labels_fontsize)
            ax.set_ylabel('Percentage of Responses', fontsize=labels_fontsize)
            ax.set_yticklabels([t*20 for t in range(6)], fontsize=ticks_fontsize)
            # print(f"labels = {self.metrics[f'q{question}']['labels']}")
            for i, c in enumerate(ax.containers):
                labels = []
                lim = 12 if self.metrics[f'q{question}']['labels'][i].count('\n') < 2 else 15
                for k in question_ordering[question]:
                    labels.append(self.metrics[f'q{question}']['labels'][i] if list(self.metrics[f'q{question}']['order'].values())[i][k] > lim else '')
                ax.bar_label(c, labels=labels, label_type='center', fontsize=graph_labels_fontsize)
                ax.set_xticklabels(ques_index, fontsize=ticks_fontsize, rotation=45)
            
            handles, labels = ax.get_legend_handles_labels()
                

        if common_legend: f.legend(handles, labels, bbox_to_anchor=legend_pos, fontsize=legend_fontsize)

        # f.supxlabel("Top-k allocations", y=-0.14, fontsize=title_fontsize-2) 
        # f.suptitle(f"{title}", fontsize=title_fontsize)
        title_name = '_'.join(title.lower().split(' '))
        title_name = ''.join(letter for letter in title_name if letter.isalnum() or letter == '_')
        f.savefig(f"../Graphs/{title_name}.pdf", bbox_inches="tight")

    def get_iou(self, vector1, vector2):
        num, den = 0, 0
        for i in range(len(vector1)):
            num += min(vector1[i], vector2[i])
            den += max(vector1[i], vector2[i])
        
        return num/den
    
    def get_dp(self, vector1, vector2):
        ans = 0
        for i in range(len(vector1)):
            ans += vector1[i]*vector2[i]
        
        return ans/10000
    
    def pairwise_iou(self, questions=[1], option="word", prefix_string = '', humans=True):
        vector_dict = {}
        similarity = {}
        for i, question in enumerate(questions):
            print('----------------------------------------------------------')
            print(question)
            self.get_comparison(question, option, humans=humans)
            vector_dict[question] = {}
            similarity[question] = {}
            self.model_paths = ["humans"] + self.model_paths
            for k in range(len(self.model_paths)):
                vector = []
                for key, value in self.metrics[f'q{question}']['order'].items():
                    if key in self.metrics[f'q{question}']['relevant']: vector.append(value[k])
                vector_dict[question][self.model_paths[k]] = vector
                for j in range(k):
                    # print(vector_dict[question][self.model_paths[j]], vector_dict[question][self.model_paths[k]])
                    chi = self.get_iou(vector_dict[question][self.model_paths[k]], vector_dict[question][self.model_paths[j]])
                    dp = self.get_dp(vector_dict[question][self.model_paths[k]], vector_dict[question][self.model_paths[j]])
                    similarity[question][(self.model_paths[j],self.model_paths[k])] = (chi, dp)
                    print(question, self.model_paths[j], self.model_paths[k], vector_dict[question][self.model_paths[j]], vector_dict[question][self.model_paths[k]], chi, dp)

            for k in range(len(self.model_paths)-1):
                for j in range(len(vector_dict[question][self.model_paths[k]])):
                    v1 = vector_dict[question][self.model_paths[k]][j]
                    v2 = vector_dict[question][self.model_paths[k+1]][j]
                    data = [
                        [v1, 100-v1],
                        [v2, 100-v2]
                    ]
                    stat, pval = fisher_exact(data)
                    print(v1, v2, stat, pval)

            if humans: self.model_paths = self.model_paths[1:]

        return similarity
    
    def pairwise_chi(self, questions = [1], option="word", humans=False):
        vector_dict = {}
        similarity = {}
        for i, question in enumerate(questions):
            print('----------------------------------------------------------')
            print(question)
            self.get_comparison(question, option, humans=humans)
            vector_dict[question] = {}
            similarity[question] = {}
            if humans: self.model_paths = ["humans"] + self.model_paths
            for k in range(len(self.model_paths)):
                vector = []
                for key, value in self.metrics[f'q{question}']['order'].items():
                    if key in self.metrics[f'q{question}']['relevant']: vector.append(value[k])
                vector_dict[question][self.model_paths[k]] = vector
                for j in range(k):
                    # print(vector_dict[question][self.model_paths[j]], vector_dict[question][self.model_paths[k]])
                    chi = chisquare(vector_dict[question][self.model_paths[k]], vector_dict[question][self.model_paths[j]])
                    similarity[question][(self.model_paths[j],self.model_paths[k])] = chi
                    print(question, self.model_paths[j], self.model_paths[k], vector_dict[question][self.model_paths[j]], vector_dict[question][self.model_paths[k]], chi)

            for k in range(len(self.model_paths)-1):
                for j in range(len(vector_dict[question][self.model_paths[k]])):
                    v1 = vector_dict[question][self.model_paths[k]][j]
                    v2 = vector_dict[question][self.model_paths[k+1]][j]
                    data = [
                        [v1, 100-v1],
                        [v2, 100-v2]
                    ]
                    stat, pval = fisher_exact(data)
                    print(v1, v2, stat, pval)

            if humans: self.model_paths = self.model_paths[1:]

        return similarity

    def notion_detection(self, questions=[1], option='word', notion = 'EF'):
        gemini = []
        gpt = []
        for j, question in enumerate(questions):
            self.get_comparison(question, option)
            df = pd.DataFrame(data=self.metrics[f'q{question}']['order'], index=self.index)
            df.loc['gem_diff'] = df.loc[self.index[2]] - df.loc[self.index[1]]
            df.loc['gpt_diff'] = df.loc[self.index[4]] - df.loc[self.index[3]]
            if notion == 'EF':
                if question == 1:
                    df = df['1st']
                elif question == 2:
                    df = df['2nd']
                elif question == 3:
                    df['Diff'] = df['1st'] + df['4th']
                    df = df['Diff']
                elif question == 4:
                    df = df['1st']
                elif question == 5:
                    df['Diff'] = df['1st'] + df['3rd'] + df['4th']
                    df = df['Diff']
                elif question == 6:
                    df['Diff'] = df['2nd'] + df['3rd'] 
                    df = df['Diff']
                elif question == 7:
                    df = df['2nd']
                # elif question == 8:
                #     df = 
                elif question == 9:
                    df['Diff'] = df['1st'] + df['3rd']
                    df = df['Diff']
                elif question == 10:
                    df = df['2nd']
            elif notion == 'EQ':
                if question == 2:
                    df = df['1st']
                elif question == 4:
                    df = df['2nd']
                elif question == 5:
                    df = df['2nd']
                elif question == 6:
                    df = df['2nd']
                elif question == 6:
                    df['Diff'] = df['2nd'] + df['3rd'] 
                    df = df['Diff']
                elif question == 7:
                    df = df['1st']
                elif question == 9:
                    df = df['2nd']
                elif question == 10:
                    df = df['1st']
            if notion == 'USW':
                if question == 1:
                    df['Diff'] = df['2nd'] + df['3rd'] 
                    df = df['Diff']
                elif question == 2:
                    df = df['4th']
                elif question == 3:
                    df = df['USW']
                elif question == 4:
                    df = df['USW']
                elif question == 5:
                    df = df['1st'] 
                elif question == 6:
                    df = df['4th']
                elif question == 9:
                    df = df['4th']
            if notion == 'RMM':
                if question == 1:
                    df = df['2nd']
                elif question == 2:
                    df['Diff'] = df['1st'] + df['3rd'] 
                    df = df['Diff']
                elif question == 3:
                    df['Diff'] = df['1st'] + df['2nd']
                    df = df['Diff']
                elif question == 4:
                    df = df['1st']
                elif question == 5:
                    df = df['1st']
                elif question == 6:
                    df = df['4th']
                elif question == 9:
                    df = df['1st']
                

            gemini.append(df['gem_diff'])
            gpt.append(df['gpt_diff'])
            
        return gemini, gpt

    def plot_tweak_comparison(
            self, 
            questions=[1], 
            option='word', 
            title = 'Comparison between LLM and Human Responses', 
            common_legend = True, 
            figsize=(12,6), 
            title_fontsize = 18, 
            xlabels_fontsize=16,
            ylabels_fontsize=16,
            ticks_fontsize=14,
            graph_labels_fontsize=12,
            legend_fontsize=12,
            yticks = None,
            marks='None',
            colorlist=['tab:orange', 'tab:pink', 'tab:green', 'tab:red'],
            legend_pos = (1,1),
            xticks = True,
            bar_width=0.8,
            questions_colors = {
                2: ['limegreen', 'palegreen', 'yellow', 'gold', 'lightsalmon', 'coral', 'lightgrey', 'silver', 'lightsteelblue', 'plum'],
                6: ['limegreen', 'palegreen', 'yellow', 'gold', 'lightsalmon', 'coral', 'orangered'],
                9: ['limegreen', 'palegreen', 'yellow', 'gold', 'lightsalmon', 'coral', 'orangered'],
            },
            question_labels = {
                2: r'$I_2$',
                23: r'$I_2\;(a_2)$',
                24: r'$I_2\;(a_3)$',
                6: r'$I_6$',
                9: r'$I_9$',
                91: r'$I_9\;(a_2)$'
            },
            humans=False
        ):
        if not humans: self.index = self.index[1:]
        f, axes = plt.subplots(nrows=1, ncols=len(questions), sharey=True, figsize=figsize)
        # cm = plt.get_cmap('gist_rainbow')
        # colors = [cm(1.*i/50) for i in range(50)]
        colors = plt.cm.tab20(range(20))

        human_data = {
            6: [32.6, 28.1, 18.4, 2.6, 7.9, 0.4, 10],
            9: [34.1, 30, 17.6, 2.2, 4.1, 0, 12]
        }

        for j, question in enumerate(questions):
            # print(j, question, marks)
            self.get_comparison(question, option, humans=False)
            # print(self.metrics[f'q{question}'], self.index)

            if humans:
                for i, key in enumerate(self.metrics[f'q{question}']['allocations']):
                    self.metrics[f'q{question}']['allocations'][key] = [human_data[question][i]] + self.metrics[f'q{question}']['allocations'][key]
            
            # print(self.metrics[f'q{question}']['allocations'])
            df = pd.DataFrame(data=self.metrics[f'q{question}']['allocations'], index=self.index)
            if marks == 'fixed':
                ax = df.plot(kind="bar",ax=axes[j],stacked=True, rot=0, width=bar_width, legend=False, color=colorlist[question//10 if question > 10 else question])
            elif marks == "large":
                ax = df.plot(kind="bar",ax=axes[j],stacked=True, rot=0, width=0.8, legend=False, color=colors)
            else:
                ax = df.plot(kind="bar",ax=axes[j],stacked=True, rot=0, width=bar_width, legend=False, color=questions_colors[question//10 if question > 10 else question])
            qtext = '{'+str(question)+'}' if question <= 10 else '{'+str(question//10)+'.'+str(question%10)+'}'
            ax.set_xlabel(question_labels[question], fontsize=xlabels_fontsize)
            ax.set_ylabel('Percentage of Responses', fontsize=ylabels_fontsize)
            if yticks:
                ax.set_yticks([t*yticks[1] for t in range(yticks[0])])
                ax.set_yticklabels([t*yticks[1] for t in range(yticks[0])], fontsize=ticks_fontsize)
            # print(question, len(ax.containers), len(self.metrics[f'q{question}']['allocations']), len(self.metrics[f'q{question}']['alloc_labels']))
            for i, c in enumerate(ax.containers):
                labels = []
                for k in range(len(self.model_paths)+1):
                    # if 'EQ' in self.metrics[f'q{question}']['alloc_labels'][i]:
                    #     print(question, self.model_paths[k])
                    #     print(self.metrics[f'q{question}']['alloc_labels'][i], list(self.metrics[f'q{question}']['allocations'].values())[i][k])
                    #     print()
                    if question in [1,11,12,13]:
                        payoff = f"\n{self.metrics[f'q{question}']['payoffs'][list(self.metrics[f'q{question}']['allocations'].keys())[i][:8]]}" if list(self.metrics[f'q{question}']['allocations'].values())[i][k] > 10 else ''
                        # payoff = ''
                    elif question >= 14 and question <= 17:
                        payoff = ''
                    else:
                        payoff = f"{list(self.metrics[f'q{question}']['allocations'].keys())[i][:8]}" if list(self.metrics[f'q{question}']['allocations'].values())[i][k] > 5 else ''
                    
                    if question >= 14 and question <= 17:
                        graph_label =  f"{self.metrics[f'q{question}']['alloc_labels'][i]}{payoff}" if list(self.metrics[f'q{question}']['allocations'].values())[i][k] > 8 else ''
                    else:
                        if list(self.metrics[f'q{question}']['allocations'].values())[i][k] > 8:
                            graph_label = f"{self.metrics[f'q{question}']['alloc_labels'][i]}\n{payoff}"  
                        elif list(self.metrics[f'q{question}']['allocations'].values())[i][k] > 5:
                            graph_label = payoff
                        else: 
                            graph_label = ''
                    labels.append(graph_label)
                ax.bar_label(c, labels=labels, label_type='center', fontsize=graph_labels_fontsize)
                if xticks: 
                    ax.set_xticklabels(self.index, rotation=45, fontsize=ticks_fontsize)
                else:
                    ax.set_xticklabels(['']*len(self.index))

                handles, labels = ax.get_legend_handles_labels()

        if common_legend:
            f.legend(handles, labels, loc='upper right', fontsize=legend_fontsize, bbox_to_anchor=legend_pos)

        # f.supxlabel("Top-k allocations", y=-0.14, fontsize=title_fontsize-2) 
        # f.suptitle(title, fontsize=title_fontsize)
        title_name = '_'.join(title.lower().split(' '))
        title_name = ''.join(letter for letter in title_name if letter.isalnum() or letter == '_')
        f.savefig(f"../Graphs/{title_name}.pdf", bbox_inches="tight")
        if not humans: self.index = ["Humans"] + self.index

        # vector_dict = {}
        # similarity = {}
        # for i, question in enumerate(questions):
        #     self.get_comparison(question, option)
        #     vector_dict[question] = {}
        #     similarity[question] = {}
        #     self.model_paths = ['humans'] + self.model_paths
        #     for k in range(len(self.model_paths)):
        #         vector = []
        #         for key, value in self.metrics[f'q{question}']['order'].items():
        #             vector.append(value[k])
        #         vector_dict[question][self.model_paths[k]] = vector
        #         for j in range(k):
        #             print(vector_dict[question][self.model_paths[j]], vector_dict[question][self.model_paths[k]])
        #             chi = chisquare(vector_dict[question][self.model_paths[k]], vector_dict[question][self.model_paths[j]])
        #             similarity[question][(self.model_paths[j],self.model_paths[k])] = chi
        #             print(question, self.model_paths[j], self.model_paths[k], vector_dict[question][self.model_paths[j]], vector_dict[question][self.model_paths[k]], chi)
                
        #     self.model_paths = self.model_paths[1:]

        # print(similarity)

    def plot_options(
            self, 
            questions=[1], 
            option='word', 
            title = 'Fairest within Fair', 
            figsize=(12,6), 
            title_fontsize = 18, 
            xlabels_fontsize=16,
            ylabels_fontsize=16,
            ticks_fontsize=14,
            graph_labels_fontsize=12,
            yticks = None,
            xticks = True,
            bar_width=0.8,
            questions_colors = {
                19: ['lightgreen', 'lightskyblue', 'pink', 'gold', 'orange', 'orangered'],
                2: ['lightgreen', 'lightskyblue', 'pink', 'gold', 'orange', 'orangered'],
                5: ['lightskyblue', 'lightgreen', 'pink', 'gold', 'orange', 'orangered'],
                6: ['lightgreen', 'lightskyblue', 'skyblue', 'gold', 'orange', 'orangered'],
                7: ['lightgreen', 'lightskyblue', 'pink', 'orange', 'gold', 'orangered'],
            }, 
            option_type = 'fair'
        ):
        self.index = self.index[1:]
        f, axes = plt.subplots(nrows=1, ncols=len(questions), sharey=True, figsize=figsize)
        if option_type == 'fair':
            source = 'options'
            label_list = 'option_labels'
        else: 
            source = 'bads'
            label_list = 'ranges'

        for j, question in enumerate(questions):
            self.get_comparison(question, option, humans=False)
            
            df = pd.DataFrame(data=self.metrics[f'q{question}'][source], index=self.index)
            ax = df.plot(kind="bar",ax=axes[j],stacked=True, rot=0, width=bar_width, legend=False, 
                color=questions_colors[question])
            qtext = '{'+str(question)+'}' if question != 19 else '{0}'
            ax.set_xlabel(fr'$I_{qtext}$', fontsize=xlabels_fontsize)
            ax.set_ylabel('Percentage of Responses', fontsize=ylabels_fontsize)
            ax.set_yticklabels([t*20 for t in range(6)], fontsize=ticks_fontsize)
            if yticks:
                ax.set_yticks([t*yticks[1] for t in range(yticks[0])])
                ax.set_yticklabels([t*yticks[1] for t in range(yticks[0])], fontsize=ticks_fontsize)
            for i, c in enumerate(ax.containers):
                labels = []
                for k in range(len(self.model_paths)):
                    # print(question, self.index[k], list(self.metrics[f'q{question}'][source].values())[i][k])
                    lim = 10 if self.metrics[f'q{question}'][label_list][i].count('\n') < 1 else 15
                    graph_label =  f"{self.metrics[f'q{question}'][label_list][i]}" if list(self.metrics[f'q{question}'][source].values())[i][k] > lim else ''
                    
                    labels.append(graph_label)
                ax.bar_label(c, labels=labels, label_type='center', fontsize=graph_labels_fontsize)
                if xticks: 
                    ax.set_xticklabels(self.index, rotation=45, fontsize=ticks_fontsize)
                else:
                    ax.set_xticklabels(['']*len(self.index))

        # f.suptitle(title, fontsize=title_fontsize)
        title_name = '_'.join(title.lower().split(' '))
        title_name = ''.join(letter for letter in title_name if letter.isalnum() or letter == '_')
        f.savefig(f"../Graphs/{title_name}.pdf", bbox_inches="tight")
        self.index = ["Humans"] + self.index

    def plot_tweak_comparison_single(
            self, 
            sets, 
            title = 'Comparison between LLM and Human Responses', 
            common_legend = True, 
            figsize=(12,6), 
            title_fontsize = 18, 
            labels_fontsize=16,
            ticks_fontsize=14,
            graph_labels_fontsize=12,
            legend_fontsize=12,
            yticks = None,
            marks='None',
            colorlist=['tab:orange', 'tab:pink', 'tab:green', 'tab:red'],
            legend_pos = (1,1),
        ):
        self.index = self.index[1:]
        f, axes = plt.subplots(nrows=1, ncols=len(sets), sharey=True, figsize=figsize)
        dfs = {}
        # cm = plt.get_cmap('gist_rainbow')
        # colors = [cm(1.*i/50) for i in range(50)]
        colors = plt.cm.tab20(range(20))
        for j, set_ in enumerate(sets):
            print(set_)
            questions, model = set_
            for question in questions:
                self.get_comparison(question, model[1], model_paths_custom=[model], humans=False)
                # print(self.metrics[f'q{question}'], self.index)
                
                dfs[question] = pd.DataFrame(data=self.metrics[f'q{question}']['allocations'], index=self.index)

            for ques in dfs:
                print(dfs[ques].head())
                # if marks == 'fixed':
                #     ax = df.plot(kind="bar",ax=axes[j],stacked=True, rot=0, width=0.8, legend=False, color=colorlist)
                # elif marks == "large":
                #     ax = df.plot(kind="bar",ax=axes[j],stacked=True, rot=0, width=0.8, legend=False, color=colors)
                # else:
                #     ax = df.plot(kind="bar",ax=axes[j],stacked=True, rot=0, width=0.8, legend=False)
                # qtext = '{'+str(question)+'}' if question <= 10 else '{'+str(question//10)+'.'+str(question%10)+'}'
                # ax.set_xlabel(fr'$I_{qtext}$', fontsize=labels_fontsize)
                # ax.set_ylabel('Percentage of Responses', fontsize=labels_fontsize)
                # if yticks:
                #     ax.set_yticks([t*yticks[1] for t in range(yticks[0])])
                #     ax.set_yticklabels([t*yticks[1] for t in range(yticks[0])], fontsize=ticks_fontsize)
                # # print(question, len(ax.containers), len(self.metrics[f'q{question}']['allocations']), len(self.metrics[f'q{question}']['alloc_labels']))
                # for i, c in enumerate(ax.containers):
                #     labels = []
                #     for k in range(len(self.model_paths)):
                #         # if 'EQ' in self.metrics[f'q{question}']['alloc_labels'][i]:
                #         #     print(question, self.model_paths[k])
                #         #     print(self.metrics[f'q{question}']['alloc_labels'][i], list(self.metrics[f'q{question}']['allocations'].values())[i][k])
                #         #     print()
                #         if question in [1,11,12,13]:
                #             payoff = f"\n{self.metrics[f'q{question}']['payoffs'][list(self.metrics[f'q{question}']['allocations'].keys())[i][:8]]}" if list(self.metrics[f'q{question}']['allocations'].values())[i][k] > 10 else ''
                #             # payoff = ''
                #         elif question >= 14 and question <= 17:
                #             payoff = ''
                #         else:
                #             payoff = f"\n{list(self.metrics[f'q{question}']['allocations'].keys())[i][:8]}" if list(self.metrics[f'q{question}']['allocations'].values())[i][k] > 10 else ''
                        
                #         if question >= 14 and question <= 17:
                #             graph_label =  f"{self.metrics[f'q{question}']['alloc_labels'][i]}{payoff}" if list(self.metrics[f'q{question}']['allocations'].values())[i][k] > 7 else ''
                #         else:
                #             graph_label = f"{self.metrics[f'q{question}']['alloc_labels'][i]}{payoff}" if list(self.metrics[f'q{question}']['allocations'].values())[i][k] >= 10 else ''
                #         labels.append(graph_label)
                #     ax.bar_label(c, labels=labels, label_type='center', fontsize=graph_labels_fontsize)
                #     ax.set_xticklabels(self.index, rotation=45, fontsize=ticks_fontsize)

                #     handles, labels = ax.get_legend_handles_labels()

            # if common_legend:
            #     f.legend(handles, labels, loc='upper right', fontsize=legend_fontsize, bbox_to_anchor=legend_pos)

            # # f.supxlabel("Top-k allocations", y=-0.14, fontsize=title_fontsize-2) 
            # f.suptitle(title, fontsize=title_fontsize)
            # title_name = '_'.join(title.lower().split(' '))
            # title_name = ''.join(letter for letter in title_name if letter.isalnum() or letter == '_')
            # f.savefig(f"../Graphs/{title_name}.pdf", bbox_inches="tight")
            # self.index = ["Humans"] + self.index

    def plot_strategy_comparison(self, questions=[1], variation_dict = {'word': 'Word', 'word_best': 'Word | Best'}, option="word"):

        f, axes = plt.subplots(nrows=1, ncols=len(questions), sharey=True, figsize=(3*(len(questions)+(len(self.model_paths)+1)),6))
        # axes_ = []
        for j, question in enumerate(questions):
            self.get_comparison(question, option)
            df = pd.DataFrame(data=self.metrics[f'q{question}']['order'], index=self.index)
            ax = df.plot(kind="bar",ax=axes[j],stacked=True, rot=0, xlabel=f'Question {question}', ylabel='Percentage of Responses', width=0.8)
            for i, c in enumerate(ax.containers):
                labels = []
                for k in range(len(self.model_paths)+1):
                    labels.append(self.metrics[f'q{question}']['labels'][i] if list(self.metrics[f'q{question}']['order'].values())[i][k] > 5 else '')
                ax.bar_label(c, labels=labels, label_type='center')
                ax.set_xticklabels(self.index, rotation=45)
            # axes_.append(ax)
                
        f.supxlabel("Top-k allocations", y=-0.20) 
        # f.supylabel("Percentage of Responses") 
        f.suptitle(f"Comparison between LLM and Human Responses for Configuration {option}")

    def plot_preferred_notions(self, questions=[1], variation = 'word', notions = ['IA', 'EF', 'USW', 'PO', 'RMM'], transpose=True):
        notion_dict = {}
        for question in questions:
            self.get_comparison(question, variation)
            if not notion_dict:
                notion_dict = self.metrics[f'q{question}']['notions']
            else:
                for notion in notion_dict:
                    for i in range(len(notion_dict[notion])):
                        print(f"{self.index[i]}||{notion}||{question} = {self.metrics[f'q{question}']['notions'][notion][i]}")
                        notion_dict[notion][i] += self.metrics[f'q{question}']['notions'][notion][i]
                    print()
        notion_dict = {notion:notion_dict[notion] for notion in notion_dict if notion in notions}
        notion_df = pd.DataFrame(notion_dict, index=self.index)

        if transpose: 
            notion_df = notion_df.transpose()
            ax = notion_df.plot(kind="bar")
            print(notion_df.head())
            # ax.set_xticklabels(notion_df.columns, rotation=90)
        else:
            ax = notion_df.plot(kind="bar")     

    def get_rows(self):
        question_nums = []
        model_families = []
        model_versions = []
        percentage = []
        metric = []
        variations = []
        families = os.listdir('results')
        if '.DS_Store' in families: families.remove('.DS_Store')
        for family in families:
            if family not in ('gemini', 'chatgpt'): continue
            print(family)
            models = os.listdir(f'results/{family}')
            if '.DS_Store' in models: models.remove('.DS_Store')
            for model in models:
                print(model)
                questions = list(sorted(os.listdir(f'results/{family}/{model}')))
                for question in questions:
                    if 'question' not in question: 
                        print("Not consodering question 10")
                        continue
                    q = int(question.split('_')[1])
                    if q > 10:
                        continue
                    print(question, q)
                    strategies = os.listdir(f'results/{family}/{model}/{question}')
                    if '.DS_Store' in strategies: strategies.remove('.DS_Store')
                    for strategy in strategies:
                        # if strategy != "word.csv": continue
                        print(strategy)
                        self.get_comparison(q, strategy, [(f'{family}/{model}', strategy)], True)
                        humans = [self.metrics[f'q{q}']['order'][key][0] for key in self.metrics[f'q{q}']['order']]
                        humans.append(100-sum(humans))
                        llm_lim = [self.metrics[f'q{q}']['order'][key][1] for key in self.metrics[f'q{q}']['order']]
                        llm_lim.append(100-sum(llm_lim))
                        full_data = self.get_data(model_path=f'{family}/{model}',question=q, variation=strategy)
                        columns = list(full_data.columns[1:])
                        if 'Tokens' in columns: columns.remove('Tokens')
                        grouped = full_data.groupby(columns).size().sort_values(ascending=False)
                        llm_full = [int(val) for val in grouped.values]
                        # print(humans, llm_lim)

                        data = [
                            [sum(humans), sum(llm_lim)],
                            [100-sum(humans), 100-sum(llm_lim)]
                        ]

                        # chi = round(np.sum([((a - b) ** 2) / (a) for (a, b) in zip(humans, llm_lim)]),2)
                        # stat, pvalue = mannwhitneyu(humans, llm_lim)
                        stat, pvalue = chisquare(llm_lim, humans)
                        # stat, pvalue = fisher_exact(data)

                        # print(humans, llm_lim, pvalue, stat)
                        # print(data, stat, pvalue)

                        # for notion in self.metrics[f'q{q}']['notions']:
                        #     model_families.append(family)
                        #     model_versions.append(model)
                        #     question_nums.append(q)
                        #     variations.append(strategy)
                        #     percentage.append(self.metrics[f'q{q}']['notions'][notion][0])
                        #     metric.append(notion)

                        model_families.append(family)
                        model_versions.append(model)
                        question_nums.append(q)
                        variations.append(strategy)
                        metric.append('similarity')
                        percentage.append(stat)

                        # model_families.append(family)
                        # model_versions.append(model)
                        # question_nums.append(q)
                        # variations.append(strategy)
                        # metric.append('chi2')
                        # percentage.append(chi)

                        # chisq = chisquare(f_obs=llm_lim, f_exp=humans)
                        # print(chisq)

                        model_families.append(family)
                        model_versions.append(model)
                        question_nums.append(q)
                        variations.append(strategy)
                        metric.append('clarity')
                        percentage.append(round(statistics.stdev(llm_lim),2))
                        
        
        results = pd.DataFrame(
            zip(model_families, model_versions, question_nums, variations, metric, percentage),
            columns = ['LLM Type', 'Version', 'Question', 'Strategy', 'Metric', 'Percentage']
        )

        return results
                    




            

            






