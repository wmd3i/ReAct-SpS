from typing import Union, List, Tuple
from dataclasses import dataclass
import re
import string
import argparse
from llama import Llama
from llamassp import LlamaSSP
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
from time import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup


def process_hotpot_qa(hotpot, llm, args):
    # Processes each question in HotPotQA dataset using ReAct paradigm
    stop_word = 'Observation'
    docstore = DocstoreExplorer(Wikipedia())
    max_steps = 7
    now = datetime.now()
    filename = f"hotpotqa-{args.mode}-{now.strftime('%y%m%d%H%M')}.txt"
    action_str = '\nAction: '

    for idx, row in enumerate(hotpot):
        start_time = time()
        observation = None
        intermediate_steps = []  # (action, observation) tuple
        question = row['question']
        key = row['answer']
        write_question(idx+1, question, filename)
        for i in range(max_steps):
            scratchpad = construct_scratchpad(intermediate_steps)
            prompt = construct_prompt(question, scratchpad)
            try:
                llm_output, valid_action = get_llm_response(
                    llm, prompt, stop_word)
            except MemoryLimitExceededError as e:
                print(e)
                write_exception(e.extra_data, key, filename,
                                start_time, time())
                break  # stop due to GPU out of mem
            if valid_action:
                agent_action = resp_to_action(llm_output)
            else:
                llm_output = llm_output.split('\nAction')[0]
                print(f"Question: {idx+1}!!invalid action!!")
                try:
                    llm_action_output = get_llm_action(
                        llm, prompt, llm_output, stop_word)
                except MemoryLimitExceededError as e:
                    print(e)
                    write_exception(e.extra_data, key,
                                    filename, start_time, time())
                    break  # stop due to GPU out of mem
                agent_action = resp_to_action(
                    f'{llm_output}{action_str}{llm_action_output}')
            write_agent_action(agent_action, filename, key, start_time, time())
            if agent_action.tool == 'Finish':
                break
            observation = perform_action(docstore, agent_action)
            intermediate_steps.append((agent_action, observation))
            write_observation(agent_action, observation, filename)
            
def process_fever(fever, llm, args):
    # Processes each question in FEVER dataset using ReAct paradigm
    stop_word = 'Observation'
    docstore = DocstoreExplorer(Wikipedia())
    max_steps = 7
    now = datetime.now()
    filename = f"fever-{args.mode}-{now.strftime('%y%m%d%H%M')}.txt"
    action_str = '\nAction: '

    for idx, row in enumerate(fever):
        start_time = time()
        observation = None
        intermediate_steps = []  # (action, observation) tuple
        claim = row['claim']
        label = row['label']
        write_claim(idx+1, claim, filename)
        for i in range(max_steps):
            scratchpad = construct_scratchpad(intermediate_steps)
            prompt = construct_fever_prompt(claim, scratchpad)
            print("line 86 prompt: ", prompt)
            try:
                llm_output, valid_action = get_llm_response(
                    llm, prompt, stop_word)
                print("line 90 llm_output: ", llm_output)
            except MemoryLimitExceededError as e:
                print(e)
                write_exception(e.extra_data, label, filename,
                                start_time, time())
                break  # stop due to GPU out of mem
            if valid_action:
                agent_action = resp_to_action(llm_output)
            else:
                llm_output = llm_output.split('\nAction')[0]
                print("line 98 after llm_output: ", llm_output)
                print(f"Claim: {idx+1}!!invalid action!!")
                try:
                    llm_action_output = get_llm_action(
                        llm, prompt, llm_output, stop_word)
                    print("line 104 llm_action_output: ", llm_action_output)
                except MemoryLimitExceededError as e:
                    print(e)
                    write_exception(e.extra_data, label,
                                    filename, start_time, time())
                    break  # stop due to GPU out of mem
                agent_action = resp_to_action(
                    f'{llm_output}{action_str}{llm_action_output}')
            write_agent_action(agent_action, filename, label, start_time, time())
            if agent_action.tool == 'Finish':
                break
            observation = perform_action(docstore, agent_action)
            intermediate_steps.append((agent_action, observation))
            write_observation(agent_action, observation, filename)


def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
        sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])


  
def search_step(Docstore, entity):
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    response_text = requests.get(search_url).text
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:  # mismatch
      result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
      obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
    else:
        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if any("may refer to:" in p for p in page):
            return search_step(Docstore, "[" + entity + "]")
        else:
            self_page = ""
            for p in page:
                if len(p.split(" ")) > 2:
                    self_page += clean_str(p)
                    if not p.endswith("\n"):
                        self_page += "\n"
            obs = get_page_obs(self_page)
    return obs


def parse_arguments():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", help="Specify the mode to use: 'llama2' or 'sps'")
    parser.add_argument(
        "--dataset", help="Specify the dataset to use: 'hotpot' or 'fever'")
    parser.add_argument("--compare", action='store_true',
                        help="Compare sps against baseline in basic text generation task")
    return parser.parse_args()


def initialize_llm(args):
    # Initializes the LLM based on the specified mode
    if args.mode == 'llama2':
        print("Using Llama2 70b model with 4-bit quantization")
        return Llama()
    elif args.mode == 'sps':
        print("Using speculative sampling (Llama2 70b + Llama2 7b) with 4-bit quantization")
        return LlamaSSP()
    else:
        raise ValueError("Invalid mode specified. Use 'llama2' or 'sps'.")


def compare_model_performance():
    # Compares performance between Llama2-70b and Speculative Sampling
    llama2_llm = Llama()
    ssp_llm = LlamaSSP()
    print("--- Llama2 70b model with 4-bit quantization ---")
    llama2_gen_ids, llama2_ms_per_token = llama2_llm.measure(texts)
    print("--- Speculative sampling (Llama2 70b + Llama2 7b) with 4-bit quantization ---")
    ssp_gen_ids, ssp_ms_per_token = ssp_llm.measure(texts)
    llama2_llm.print_results(llama2_ms_per_token, llama2_gen_ids)
    ssp_llm.print_results(ssp_ms_per_token, ssp_gen_ids)


class MemoryLimitExceededError(Exception):
    # Custom exception class for GPU memory limit exceeded scenarios.
    def __init__(self, message, extra_data):
        super().__init__(message)
        self.extra_data = extra_data


@dataclass
class AgentAction:
    # Data class representing an action an agent will execute

    tool: str
    # The name of the Tool to execute (Search/Lookup/Finish)
    tool_input: Union[str, dict]
    # The input to pass in to the Tool
    log: str
    # Complete log of the action


def perform_search(docstore, tool_input):
    # Perform a search using the docstore
    try:
        return format_step(docstore.search(tool_input))
    except Exception:
        return 'Could not find that page, please try again.'


def perform_lookup(docstore, tool_input):
    # Perform a lookup using the docstore
    try:
        return format_step(docstore.lookup(tool_input))
    except ValueError:
        return 'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'


def perform_action(docstore, action: AgentAction):
    # Mapping of tool names to their corresponding functions
    print("line 235 perform_action: ", action.tool, action.tool_input)
    action_mapping = {
        'Search': search_step,
        'Lookup': perform_lookup,
    }
    action_function = action_mapping.get(action.tool)
    # If the action tool is not recognized or is 'Finish', return None
    if not action_function:
        print("line 243 action_function not recognized")
        return None
    return action_function(docstore, action.tool_input)


def construct_scratchpad(intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
    # Construct the scratchpad to allow the agent continue its thought process.
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts


observation_prefix = 'Observation: '
llm_prefix = 'Thought: '
action_prefix = '''Action can be three types: 
Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
Finish[answer], which returns the answer and finishes the task.
Action: '''


def resp_to_action(resp):
    # Parse the LLM output to agent action
    print("!!resp: ", resp)
    # Using regex to split on ' Action:' or '\nAction:'
    parts = re.split(r'\s*Action:', resp.strip())
    print("!!parts :", parts)
    tool = parts[1].split('[')[0].strip()
    print("!!tool: ", tool)
    tool_input = parts[1].split('[')[1].rstrip(']')
    tool_input = tool_input.split(']')[0]
    action_log = f"{parts[0]}\nAction: {tool}[{tool_input}]" if parts else resp
    agent_action = AgentAction(tool, tool_input, action_log)
    return agent_action


def check_valid_action(llm_output):
    # Define regex patterns for valid actions
    search_pattern = r'Action: Search\[[^\]]*\]'  # Matches 'Action: Search[xxx]'
    lookup_pattern = r'Action: Lookup\[[^\]]*\]'  # Matches 'Action: Lookup[xxx]'
    finish_pattern = r'Action: Finish\[[^\]]*\]'  # Matches 'Action: Finish[xxx]'
    combined_pattern = f'({search_pattern})|({lookup_pattern})|({finish_pattern})'
    return re.search(combined_pattern, llm_output) is not None

def format_step(step: str) -> str:
    return step.strip()


def get_llm_response(llm, prompt, stop):
    # Get the next response from LLM
    try:
        resp = llm(prompt, 500, stop=stop)
        resp = format_step(resp)
        print("line 296 resp: ", resp)
    except MemoryLimitExceededError as e:
        print("get_llm_response: ", e)
        raise
    return resp, check_valid_action(resp)


def get_llm_action(llm, prompt, llm_thought, stop):
    # Get the next action from LLM
    prompt = f'{prompt}{llm_thought}\n{action_prefix}'
    print("line 306 prompt: ", prompt)
    try:
        resp = llm(prompt, 500, stop=stop)
        resp = format_step(resp)
        print("line 310 resp: ", resp)
    except MemoryLimitExceededError as e:
        print("get_llm_action: ", e)
        raise
    return resp


def is_correct(answer, key) -> bool:
    return EM(answer, key)


def EM(answer, key) -> bool:
    # Exact match
    ans_norm = normalize_answer(answer)
    key_norm = normalize_answer(key)
    return ans_norm == key_norm


def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text):
    return " ".join(text.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def lower(text):
    return text.lower()


def normalize_answer(s):
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def construct_prompt(input_question: str, agent_scratchpad: str) -> str:
    # Construct the prompt for LLM
    if agent_scratchpad == '':
        constructed_prompt = f"{prompt_prefix}\n\n\nQuestion: {input_question}\n{llm_prefix}"
    else:
        constructed_prompt = f"{prompt_prefix}\n\n\nQuestion: {input_question}\n{llm_prefix}{agent_scratchpad}"
    return constructed_prompt

def construct_fever_prompt(input_claim: str, agent_scratchpad: str) -> str:
    # Construct the fever prompt for LLM
    if agent_scratchpad == '':
        constructed_prompt = f"{fever_prompt_prefix}\n\nClaim: {input_claim}\n{llm_prefix}"
    else:
        constructed_prompt = f"{fever_prompt_prefix}\n\nClaim: {input_claim}\n{llm_prefix}{agent_scratchpad}"
    return constructed_prompt

def write_question(idx, question, filename):
    # Write question log to file
    with open(filename, 'a') as f:
        f.write(f"Question {idx}: {question}\n")


def write_claim(idx, question, filename):
    # Write claim log to file
    with open(filename, 'a') as f:
        f.write(f"Claim {idx}: {question}\n")

def write_agent_action(action, filename, key, start_time, end_time):
    # Write agent action log to file
    with open(filename, 'a') as f:
        f.write(f'{llm_prefix}{action.log}\n')
        if (action.tool == 'Finish'):
            eval = is_correct(action.tool_input, key)
            f.write(f'Key: {key}\n')
            f.write(f'Answer: {action.tool_input}\n')
            f.write(f'Evaluation: {eval}\n')
            f.write(f'Question Time: {end_time - start_time}\n')
            f.write('\n\n')


def write_observation(action, observation, filename):
    # Write observation log to file
    with open(filename, 'a') as f:
        if (action.tool != 'Finish'):
            f.write(f'{observation_prefix}{observation}\n')


def write_exception(llm_output, key, filename, start_time, end_time):
    # Write exception log to file
    with open(filename, 'a') as f:
        f.write(f'{"LLM last output:"}{llm_output}\n')
        f.write(f'Key: {key}\n')
        f.write(f'Evaluation: Out of mem\n')
        f.write(f'Question Time: {end_time - start_time}\n')
        f.write('\n\n')


prompt_prefix = """Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action: Search[Milhouse]
Observation: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action: Lookup[named after]
Observation: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
Thought: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action: Finish[Richard Nixon]

Question: Which magazine was started first Arthur’s Magazine or First for Women?
Thought: I need to search Arthur’s Magazine and First for Women, and find which was started first.
Action: Search[Arthur’s Magazine]
Observation: Arthur’s Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought: Arthur’s Magazine was started in 1844. I need to search First for Women next.
Action: Search[First for Women]
Observation: First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First for Women), so Arthur’s Magazine was started first.
Action: Finish[Arthur’s Magazine]"""

fever_prompt_prefix = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFO. 
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action: Search[Nikolaj Coster-Waldau]
Observation: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993,[1] and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series.. Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon.[2] He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Thought: Because he "appeared in the 2009 Fox television film Virtuality", he should have worked with the Fox Broadcasting Company.
Action: Finish[SUPPORTS]

Claim: Stranger Things is set in Bloomington, Indiana.
Thought: I should search for Stranger Things, and see if it is set in Bloomington, Indiana.
Action: Search[Stranger Things]
Observation: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters. 
Thought: The observation says that it is set in a "fictional town of Hawkins, Indiana", so it is not set in Bloomington.
Action: Finish[REFUTES]
"""

texts = [
    'In which country is Hamburg?\n',
    'How are you doing today?\n',
    'It was a dark and stormy night.',
    'The sun rose slowly over the horizon, casting a warm glow on the world below.',
    'I never believed in ghosts until the day I met one.',
    'The sound of the train whistle echoed through the valley as I stood at the station, waiting.',
    'She walked into the room and everything changed.',
    'The smell of freshly baked bread filled the air as I entered the bakery.',
    'The first time I saw her, I knew she was trouble.'
    'The world was ending, and I was the only one who knew.',
    'It was the best of times, it was the worst of times.',
    'The forest was alive with the sound of animals as I walked deeper into the woods.',
    'As I looked out over the city, I knew that anything was possible.',
    'The sound of gunfire echoed through the streets as I ran for cover.',
    'The waves crashed against the shore, a never-ending cycle of destruction and creation.',
    'I woke up to find myself in a strange place, with no memory of how I got there.',
    'The clock struck midnight, and I knew that my life would never be the same.',]
