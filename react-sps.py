import sys
import utils
import joblib
import json
            
hotpot_path = './data/hotpot.jsonl'
fever_path = './data/fever.jsonl'

def main():
    args = utils.parse_arguments()
    # If the 'compare' flag is set, compare model performance
    if args.compare:
        utils.compare_model_performance()
        sys.exit() 
        
    # Initialize the model based on the mode ('llama2' or 'sps') flag
    llm = utils.initialize_llm(args)
    
    # Process HotPotQA/FEVER dataset
    if args.dataset == 'hotpot':
        hotpot = []
        with open(hotpot_path, 'r') as file:
            for line in file:
                hotpot.append(json.loads(line))
        utils.process_hotpot_qa(hotpot, llm, args)
    elif args.dataset == 'fever':
        fever = []
        with open(fever_path, 'r') as file:
            for line in file:
                fever.append(json.loads(line))
        utils.process_fever(fever, llm, args)
    else:
        print("Error: Invalid dataset specified.")
        sys.exit(1)
        
if __name__ == '__main__':
    main()