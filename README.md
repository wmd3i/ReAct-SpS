This repository provides an implementation for the paper [Can Speculative Sampling Accelerate ReAct Without Compromising Reasoning Quality?](https://openreview.net/forum?id=42b9hJrIpX), for the ReAct paradigm using Llama2-70B and Speculative Sampling (SpS) techniques for answering questions from datasets like HotPotQA and FEVER. The code primarily refers to [dust-tt/llama-ssp](https://github.com/dust-tt/llama-ssp).

# Installation Instructions
## Step1: Install dependencies
To install the necessary dependencies, run the following command:

```
pip install -r requirements.txt
```

## Step 2: Enter Hugging Face Token
Authenticate with Hugging Face using the CLI:
```
huggingface-cli login
```
# Usage
## Using Llama2-70b
To use Llama2-70b model for answering sample HotPotQA questions through the ReAct paradigm, execute the following command (requires over 35GB of GPU memory for model loading and extra memory for processing):

For HotPotQA:
```
python3 react-sps.py --dataset hotpot --mode llama2
```
For FEVER:
```
python3 react-sps.py --dataset fever --mode llama2
```
This will generate a `<dataset>-llama2-<yymmddHHMM>.txt` file to record the ReAct trajectories for each processed question.
## Using Speculative Sampling (SpS)
To Speculative Sampling (Llama2-7b as approximate model and Llama-2 70b as target model) for answering sample HotPotQA questions through the ReAct paradigm, execute the following command (requires over 38.5GB of GPU memory for model loading and extra memory for processing):

For HotPotQA:
```
python3 react-sps.py --dataset hotpot --mode sps
```
For FEVER:
```
python3 react-sps.py --dataset fever --mode sps
```
This will generate a `<dataset>-sps-<yymmddHHMM>.txt` file to record the ReAct trajectories for each processed question.
## Comparing Llama2-70b and SpS Performance
To compare the performance of Llama2-70b and SpS in basic text generation tasks, use this command:
```
python3 react-sps.py --compare
```
You should see output like this:
```
Results for  Llama-2-70b
Throughput: 1.28 tokens/sec
Latency: 0.780 sec/token
Results for  SpS
Throughput: 2.80 tokens/sec
Latency: 0.356 sec/token
```
