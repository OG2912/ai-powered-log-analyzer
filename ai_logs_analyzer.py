#!/usr/bin/env python3
import argparse
import pandas as pd
import re
from collections import Counter
import nltk
import os
import openai

nltk.download('punkt')

def read_logs(file_path):
    with open(file_path, 'r', errors='ignore') as f:
        return f.readlines()

def tokenize_logs(logs):
    tokens = [nltk.word_tokenize(line) for line in logs]
    flat_tokens = [item for sublist in tokens for item in sublist]
    return Counter(flat_tokens)

def summarize_logs(logs, api_key=None):
    if not api_key:
        print("No API key provided. Skipping summarization.")
        return
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a helpful assistant that summarizes server logs."},
            {"role": "user", "content": f"Summarize the following logs:\n\n{''.join(logs[:100])}"}
        ]
    )
    print("\n[Summary]\n")
    print(response.choices[0].message['content'])

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Log Analyzer")
    parser.add_argument("--file", required=True, help="Path to log file")
    parser.add_argument("--summarize", action="store_true", help="Summarize logs using OpenAI")
    parser.add_argument("--api-key", help="OpenAI API Key")
    args = parser.parse_args()

    logs = read_logs(args.file)
    print(f"[+] Loaded {len(logs)} log entries")
    
    tokens = tokenize_logs(logs)
    print("\n[Top 20 Keywords]\n")
    for word, count in tokens.most_common(20):
        print(f"{word}: {count}")
    
    if args.summarize:
        summarize_logs(logs, args.api_key)

if __name__ == "__main__":
    main()
