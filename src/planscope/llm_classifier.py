'''
Classify application descriptions with an Amazon Nova model on Bedrock, for the benchmark
against the fine-tuned classifier.
'''

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / 'data' / 'llm_cache.jsonl'
REGION = 'eu-west-2'  # Nova Micro, Lite and Pro run in London itself, so the data stays in the UK

# What each category means: the same rules clean.py uses to group the official application types
CATEGORIES = {
    'householder': 'works to a single house or its garden (extensions, loft conversions, outbuildings), '
                   'including when combined with listed building consent',
    'full_planning': 'full or outline planning permission for anything that is not a householder project: '
                     'new buildings, flats, changes of use, shopfronts, commercial works',
    'listed_building': 'listed building consent on its own, often for internal or repair works',
    'conditions': 'approval of details required by a condition, or removing or varying a condition of an earlier permission',
    'amendment': 'a non-material or minor material amendment to an earlier permission, or approval of reserved matters',
    'lawful_development': 'a certificate of lawfulness for a proposed or existing use or development',
    'prior_approval': 'prior approval under permitted development rights, such as larger home extensions, '
                      'offices or shops to homes, telecoms masts, solar panels or demolition',
    'trees': 'works to trees protected by a tree preservation order or in a conservation area',
    'advert': 'consent to display an advertisement (signs, fascias, hoardings)',
}

SYSTEM = ('You classify London planning application descriptions by the type of application '
          '(the legal route) they were submitted under. The categories are:\n'
          + '\n'.join(f'- {label}: {meaning}' for label, meaning in CATEGORIES.items())
          + '\nRecord the single most likely category with the record_category tool.')

# Structured output: the only way to answer is this tool, and its argument must be a category
TOOLS = {
    'tools': [{'toolSpec': {
        'name': 'record_category',
        'description': 'Record the application type of the description.',
        'inputSchema': {'json': {
            'type': 'object',
            'properties': {'category': {'type': 'string', 'enum': list(CATEGORIES)}},
            'required': ['category'],
        }},
    }}],
    'toolChoice': {'tool': {'name': 'record_category'}},
}


def few_shot(examples):
    '''Worked examples appended to the instructions, from (description, category) pairs.'''
    if not examples:
        return ''
    return '\n\nExamples:\n' + '\n'.join(f'{text} -> {label}' for text, label in examples)


class NovaClassifier:
    '''One Nova model with one prompt; classify() returns the label, tokens used and latency.'''

    def __init__(self, model_id, examples=None, requests_per_minute=None, client=None, cache_path=CACHE_PATH):
        if client is None:
            # Imported here, so the API and its tests never need boto3
            import boto3
            from botocore.config import Config
            client = boto3.client('bedrock-runtime', region_name=REGION,
                                  config=Config(retries={'max_attempts': 10, 'mode': 'adaptive'}))
        self.client, self.model_id = client, model_id
        self.system = SYSTEM + few_shot(examples)
        self.cache_path = Path(cache_path)
        self.lock = threading.Lock()
        # Pace calls to stay under the account's requests-per-minute quota (None = no pacing)
        self.interval = 60 / requests_per_minute if requests_per_minute else 0
        self.next_slot = 0.0
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                record = json.loads(line)
                self.cache[record.pop('key')] = record

    def key(self, text):
        '''Same model, prompt and text -> same key; changing any of them means a fresh call.'''
        return hashlib.sha256(json.dumps([self.model_id, self.system, text]).encode()).hexdigest()

    def classify(self, text):
        key = self.key(text)
        if key in self.cache:
            return self.cache[key]
        self.wait_for_slot()
        response = self.client.converse(
            modelId=self.model_id,
            system=[{'text': self.system}],
            messages=[{'role': 'user', 'content': [{'text': text}]}],
            toolConfig=TOOLS,
            inferenceConfig={'maxTokens': 50, 'temperature': 0},
        )
        calls = [block['toolUse']['input'] for block in response['output']['message']['content'] if 'toolUse' in block]
        label = calls[0].get('category') if calls else None
        result = {
            'label': label if label in CATEGORIES else None,  # anything else counts as an invalid answer
            'input_tokens': response['usage']['inputTokens'],
            'output_tokens': response['usage']['outputTokens'],
            'latency_ms': response['metrics']['latencyMs'],
        }
        with self.lock:  # several threads may finish at once; write one line at a time
            self.cache[key] = result
            with self.cache_path.open('a') as f:
                f.write(json.dumps({'key': key, **result}) + '\n')
        return result

    def wait_for_slot(self):
        '''Book the next free start time, then sleep until it, so calls start at most once per interval.'''
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_slot)
            self.next_slot = start + self.interval
        time.sleep(start - now)

    def classify_all(self, texts, workers=4):
        '''Classify many texts, a few calls at a time, in the original order.'''
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(self.classify, texts))