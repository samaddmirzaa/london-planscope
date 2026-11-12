import time

from planscope.llm_classifier import CATEGORIES, NovaClassifier, few_shot


class FakeBedrock:
    '''Stands in for the Bedrock client: answers with a tool call and counts the calls.'''

    def __init__(self, category='trees'):
        self.category, self.requests = category, []

    def converse(self, **request):
        self.requests.append(request)
        return {
            'output': {'message': {'content': [{'toolUse': {'toolUseId': '1', 'name': 'record_category',
                                                            'input': {'category': self.category}}}]}},
            'usage': {'inputTokens': 300, 'outputTokens': 20, 'totalTokens': 320},
            'metrics': {'latencyMs': 250},
        }


def test_answer_is_parsed_and_cached_across_instances(tmp_path):
    client = FakeBedrock()
    model = NovaClassifier('nova', client=client, cache_path=tmp_path / 'cache.jsonl')
    first = model.classify('fell one oak tree')
    assert first == {'label': 'trees', 'input_tokens': 300, 'output_tokens': 20, 'latency_ms': 250}
    assert model.classify('fell one oak tree') == first
    reloaded = NovaClassifier('nova', client=client, cache_path=tmp_path / 'cache.jsonl')
    assert reloaded.classify('fell one oak tree') == first
    assert len(client.requests) == 1


def test_request_forces_the_tool_and_greedy_decoding(tmp_path):
    client = FakeBedrock()
    NovaClassifier('nova', client=client, cache_path=tmp_path / 'cache.jsonl').classify('rear extension')
    request = client.requests[0]
    assert request['toolConfig']['toolChoice'] == {'tool': {'name': 'record_category'}}
    schema = request['toolConfig']['tools'][0]['toolSpec']['inputSchema']['json']
    assert schema['properties']['category']['enum'] == list(CATEGORIES)
    assert request['inferenceConfig']['temperature'] == 0


def test_unknown_label_is_invalid_and_a_new_prompt_is_a_new_call(tmp_path):
    client = FakeBedrock(category='castle')
    assert NovaClassifier('nova', client=client, cache_path=tmp_path / 'c.jsonl').classify('x y z')['label'] is None
    examples = [('fell one oak tree', 'trees')]
    NovaClassifier('nova', examples=examples, client=client, cache_path=tmp_path / 'c.jsonl').classify('x y z')
    assert len(client.requests) == 2
    assert few_shot(examples) == '\n\nExamples:\nfell one oak tree -> trees'


def test_calls_are_paced_to_the_quota_but_cached_answers_are_not(tmp_path):
    model = NovaClassifier('nova', requests_per_minute=600, client=FakeBedrock(), cache_path=tmp_path / 'c.jsonl')
    start = time.monotonic()
    model.classify_all(['a b', 'c d', 'e f', 'g h'])
    assert time.monotonic() - start >= 0.25  # 4 calls 0.1 s apart (0.3 s), less clock tolerance on Windows
    start = time.monotonic()
    model.classify_all(['a b', 'c d', 'e f', 'g h'])
    assert time.monotonic() - start < 0.1