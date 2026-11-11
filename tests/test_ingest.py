from planscope.ingest import flatten


def test_flatten_stores_nested_values_as_json_text():
    record = {'id': 'a', 'centroid': {'lon': -0.1, 'lat': 51.5}, 'uprn': 123}
    assert flatten(record) == {'id': 'a', 'centroid': '{"lon": -0.1, "lat": 51.5}', 'uprn': 123}