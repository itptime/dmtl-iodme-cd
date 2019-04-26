import json


def json_read(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as data_file:
        return json.load(data_file)


def json_write(filename, data, encoding='utf-8', default_type=str):
    with open(filename, 'w', encoding=encoding) as data_file:
        data_file.write(json.dumps(
            data,
            default=default_type,
            ensure_ascii=False))
