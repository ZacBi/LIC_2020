"""Process date file."""

import json


def read_json_line(f_obj):
    """For json file whose group is not wrapped by '[]' but one group a line."""
    for line in f_obj:
        yield json.loads(line)
