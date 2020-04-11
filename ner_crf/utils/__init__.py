# pylint: disable=bad-continuation

from ner_crf.utils.utils_lic import (
    read_by_lines,
    write_by_lines,
    get_label_map_by_file,
)
# TODO: remove common.py after extract useful funcs
from ner_crf.utils.common import (seed_everything, json_to_text)
