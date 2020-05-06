from transformers import BertTokenizer


class Tokenizer(BertTokenizer):
    @staticmethod
    def _is_special(char: str):
        return bool(char) \
                and char.startswith('[') \
                and char.endswith(']')
