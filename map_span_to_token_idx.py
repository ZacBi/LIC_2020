    def map_span_to_token_idx(self, paragraph, tokens, span, offset=0):
        """
        paragraph: '1988我爱北京天安门'
        tokens:    ['1988', '我', '爱', '北', '京', '天', '安', '门']
        span:      (6,8)
        return:     (3,5)
        """
        start = end = 0
        span = (span[0] - offset, span[1] - offset)
        left = right = None
        paragraph = paragraph.lower()
        for i, token in enumerate(tokens):
            token = token.lower()
            if token.startswith('##'):
                token = token[2:]
            try:
                start = paragraph.index(token, end)
                end = start + len(token)
                if start >= span[0] and end <= span[1]:
                    if left is None:
                        left = i
                    right = i
            except:
                pass
        return left, right+1