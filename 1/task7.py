import re


def find_shortest(l):
    return len(min(re.findall('[A-Za-z]+', l), key=len, default=''))
    # return len(min(re.sub('[^A-Za-z]',' ',l).split(),key=len,default=''))
# 88 символов без комментария (что закомментировано - 95 символов)
