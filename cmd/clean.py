import re
import sys
import os

def clean(file):
    '''
    Clean a file from multibyte characters, 
    and remove all comments and empty lines.
    '''
    code = open(file, 'r').read()
    code = clean_multibyte(code)
    code = clean_comments(code)
    code = clean_empty_lines(code)
    code = replace_tabs(code)
    with open(file, 'w') as f:
        f.write(code)

def clean_multibyte(str):
    '''
    Clean a file from multibyte characters.
    '''
    result = ''
    for c in str:
        if len(c) != len(c.encode()):
            result += '#'
            continue
        result += c
    return result

def clean_comments(code):
    '''
    Remove all comments from a .sol file.
    '''
    singleLinePattern = re.compile(r"//.*$", re.MULTILINE)
    multipleLinePattern = re.compile(r"/\*.*\*/", re.DOTALL)
    code_no_multiline = multipleLinePattern.sub('', code)
    code_no_singleline = singleLinePattern.sub('', code_no_multiline)
    return code_no_singleline

    # indexList = list()
    # for item in singleLinePattern.finditer(code):
    #     indexList.append(item.span())
    # for item in multipleLinePattern.finditer(code, re.S):
    #     indexList.append(item.span())
    # startIndedx = 0
    # newCode = str()
    # for item in indexList:
    #     newCode += code[startIndedx: item[0]]
    #     startIndedx = item[1] + 1
    # newCode += code[startIndedx:]
    # return newCode

def clean_empty_lines(code):
    '''
    Remove all empty lines from a .sol file.
    '''
    return '\n'.join([line for line in code.split('\n') if line.strip() != ''])

def replace_tabs(code):
    '''
    Replace all tabs with spaces in a .sol file.
    '''
    return code.replace('\t', ' '*4)

if __name__ == '__main__':
    # read argument
    if len(sys.argv) < 2:
        print("Usage: python clean.py <file>/<dir>")
        sys.exit(1)
    input = sys.argv[1]
    if os.path.isdir(input):
        for root, _, files in os.walk(input):
            for file in files:
                if file.endswith('.sol'):
                    clean(os.path.join(root, file))
    else:
        file = sys.argv[1]
        clean(file)
