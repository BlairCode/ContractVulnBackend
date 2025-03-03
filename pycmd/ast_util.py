import os

def slice_sol(sc_filepath, sc_byteset):
    byteset = sorted(sc_byteset)  # Ensure the byteset is sorted
    filtered = set()
    current_index = 0

    try:
        # Read the entire file as a single string
        with open(sc_filepath, "r", encoding="utf-8") as f:
            code = f.read()
    except:
        raise Exception("Failed to get source code when detecting.")
    
    # Convert the file to bytes for direct comparison
    byteset_ptr = 0  # Pointer to the current byte position in byteset
    lines = code.splitlines()
    lines = {i : lines[i] for i in range(len(lines))}

    # Process the file line by line
    for line, c in lines.items():
        # Calculate the range of this line in the byte representation
        line_bytes = c.encode('utf-8')
        line_start = current_index
        line_end = current_index + len(line_bytes)
        current_index = line_end + 1  # Account for newline (1 byte in UTF-8)

        foundall = False
        # Check if any byte positions in `byteset` fall within this line
        while byteset_ptr < len(byteset) and line_start <= byteset[byteset_ptr] <= line_end:
            filtered.add((line, c))
            byteset_ptr += 1  # Move to the next byte position in byteset

            # Exit early if all byte positions are found
            if byteset_ptr >= len(byteset):
                foundall = True
                break
        if foundall:
            break

    sorted_filtered = sorted(filtered, key=lambda x: x[0])
    filtered_lines = [line for _, line in sorted_filtered]
    return filtered_lines

def splitTempName(_str):
    result = list()
    flag = False
    temp = str()
    for char in _str:
        if char == "_" and flag == False:
            flag = True
            result.append(temp)
            temp = ""
        else:
            temp += char
    result.append(temp)
    return result[0][1:], result[1][:-1]  #

# Maybe handle nested arrays as well?
def findASTNode(ast_json, key, val):
    queue = [ast_json]
    result = list()
    while len(queue) > 0:
        data = queue.pop()
        for _key in data:
            if _key == key and data[_key] == val:
                result.append(data)
            elif type(data[_key]) == dict:
                queue.append(data[_key])
            elif type(data[_key]) == list:
                for item in data[_key]:
                    if type(item) == dict:
                        queue.append(item)
    return result

# In .dot files, contract appears like so:
# ```dot
# subgraph cluster_66_FibonacciBalance {
# ```
CLUSTER_FLAG = "cluster_"
def toContractFuncCall(chains, dot_files, root_dir):
    '''
    :return: a list of lists of 'contract.function', e.g. [['c1.f1', 'c1.f2', 'c3.f3'], ['c2.f1', 'c2.f2']]
    '''
    # result is a list of lists of 'contract.function'
    # e.g. result = [['c1.f1', 'c1.f2', 'c3.f3'], ['c2.f1', 'c2.f2']]
    result = list()
    # print(fileName)
    for dot_fileName in dot_files:
        dotFile = os.path.join(root_dir, dot_fileName)
        f = open(dotFile, 'r')
        num_namepair = dict()
        for line in f.readlines():
            if line.find(CLUSTER_FLAG) == -1:
                # no contract on this line
                continue
            try:
                # TODO
                subgraph_name = line.split(" ")[1]
                splits = subgraph_name.split("_")
                num, contractName = splits[1], splits[2]
                num_namepair[num] = contractName
            except:
                continue
        # num <-> contractName
        for chain in chains:
            curlist = list()
            for func in chain:
                try:
                    num, funcName = func.split("_")[0], func.split("_")[1]
                    if num not in num_namepair:
                        continue
                    curlist.append(num_namepair[num] + "." + funcName)
                except:
                    continue
            # curlist is a list of "contract.function"
            result.append(curlist)
    return result

def srcToPos(src):
    '''
    :param src: a string of the form "start:end:fileID"
    :return: a tuple of (start, end) where they're both byte count,
    indicating the start and end position of part of the source code
    '''
    temp = src.split(":")
    return int(temp[0]), int(temp[0]) + int(temp[1])

def getMaliciousChains(ast_json, chains, dot_files, root_dir, cond):
    '''
    :param ast_json: the json representation of the AST
    :param chains: a list of lists of 'contract.function', e.g. [['c1.f1', 'c1.f2', 'c3.f3'], ['c2.f1', 'c2.f2']]
    :param dot_files: a list of .dot files
    :param root_dir: the root directory
    :param cond: the condition that the chain must satisfy
    :return: a list of function call chains that have a call.value
    '''
    pathList = []
    # a list of chains of "contract.function"
    callPaths = toContractFuncCall(chains, dot_files, root_dir)
    # print(callPaths)
    for cp in callPaths:
        # cp_item takes the form of "contract.function"
        for cp_item in cp:
            contractName = cp_item.split('.')[0]
            funcName = cp_item.split('.')[1]
            # if the call.value is within the function
            if cond(ast_json, contractName, funcName):
                pathList.append(cp)
                break
            # if haveCallVal(ast_json, contractName, funcName):
            #     pathList.append(cp)
            #     break
            else:
                pass
    # remove duplicates
    # path list is all chain that have a function that contains call.value
    l = list(set([tuple(t) for t in pathList]))
    result = [list(v) for v in l]
    return result

