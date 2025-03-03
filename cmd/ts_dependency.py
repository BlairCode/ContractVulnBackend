# internal
from .ast_util import findASTNode, srcToPos, getMaliciousChains

if __name__ == '__main__':
    pass

def getBlockTSLocs(ast_json):
    '''
    :param ast_json: json of ast, loaded as a python dict
    :return: a list of locations of block.timestamp (start byte, end byte)
    '''
    memberList = findASTNode(ast_json,'name', 'MemberAccess')
    location = []
    for item in memberList:
        if item["attributes"]["member_name"] == "timestamp" and item["children"][0]["name"] == "Identifier":
            if item["children"][0]["attributes"]["value"] == "block":
                memStartPos,memEndPos = srcToPos(item['src'])
                location.append((memStartPos,memEndPos))
    return location

def haveTSD(ast_json, contractName, functionName):
    '''
    :return: True if the function contains timestamp dependence, False otherwise
    '''
    contractAST = findASTNode(ast_json, 'name', 'ContractDefinition')
    ts_locations = getBlockTSLocs(ast_json)
    for contractItem in contractAST:
        if contractItem['attributes']['name'] != contractName:
            continue
        functionAST = findASTNode(contractItem, 'name', 'FunctionDefinition')
        for functionItem in functionAST:
            if functionItem['attributes']['name'] != functionName:
                continue
            funcStartPos, funcEndPos = srcToPos(functionItem['src'])
            for start, end in ts_locations:
                if start >= funcStartPos and end <= funcEndPos:
                    return True

def isBlockTimestamp(json_node):
    '''
    :param json_node: a node in the AST
    :return: True if the node is a block.timestamp, False otherwise
    '''
    if json_node['name'] != 'MemberAccess':
        return False
    if json_node['attributes']['member_name'] != 'timestamp':
        return False
    if json_node['children'][0]['name'] != 'Identifier':
        return False
    if json_node['children'][0]['attributes']['value'] != 'block':
        return False

    return True

def isBlockNumber(json_node):
    '''
    :param json_node: a node in the AST
    :return: True if the node is a block.timestamp, False otherwise
    '''
    if json_node['name'] != 'MemberAccess':
        return False
    if json_node['attributes']['member_name'] != 'number':
        return False
    if json_node['children'][0]['name'] != 'Identifier':
        return False
    if json_node['children'][0]['attributes']['value'] != 'block':
        return False

    return True

def getDecStatements(ast_json, contractName, functionName):
    declar_nodes = findASTNode(ast_json, 'name', 'VariableDeclarationStatement')
    declar = []
    for dec in declar_nodes:
        child = dec['children'][0]
        if child['name'] != 'VariableDeclaration':
            continue
        child2 = dec['children'][1]
        if isBlockTimestamp(child2) or isBlockNumber(child2):
            declar.append(child)
    return declar

def getIdentifiers(ast_json, contractName, functionName):
    '''
    :param ast_json: json of ast, loaded as a python dict
    :param contractName: name of the contract
    :param functionName: name of the function
    :return: a list of identifier nodes/declaration nodes in the function
    '''
    contractAST = findASTNode(ast_json, 'name', 'ContractDefinition')
    ids = set() 
    for contractItem in contractAST:
        if contractItem['attributes']['name'] != contractName:
            continue
        functionAST = findASTNode(contractItem, 'name', 'FunctionDefinition')
        for functionItem in functionAST:
            if functionItem['attributes']['name'] != functionName:
                continue

            assignments = findASTNode(ast_json, 'name', 'Assignment')
            for assignmentItem in assignments:
                # a bunch of 'assert's
                if assignmentItem['attributes']['operator'] != '=':
                    continue
                if len(assignmentItem['children']) != 2:
                    continue
                child = assignmentItem['children'][1]
                if isBlockTimestamp(child):
                    idList = findASTNode(assignmentItem, 'name', 'Identifier')
                    for idItem in idList:
                        ids.add(idItem)

            decs = getDecStatements(ast_json, contractName, functionName)
            for dec in decs:
                ids.add(dec)
    return list(ids)

def findIdentifierByteLocs(ast_json, identifier):
    '''
    :param ast_json: json of ast, loaded as a python dict
    :param identifier: an identifier
    '''
    location = []
    idList = findASTNode(ast_json, 'name', 'Identifier')
    for item in idList:
        if item['attributes']['value'] == identifier:
            idStartPos, idEndPos = srcToPos(item['src'])
            location.append(idStartPos)
            location.append(idEndPos)
    idList = findASTNode(ast_json, 'name', 'VariableDeclaration')
    for item in idList:
        if item['attributes']['name'] == identifier:
            idStartPos, idEndPos = srcToPos(item['src'])
            location.append(idStartPos)
            location.append(idEndPos)
    return list(set(location))

def getTSDependencyByteLocs(ast_json, chains, dot_files, root_dir):
    '''
    Used to slice time stamp dependency attack code.
    :param ast_json: json of ast, loaded as a python dict
    :param chains: a list of function call chains(e.g. [f1 -> f2 -> f3, g1 -> g2, h1, ...])
    :param dot_files: names of .dot files
    :param root_dir: root directory of the file being processed
    :return: a list of smart contracts that have a function that contains call.value
    '''
    # smart contract list
    # sc_list = []
    # all chains that have a function that contains call.value
    chains = getMaliciousChains(ast_json, chains, dot_files, root_dir, haveTSD)
    locs = set()
    for chain in chains:
        for onepathItem in chain:
            contractName = onepathItem.split('.')[0]
            funcName = onepathItem.split('.')[1]

            funcstart, funcend = -1, -1
            funcnode = None
            funcs = findASTNode(ast_json, 'name', 'FunctionDefinition')
            for funcItem in funcs:
                if funcItem['attributes']['name'] != funcName:
                    continue
                funcnode = funcItem
            if funcnode is None:
                continue
            funcstart, funcend = srcToPos(funcnode['src'])
            locs.add(funcstart)
            locs.add(funcend)

            ids = getIdentifiers(ast_json, contractName, funcName)
            ids_filtered = []
            for id in ids:
                if id['attributes']['value'] == 'block':
                    continue
                ids_filtered.append(id['attributes']['value'])
            if_nodes = findASTNode(funcnode, 'name', 'IfStatement')
            for id in ids_filtered:
                for l in findIdentifierByteLocs(ast_json, id['attributes']['value']):
                    locs.add(l)
                idstart, idend = 1
                for if_node in if_nodes:
                    if_start, if_end = srcToPos(if_node['src'])
                    locs.add(if_start)
                    locs.add(if_end)

    return list(locs)
