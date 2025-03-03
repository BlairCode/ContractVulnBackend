# internal
from .ast_util import findASTNode, srcToPos, getMaliciousChains

# AST JSON example:
# {
#     "attributes" : 
#     {
#         "argumentTypes" : null,
#         "isConstant" : false,
#         "isLValue" : false,
#         "isPure" : false,
#         "lValueRequested" : false,
#         "member_name" : "sender",
#         "referencedDeclaration" : null,
#         "type" : "address"
#     },
#     "children" : 
#     [
#         {
#             "attributes" : 
#             {
#                 "argumentTypes" : null,
#                 "overloadedDeclarations" : 
#                 [
#                     null
#                 ],
#                 "referencedDeclaration" : 78,
#                 "type" : "msg",
#                 "value" : "msg"
#             },
#             "id" : 13,
#             "name" : "Identifier",
#             "src" : "172:3:0"
#         }
#     ],
#     "id" : 14,
#     "name" : "MemberAccess",
#     "src" : "172:10:0"
# }

# find all call.value and return a list of their location (start byte, end byte)
def getCallValueLocs(ast_json):
    '''
    :param ast_json: json of ast, loaded as a python dict
    :return: a list of locations of call.value (start byte, end byte)
    '''
    memberList = findASTNode(ast_json,'name', 'MemberAccess')
    location = []
    for item in memberList:
        if item["attributes"]["member_name"] == "value" and item["children"][0]["name"] == "MemberAccess":
            if item["children"][0]["attributes"]["member_name"] == "call":
                memStartPos,memEndPos = srcToPos(item['src'])
                location.append((memStartPos,memEndPos))
    return location

FLAG_FUNC = -2
def haveCallVal(ast_json, contractName, functionName):
    '''
    :return: True if the function contains a call.value, False otherwise
    '''
    # find all dict containing contract definition
    contractAST = findASTNode(ast_json, 'name', 'ContractDefinition')
    # find all call.value and return a list of their location (start byte, end byte)
    callval_locations = getCallValueLocs(ast_json)
    for contractItem in contractAST:
        # the contractItem that represents the contract we are looking for
        if contractItem['attributes']['name'] != contractName:
            # not the contract we are looking for
            continue;

        # find all its function definitions
        functionAST = findASTNode(contractItem, 'name', 'FunctionDefinition')
        for functionItem in functionAST:
            if functionItem['attributes']['name'] != functionName:
                # not the function we are looking for
                continue;

            # contract item is the contract we want,
            # function item is the function we want
            # now we get the span of that function
            funcStartPos, funcEndPos = srcToPos(functionItem['src'])
            for start, end in callval_locations:
                # call.value is within the function
                if start >= funcStartPos and end <= funcEndPos:
                    return True
    return False


def getAddressVariable(ast_json, contractName, functionName):
    identifier_dict = {}
    elementList = []
    contract_ast = findASTNode(ast_json, "name", "ContractDefinition")
    for contract_item in contract_ast:
        if contract_item["attributes"]["name"] != contractName:
            continue;
        func_ast = findASTNode(contract_item, "name", "FunctionDefinition")
        for func_item in func_ast:
            if func_item["attributes"]["name"] != functionName:
                continue;

            # now contract_item is the contract we want,
            # func_item is the function we want
            # member_ast is the list of all member accesses(a.x) within the function
            member_ast = findASTNode(func_item, "name", "MemberAccess")
            for member_item in member_ast:
                if member_item["attributes"]["member_name"] != "call":
                    continue;

                # member_item is a.call
                # all identifiers in this a.call
                identifier_ast = findASTNode(member_item, "name", "Identifier")
                eleExpression_ast = findASTNode(member_item, 'name', 'ElementaryTypeNameExpression')
                mem_ast = findASTNode(member_item, 'name', 'MemberAccess')
                for eleExpression_item in eleExpression_ast:
                    if eleExpression_item['attributes']['value'] == 'address':
                        elementList.append(eleExpression_item['attributes']['argumentTypes'][0]['typeString'])
                for mem_item in mem_ast:
                    if mem_item['attributes']['type'] != 'address':
                        continue
                    # there's b.x present where x is an address
                    if mem_item['children'][0]['attributes']['referencedDeclaration'] and \
                            mem_item['attributes']['referencedDeclaration']:
                        # both b and x(address) are declared
                        # declaration of b
                        mem_declaration_ = mem_item['children'][0]['attributes']['referencedDeclaration']
                        # b's id -> x
                        # indicates that b.x is an address
                        identifier_dict[mem_declaration_] = mem_item['attributes']['member_name']
                for identifier_item in identifier_ast:
                    # check whether is referencedDeclaration
                    # if identifier_item["attributes"]["referencedDeclaration"]:
                    if "referencedDeclaration" in identifier_item["attributes"]:
                        if identifier_item["attributes"]["type"] == "address" or \
                            identifier_item['attributes']['type'] == 'address payable' or \
                            identifier_item["attributes"]["type"] == "contract OwnedUpgradeabilityProxy" or \
                            identifier_item["attributes"]["type"] == "address[] memory" :

                            identifier_name = identifier_item["attributes"]["value"]
                            identifier_id = identifier_item["attributes"]["referencedDeclaration"]
                            identifier_dict[identifier_id] = identifier_name

                        elif identifier_item['attributes']['type'] == 'msg':
                            identifier_name = identifier_item['attributes']['value']
                            identifier_id = identifier_item['attributes']['referencedDeclaration']
                            identifier_dict[identifier_id] = identifier_name

                        elif identifier_item['attributes']['type'] in elementList:
                            identifier_name = identifier_item['attributes']['value']
                            identifier_id = identifier_item['attributes']['referencedDeclaration']
                            identifier_dict[identifier_id] = identifier_name
                        else:
                            continue
                    else:
                        identifier_name = identifier_item["attributes"]["value"]
                        identifier_id = member_item["id"]
                        identifier_dict[identifier_id] = identifier_name
    return identifier_dict

def getAddressRelatedSC(ast_json, contractName, functionName, var_dict):
    pos_list = []
    address_key_ = [key for key in var_dict.keys()][0]
    # address_var_ = [var for var in var_dict.values()][0]
    addressID_ast = findASTNode(ast_json, 'id', address_key_)
    addressId_pos = []
    for addressID_item in addressID_ast:
        addressID_startPos,addressID_endPos = srcToPos(addressID_item['src'])
        addressId_pos.append([addressID_startPos,addressID_endPos])
    contract_ast = findASTNode(ast_json, 'name', 'ContractDefinition')
    for contractItem in contract_ast:
        if contractItem['attributes']['name'] != contractName:
            continue;
        contractStartPos,contractEndPos = srcToPos(contractItem['src'])
        func_ast = findASTNode(contractItem, "name", "FunctionDefinition")
        for funcItem in func_ast:
            if funcItem['attributes']['name'] != functionName:
                continue;


            # now contractItem is the contract we want,
            # funcItem is the function we want
            funcStartPos,funcEndPos = srcToPos(funcItem['src'])
            pos_list.append([funcStartPos, funcEndPos])
            # 2.msg.sender
            identifier_ast = findASTNode(funcItem, 'name', 'Identifier')

            if_nodes = findASTNode(funcItem, 'name', 'IfStatement')
            # print("found if statement:")
            if_pos = []
            for if_item in if_nodes:
                if_startPos, if_endPos = srcToPos(if_item['src'])
                # first block is always present
                _, be = srcToPos(if_item['children'][1]['src'])

                # else block is sometimes present.
                bs = if_startPos
                if len(if_item['children']) > 2:
                    bs, _ = srcToPos(if_item['children'][2]['src'])
                if_pos.append((if_startPos, be, bs, if_endPos))

            for identifierItem in identifier_ast:
                if 'referencedDeclaration' not in identifierItem['attributes']:
                    continue
                if identifierItem['attributes']['referencedDeclaration'] != address_key_:
                    continue;
                iden_startPos, iden_endPos = srcToPos(identifierItem['src'])
                pos_list.append([iden_startPos])

                for start, e, s, end in if_pos:
                    if iden_startPos >= start and iden_endPos <= end:
                        pos_list.append([start, e, s, end])

            for item in addressId_pos:
                addressID_startPos = item[0]
                addressID_endPos = item[1]
                if addressID_startPos >= funcStartPos  and addressID_endPos <= funcEndPos:
                    identifier_ast = findASTNode(funcItem, "name", "Identifier")
                    for identifier_item in identifier_ast:
                        if not "referencedDeclaration" in identifier_item["attributes"]:
                            continue
                        if identifier_item["attributes"]["referencedDeclaration"] == address_key_:
                            if identifier_item["attributes"]["type"] == "address" or \
                                    identifier_item["attributes"][
                                        "type"] == "contract OwnedUpgradeabilityProxy" or \
                                    identifier_item["attributes"]["type"] == "address[] memory":
                                identifier_startPos, _ = srcToPos(identifier_item["src"])
                                pos_list.append([identifier_startPos])
                            elif identifier_item['attributes']['type'] == 'msg':
                                identifier_startPos,_ = srcToPos(identifier_item['src'])
                                pos_list.append([identifier_startPos])
                            else:
                                continue
                        else:
                            continue
                elif addressID_startPos >= contractStartPos and addressID_endPos <= contractEndPos:
                    pos_list.append([funcStartPos, funcEndPos])
                    identifier_ast_ = findASTNode(funcItem, "name", "Identifier")
                    for identifier_item_ in identifier_ast_:
                        if identifier_item_["attributes"]["referencedDeclaration"] == address_key_:
                            if identifier_item_["attributes"]["type"] == "address" or \
                                    identifier_item_["attributes"][
                                        "type"] == "contract OwnedUpgradeabilityProxy" or \
                                    identifier_item_["attributes"]["type"] == "address[] memory":
                                identifier_startPos_, _ = srcToPos(identifier_item_["src"])
                                pos_list.append([identifier_startPos_])
                            elif identifier_item_['attributes']['type'] == 'msg':
                                identifier_startPos_, _ = srcToPos(identifier_item_['src'])
                                pos_list.append([identifier_startPos_])
                            else:
                                continue
                        else:
                            continue
                else:
                    continue

    return pos_list

if __name__ == '__main__':
    pass

def getCallValueRelatedByteLocs(ast_json, chains, dot_files, root_dir):
    '''
    Used to slice reentrency attack code.
    :param ast_json: json of ast, loaded as a python dict
    :param chains: a list of function call chains(e.g. [f1 -> f2 -> f3, g1 -> g2, h1, ...])
    :param dot_files: names of .dot files
    :param root_dir: root directory of the file being processed
    :return: a list of byte locations of reentrency attack code
    '''
    # smart contract list
    sc_list = []
    # all chains that have a function that contains call.value
    chains = getMaliciousChains(ast_json, chains, dot_files, root_dir, haveCallVal)
    for chain in chains:
        for onepathItem in chain:
            contractName = onepathItem.split('.')[0]
            funcName = onepathItem.split('.')[1]
            var_dict = getAddressVariable(ast_json, contractName, funcName)
            if len(var_dict) == 0:
                pass
            else:
                # print(var_dict)
                sc = getAddressRelatedSC(ast_json, contractName, funcName, var_dict)
                sc_list.append(sc)
                # print(sc)
    sc = list(set([m for i in sc_list for j in i for m in j]))
    return sc
