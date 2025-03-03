def get_graph(edges):
    graph = {} # adjacency list
    s = set() # start nodes
    ind = {} # in-degree
    for edge in edges:
        if edge[0] not in graph:
            graph[edge[0]] = set()
        graph[edge[0]].add(edge[1])
        ind[edge[0]] = ind.get(edge[0], 0)
        ind[edge[1]] = ind.get(edge[1], 0) + 1
    for v, d in ind.items():
        if d == 0:
            s.add(v)
    # print(ind)
    # print(graph)
    # print(s)
    return graph, s

# def dfs(graph, node, chain, ans, visited):
#     visited.add(node)
#     chain.append(node)

#     leaf = True
#     if node in graph:
#         for neighbor in graph[node]:
#             # if neighbor in visited:
#             #     continue
#             cp_chain = chain[:]
#             cp_vis = visited.copy()
#             leaf = False
#             dfs(graph, neighbor, cp_chain, ans, cp_vis)

#     if leaf:
#         ans.append(chain)
#         return

# Changed
def dfs(graph, node, chain, ans, visited):
    visited.add(node)
    chain.append(node)

    leaf = True
    if node in graph:
        for neighbor in graph[node]:
            if neighbor not in visited:  # 避免环路
                leaf = False
                dfs(graph, neighbor, chain, ans, visited)  # 直接使用 chain 和 visited

    if leaf:
        ans.append(chain[:]) 

    chain.pop() 
    visited.remove(node) 

def get_chains(edges):
    graph, s = get_graph(edges)
    ans = []
    visited = set()
    for node in s:
        chain = []
        dfs(graph, node, chain, ans, visited)
    return ans

def test():
    edges = set([
        ('a', 'b'),
        ('a', 'c'),
        ('a', 'd'),
        ('b', 'e'),
        ('d', 'b'),
        ('c', 'f')
    ])
    # print(edges)
    print(get_chains(edges))

if __name__ == '__main__':
    test()
    pass
