from nfa import *
from random import *
from print_color import print
from math import floor
from igraph import *
import pandas as pd
import sys


# A color dict with all basics color
color_dict = [
    'blue',
    'magenta',
    'red',
    'yellow',
    'green',
]
NFA.clear()

PARA_LIST = []
RESULT_LIST = []
GRAPH_NUMBER = 100
GRAPH_SIZE = 10
GRAPH_DENSITY = False
PRINT = False
GEN = False
RANDOM = False
COMPLEXITY_PRINT = False
BRUTEFORCE_F = False

# Test each args
"""Test the args"""
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if(arg == "--help" or arg == "-h"):
        print("Usage: calc.py [--help || -h] [--print || -P] [--gen || -G] [--random || -R] [--complexity || -C] [--brute || -B] [--graph_size || -gs =<size>] [--graph_number || -gn =<number>] [--graph_density || -gd =<density>]")
        sys.exit(1)
    elif(arg == "--print" or arg == "-P"):
        PRINT = True
    elif(arg == "--gen" or arg == "-G"):
        GEN = True
    elif(arg == "--random" or arg == "-R"):
        RANDOM = True
    elif(arg == "--complexity" or arg == "-C"):
        COMPLEXITY_PRINT = True
    elif(arg == "--brute" or arg == "-B"):
        BRUTEFORCE_F = True
    elif(arg == "--graphNum" or arg == "-gn"):
        GRAPH_NUMBER = int(sys.argv[i+1])
        i+= 1
    elif(arg == "--graphSize" or arg == "-gs"):
        GRAPH_SIZE = int(sys.argv[i+1])
        i+= 1
    elif(arg == "--graphDensity" or arg == "-gd"):
        GRAPH_DENSITY = float(sys.argv[i+1])
        i+= 1
    elif(arg == "--version" or arg == "-v"):
        print("Version: 1.0")
    else:
        print("Unknown argument")
        print(arg)
        sys.exit(1)
    i+= 1

DICT = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def tupleGen(graph, offset):
    for i, node in enumerate(graph):
        for pos in binhandler(node):
            yield(DICT[i], ' ', DICT[offset-pos-1])
    # yield((DICT(i), '', DICT(pos)) for pos in binhandler(node))

def firstBit(x):
    """Return the position of the first 1 in the binary representation of x"""
    i = 0
    while x > 0:
        if x % 2 == 1:
            return i
        x = x // 2
        i += 1
    return -1

def binhandler(x):
    """Return every position of 1 in the binary representation of x"""
    return [firstBit(x)] + binhandler(x ^ (1 << firstBit(x))) if x > 0 else []

def graphGen(graph, addname=""):
    """Visualize the graph"""
    print(graph) if PRINT else None
    A = NFA( {}, {}, 
        set(tpl for tpl in tupleGen(graph,7)), name=f"Graph{addname}" )
    A.visu(doublearrows=True)

def max_score_indexs(graph):
    """Return the index of node with the highest score"""
    SCORE = node_score(graph)
    max_score = -1
    #searching the max score in SCORE
    for i in range(0, len(SCORE)):
        if SCORE[i] > max_score:
            max_score = SCORE[i]
    #return every index of node with the highest score
    return [i for i in range(len(graph)) if SCORE[i] == max_score] if max_score > 0 else -1

def node_score(graph):
    """Return the score of every node in the graph"""
    return [len(binhandler(node)) for node in graph]

def random_graph(size):
    """return a random graph"""
    return tuple(randint(0,2**size) for i in range(size))

def random_graph_bis(size):
    """return a random graph V2"""
    tab = []
    for i in range(size):
        tab.append(randint(0,2**size) & (~(1 << size - i - 1)))
    for j,node in enumerate(tab):
        for bit in binhandler(node):
            tab[size-bit-1] = tab[size-bit-1] | (1 << size - j - 1)

    return tuple(node for node in tab)

def random_graph_tri(size):
    """return a random graph V3"""
    d = randint(4,10)/10
    if GRAPH_DENSITY:
        g = Graph.GRG(size, GRAPH_DENSITY)
    else:
        g = Graph.GRG(size, d)
    S = 0
    for tab in g.get_adjacency():
        for node in tab:
            if node != 0:
                S += 1
    print(f"{size} {d} {S}") if COMPLEXITY_PRINT else None
    PARA_LIST.append((size, d, S))
    return(tuple(tabin_toInt(node) for node in g.get_adjacency()))

def tabin_toInt(tab):
    """Convert a tuple of binary number to an integer"""
    return int(''.join(str(x) for x in tab), 2)

def int_to_tabin(x, s):
    """Convert an integer to a tuple of binary number"""
    return [int(x) for x in bin(x)[2:].zfill(s+1)]

def BRUTEFORCE(graph):
    """Bruteforce the graph to return the best vertex cover score"""
    sol_length = len(graph)
    solutions = []
    # print("The graph", graph)
    for c in combination(len(graph)):
        new_graph = graph
        for n in c:
            bitRemover = 1 << len(graph) - n - 1
            new_graph = tuple(node & (~bitRemover) if i != n else 0 for i,node in enumerate(new_graph))
        if(new_graph.count(0) == len(new_graph)):
            if(len(c) < sol_length):
                sol_length = len(c)
                solutions = []
                solutions.append(c)
            elif(len(c) == sol_length):
                solutions.append(c)
    # print("BRUTEFORCE: ",sol_length, solutions, end="")
    return sol_length, solutions

# A function wich return every possible combination of integers from 0 to x
def combination(x):
    """Return every possible tuple of integers from 0 to x"""
    for i in range(pow(2, x)):
        yield(binhandler(i)) if i else [0]

def generation(G):
    """Generating the graphs"""
    print(f"Generating {GRAPH_NUMBER} random graphs...")
    for i in range(0, GRAPH_NUMBER):
        print(f"{i} ", end="") if COMPLEXITY_PRINT else None
        new_graph = random_graph_tri(randint(3,GRAPH_SIZE))
        print("Random graph:", new_graph) if PRINT else None
        print(f"Graph {i}:", tuple(bin(node) for node in new_graph)) if PRINT else None
        G.append(new_graph)

# def complexity(i):
#     """Increase the complexity of the algorithm"""
#     COMPLEXITY += i


"""Main function"""
def resolve(G, gen = False):
    ANALYSE = [0,0,0] # [Covered, not covered, canceled]
    for j,graph in enumerate(G):
        if(GRAPHS[j].count(0) == len(GRAPHS[j])):
            print("") if PRINT else None
            print(f"Graph {j} useless: {GRAPHS[j]}", tag='useless', tag_color='yellow', color='white') if PRINT else None
            print(f'{0}') if COMPLEXITY_PRINT else None
            RESULT_LIST.append((0))
            ANALYSE[2] += 1
            continue
        SOL = "covered by "
        ADDSTR = ""
        SOL_COUNT = 0
        COMPLEXITY = 0
        step = 0
        print(f"Graph {j}") if PRINT else None
        while(max_score_indexs(graph) != -1):
            print("_____NEW STEP_____") if PRINT else None
            step += 1 
            
            # Here if you want any steps
            graphGen(graph, addname=f" - {j} - {ADDSTR}") if gen else None

            SCORE = node_score(graph)
            COMPLEXITY += len(SCORE)-step
            MAX_SCORE = max_score_indexs(graph)
            SIZE = len(graph)
            print(f"SCORE: {SCORE}") if PRINT else None
            print(f"MAX_SCORE: {max(SCORE)}, index:{MAX_SCORE}") if PRINT else None
            S = (0,0) #score, index final

            """Picking the node linked to the node with only one edge"""
            COMPLEXITY += len(SCORE)-step
            for i,score in enumerate(SCORE):
                if score == 1:
                    print(f"{DICT[i]} got only one link, so picking {DICT[SIZE-firstBit(graph[i])-1]}") if PRINT else None
                    S = (1, SIZE-firstBit(graph[i])-1)
                    break

            """Picking the node with the min score"""
            tmp_tab = []
            if(S[1] == 0):
                for index in MAX_SCORE:
                    print(f"Analysing {index} node",binhandler(graph[index]), end=" ") if PRINT else None
                    for i in binhandler(graph[index]):
                        print(f"{DICT[SIZE-i]}", end=" ") if PRINT else None
                    s = 0
                    for bit in binhandler(graph[index]):
                        s += SCORE[SIZE-bit-1]
                    # S = (s,index) if s > S[0] else S
                    tmp_tab.append(s)
                    print(f"S: {s}") if PRINT else None

            """Saving the solutions"""
            S = (min(tmp_tab), MAX_SCORE[tmp_tab.index(min(tmp_tab))]) if S[1] == 0 else S
            SOL += f"{DICT[S[1]]}>"
            SOL_COUNT += 1

            print(f"Keeping {S} = {DICT[S[1]]} node ({S[1]})") if PRINT else None
            ADDSTR = f" Removing {DICT[S[1]]} node ({S[1]})"

            """Removing every transitions to the keeped node"""
            bitRemover = 1 << SIZE - S[1] -1
            print(f"Removing {bin(bitRemover)}") if PRINT else None
            for i,node in enumerate(graph):
                node &= (~bitRemover)

            """Printing step"""
            if PRINT:
                old_graph = tuple(bin(node) for node in graph)
                new_graph = tuple(bin(node & (~bitRemover) if i != S[1] else 0) for i,node in enumerate(graph))
                print('OLD',old_graph)
                print('NEW',new_graph)

            # Here is the new graph
            graph=tuple(node & (~bitRemover) if i != S[1] else 0 for i,node in enumerate(graph))

            if (SOL_COUNT > SIZE):
                print('')
                print(f"Graph {j} canceled {SOL} {SOL_COUNT}/{SIZE} = {graph}", tag='error', tag_color='red', color='white')
                SOL += " !!canceled!!"

        """End of while, the graph is covered (or not xd)"""
        graphGen(GRAPHS[j], addname=f" - {j} - {SOL}") if gen else None
        g_complexity = COMPLEXITY
        RESULT_LIST.append((g_complexity))
        COMPLEXITY = 0

        """Trying with bruteforce"""
        bt_len, bt_sol = BRUTEFORCE(GRAPHS[j]) if BRUTEFORCE_F else [0],[[0]]

        if(BRUTEFORCE_F):
            if (bt_len == SOL_COUNT):
                print(f"Graph {j} done {SOL} {SOL_COUNT}/{SIZE}, bt_len = {bt_len}", tag='success', tag_color='green', color='white') if PRINT else None
                print(f'{g_complexity}') if COMPLEXITY_PRINT else None
                ANALYSE[0] += 1
            else:
                sols = ""
                print("") if not COMPLEXITY_PRINT else None
                for S in bt_sol:
                    for i in S:
                        sols += DICT[i] + " "
                    sols += "|| "
                print(f"Graph {j} error {SOL} {SOL_COUNT}/{SIZE}\nGraph: {GRAPHS[j]}\nBy bruteforce = {bt_len} || {sols}\nBruteforce solutions = {bt_sol}", tag='error', tag_color='red', color='white') if not COMPLEXITY_PRINT else None
                print(f'{g_complexity}') if COMPLEXITY_PRINT else None
                print('Transitions: ') if not COMPLEXITY_PRINT else None
                for node in GRAPHS[j]:
                    print(int_to_tabin(node, SIZE)) if not COMPLEXITY_PRINT else None
                ANALYSE[1] += 1

        # Loading bar
        if(RANDOM and not COMPLEXITY_PRINT):
            print('\r['+'#'*floor((j+1)/GRAPH_NUMBER*10)+' '*floor((GRAPH_NUMBER-j-1)/GRAPH_NUMBER*10)+']', f"{j+1}/{GRAPH_NUMBER}", end="", color=color_dict[floor(j/GRAPH_NUMBER * (len(color_dict)-0.5))])
    if(RANDOM):
        print('\n')
        print(f"{ANALYSE[0]}/{GRAPH_NUMBER} graphs are covered by the algorithm", tag="ANALYSE", tag_color="blue", color="white")
        print(f"{ANALYSE[1]}/{GRAPH_NUMBER} graphs are not covered by the algorithm", tag="ANALYSE", tag_color="blue", color="white")
        print(f"{ANALYSE[2]}/{GRAPH_NUMBER} graphs are useless", tag="ANALYSE", tag_color="blue", color="white")

# main
if __name__ == "__main__":
    GRAPHS = []

    GRAPHS = [
        [int(0b010000000000000000011),
int(0b101000000000000001000),
int(0b010100000000000000100),
int(0b001011000000000000000),
int(0b000101100000000000000),
int(0b000110111000000000000),
int(0b000011010000000000000),
int(0b000001100100000000000),
int(0b000001000100000000100),
int(0b000000011010000000000),
int(0b000000000101100000000),
int(0b000000000010100010000),
int(0b000000000011010000000),
int(0b000000000000101100000),
int(0b000000000000010100001),
int(0b000000000000011010000),
int(0b000000000001000101000),
int(0b010000000000000010110),
int(0b001000001000000001000),
int(0b100000000000000001001),
int(0b100000000000001000010)],
        # [88, 227, 375, 959, 455, 718, 598, 251, 509, 486]
        ]
    
    if RANDOM: 
        # Generation of the graphs
        GRAPHS = []
        generation(GRAPHS)

    print("___________________")
    
    # Resolve the graphs
    resolve(GRAPHS, GEN)

    print(PARA_LIST, RESULT_LIST)
    data = pd.DataFrame(PARA_LIST)
    R = pd.DataFrame(RESULT_LIST)
    F = pd.concat((data,R), axis=1)
    print(F)
    F.to_csv('data.csv')


"""
010000000000000000011
101000000000000000100
010100000000000000010
001011000000000000000
000101100000000000000x
000110111000000000000x
000011010000000000000
000001100100000000000x
000001000100000000100
000000011010000000000
000000000101100000000
000000000010100010000
000000000000101100000
000000000000010100001
000000000000011010000
000000000001000101000
010000000000000010110
001000001000000001000
100000000000000001001
100000000000001000010
000000100000000000000

000000100000000000000

int(0b010000000000000000011),
int(0b101000000000000001000),
int(0b010100000000000000100),
int(0b001011000000000000000),
int(0b000101100000000000000),
int(0b000110111000000000000),
int(0b000011010000000000000),
int(0b000001100100000000000),
int(0b000001000100000000100),
int(0b000000011010000000000),
int(0b000000000101100000000),
int(0b000000000010100010000),
int(0b000000000011010000000),
int(0b000000000000101100000),
int(0b000000000000010100001),
int(0b000000000000011010000),
int(0b000000000001000101000),
int(0b010000000000000010110),
int(0b001000001000000001000),
int(0b100000000000000001001),
int(0b100000000000001000010)
"""