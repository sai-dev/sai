
import os
import sys
from argparse import ArgumentParser

from sgfmill import sgf_grammar
from sgfmill import sgf


WC_id = "WC"

_sgf_merge_verbose = False

def _update_wincount( m_node, winner ):
    if not m_node.has_property(WC_id):
        m_node.set(WC_id, "0 0 0")

    b,n,w = tuple( map( lambda s: int(s), m_node.get(WC_id).split() ) )
    if   winner == 'b':
        b += 1
    elif winner == 'w':
        w += 1
    else: #draw
        n += 1

    m_node.set(WC_id, "{0} {1} {2}".format(b,n,w) )
    if _sgf_merge_verbose :
        print( "Updated win counters: %s" % m_node.get(WC_id), file=sys.stderr )
    
# we assume that the two nodes are aligned [that is they have the same move]
def _merge_with( m_node : sgf.Tree_node, g_node : sgf.Tree_node, winner ):
    #initialise the node if not done yet
    if not m_node.has_property(WC_id):
        for pr in g_node.properties():
            m_node.set( pr, g_node.get(pr) )
        if m_node.has_property("RE"):
            m_node.unset("RE")

    _update_wincount( m_node, winner )

    if not g_node:
        return #nohing left to do for this game

    if len(g_node) != 1:
        raise Exception("Input game has variations - this is not allowed")

    next_move = g_node[0].get_move()
    for child in m_node:
        if child.get_move() == next_move:
            return _merge_with( child, g_node[0], winner )
    return _merge_with( m_node.new_child(), g_node[0], winner )


def _add_wincounters_to_comments( node : sgf.Tree_node ):
    if node.has_property(WC_id):
        node.add_comment_text(node.get(WC_id))

    if _sgf_merge_verbose:
        print("Adding win counters to comments: %s" % node.get("C"), file=sys.stderr )
        
    for child in node:
        _add_wincounters_to_comments( child )
    

        
def merge_sgf_collection(pathname,add_wc_c,clean_sgf):
    print("Value of global flag is %s" % str(_sgf_merge_verbose) )
    f = open(pathname, "rb")
    sgf_src = f.read()
    f.close()
    dirname, basename = os.path.split(pathname)
    root, ext = os.path.splitext(basename)
    dstname = os.path.join(dirname, root + "_merged" + ext)
    try:
        if clean_sgf :
            sgf_src = sgf_src.replace(b"\\n",b"\n")
        coarse_games = sgf_grammar.parse_sgf_collection(sgf_src)
    except ValueError as e:
        raise Exception("error parsing file: %s" % e)

    print( "Read %d games" % len(coarse_games) )
          
    #create a dictionary of pair of players - list of games
    all_pairings = {}
    for coarse_game in coarse_games:
        sgf_game = sgf.Sgf_game.from_coarse_game_tree(coarse_game)
        all_pairings.setdefault( (sgf_game.get_player_name('b'), sgf_game.get_player_name('w')), []).append( sgf_game )

    print( "Games grouped in %d player pairs" % len(all_pairings) )

    with open(dstname, "wb") as f:
        for games in all_pairings.values():
            print("Merging %d games for players: %s vs %s" % ( len(games), games[0].get_player_name('b'), games[0].get_player_name('w') ) )
            merged_game = sgf.Sgf_game( games[0].get_size() )

            for game in games:
                _merge_with( merged_game.get_root(), game.get_root(), game.get_winner() )

            print( "Win counters for merged game: %s" % merged_game.get_root().get(WC_id) )

            if add_wc_c :
                _add_wincounters_to_comments(merged_game.get_root())

            f.write(merged_game.serialise())

            
            
_description = """\
Merge multiple sgf games with no variations into a single one per each pair of matching opponents.
"""

def parse_args(args):
    parser = ArgumentParser(description = _description)
    parser.add_argument('filename', help='sgf collection file')
    parser.add_argument('-wcc','--add_wc_to_c', help='add win count property values to comments', action="store_true")
    parser.add_argument('-cl', '--clean_sgf',   help='clean input file to drop spurious \'\\n\'', action="store_true")
    parser.add_argument('-v',  '--verbose',     help='dump additional information to stderr ',    action="store_true")
    return parser.parse_args()
    

def main(args):
    pargs = parse_args(args)
    if pargs.verbose:
        global _sgf_merge_verbose
        _sgf_merge_verbose = True
    try:
        merge_sgf_collection(pargs.filename, pargs.add_wc_to_c, pargs.clean_sgf)
    except Exception as e:
        print("sgfmerger:", str(e), file=sys.stderr)
        sys.exit(1)

        
if __name__ == "__main__":
    main(sys.argv[1:])
    
