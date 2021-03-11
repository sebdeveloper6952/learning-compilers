use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fmt;
use std::fs;
use std::iter::FromIterator;
use std::process;
use std::time::Instant;

// Global variables
// the representation of the Epsilon character
const EPSILON: char = '$';

/*
 * Utility function to create hashset from u8 slice.
 */
fn hashset(data: &[u32]) -> HashSet<u32> {
    HashSet::from_iter(data.iter().cloned())
}

/*
 * Insert an explicit concatenation operator ('.') into the regular
 * expression so parsing it is easier.
*/
fn regex_insert_concat_op(regex: &String) -> String {
    let mut new_regex = String::new();
    let bytes = regex.as_bytes();
    new_regex.push(bytes[0] as char);
    for i in 1..bytes.len() {
        let prev = bytes[i - 1] as char;
        let curr = bytes[i] as char;
        if prev.is_ascii_alphabetic() || prev == '*' || prev == ')' {
            if curr.is_ascii_alphabetic() || curr == '#' || curr == '(' {
                new_regex.push('.');
            }
        }
        new_regex.push(curr);
    }

    new_regex
}

/**
 * Process the extension operators of regexes:
 *  - '+'
 *  - '?'
 */
fn preprocess_regex(regex: &String) -> String {
    let mut new_regex = String::new();
    let mut stack = Vec::new();
    let bytes = regex.as_bytes();
    for i in 0..bytes.len() {
        let curr = bytes[i] as char;
        if curr == '(' {
            let next = bytes[i + 1] as char;
            if next != '+' || next != '?' {
                stack.clear()
            }
        }
        if curr == '+' {
            let top = stack.pop().unwrap();
            if top == ')' {
                let mut temp = String::new();
                while !stack.is_empty() {
                    temp.insert(0, stack.pop().unwrap());
                }
                for c in temp.chars() {
                    new_regex.push(c);
                }
                new_regex.push_str(&format!(")*"));
                continue;
            } else {
                new_regex.push_str(&format!("{}*", top));
            }
        } else if curr == '?' {
            let top = stack.pop().unwrap();
            if top == ')' {
                let mut temp = String::new();
                while !stack.is_empty() {
                    new_regex.pop();
                    temp.insert(0, stack.pop().unwrap());
                }
                temp.push(')');
                for c in temp.chars() {
                    new_regex.push(c);
                }
                new_regex.push_str(&format!("|{})", EPSILON));
                continue;
            } else {
                new_regex.pop();
                new_regex.push_str(&format!("({}|{})", top, EPSILON));
            }
        } else {
            // push char to stack
            stack.push(curr);
            // add char to preprocessed regex
            new_regex.push(curr);
        }
    }
    println!("replaced: {}", new_regex);
    new_regex
}

/*
 * Is the char an operator?
 */
fn is_op(c: char) -> bool {
    match c {
        '*' => true,
        '.' => true,
        '|' => true,
        '?' => true,
        '+' => true,
        _ => false,
    }
}

/*
 * Is the char valid in our regular expressions?
 */
fn is_valid_regex_symbol(c: &char) -> bool {
    c.is_ascii_alphanumeric() || *c == '#' || *c == EPSILON
}

/*
 * Depth first traversal, printing the information of each node.
 * Used during development for debugging.
 */
fn depth_first_debug(root: &Node) {
    match &root.left {
        Some(n) => depth_first_debug(&n),
        _ => (),
    }
    match &root.right {
        Some(n) => depth_first_debug(&n),
        _ => (),
    }
    println!(
        "symbol: {} | pos: {} | nullable: {} | firstpos: {:?} | lastpos: {:?}",
        root.symbol, root.position, root.nullable, root.firstpos, root.lastpos
    );
}

/*
 * AST node representation
 */
#[derive(Debug, Clone)]
struct Node {
    symbol: char,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    position: u32,
    nullable: bool,
    firstpos: HashSet<u32>,
    lastpos: HashSet<u32>,
}

impl Node {
    fn new(symbol: char, position: u32, nullable: bool) -> Node {
        Node {
            symbol,
            left: None,
            right: None,
            position: position,
            nullable: nullable,
            firstpos: HashSet::new(),
            lastpos: HashSet::new(),
        }
    }

    /**
     * Create a new node to represent the UNION operator.
     */
    fn new_union_node(left: Node, right: Node) -> Node {
        // create new node
        let mut new_node = Node::new('|', 0, false);
        // compute nullable
        let nullable = left.nullable || right.nullable;
        new_node.set_nullable(nullable);
        // compute firstpos
        let union: HashSet<&u32> = left.firstpos.union(&right.firstpos).collect();
        let mut hs: HashSet<u32> = HashSet::new();
        for x in union {
            hs.insert(*x);
        }
        new_node.firstpos = hs.clone();
        // lastpos of OR node is the same as firstpos
        new_node.lastpos = hs;
        // add children
        new_node.add_left_child(left);
        new_node.add_right_child(right);

        new_node
    }

    /**
     * Create a new node to represent the CONCATENATION
     * operator.
     */
    fn new_concat_node(left: Node, right: Node) -> Node {
        // new node instance
        let mut new_node = Node::new('.', 0, false);
        // compute and set nullable
        let nullable = left.nullable && right.nullable;
        new_node.set_nullable(nullable);
        // compute firstpos
        let mut firstpos: HashSet<u32> = HashSet::new();
        if left.nullable {
            let union: HashSet<&u32> = left.firstpos.union(&right.firstpos).collect();
            for x in union {
                firstpos.insert(*x);
            }
        } else {
            for x in &left.firstpos {
                firstpos.insert(*x);
            }
        }
        new_node.firstpos = firstpos;
        // compute lastpos
        let mut lastpos: HashSet<u32> = HashSet::new();
        if right.nullable {
            let union: HashSet<&u32> = left.lastpos.union(&right.lastpos).collect();
            for x in union {
                lastpos.insert(*x);
            }
        } else {
            for x in &right.lastpos {
                lastpos.insert(*x);
            }
        }
        new_node.lastpos = lastpos;
        // add children
        new_node.add_left_child(left);
        new_node.add_right_child(right);

        new_node
    }

    fn new_star_node(left: Node) -> Node {
        let mut new_node = Node::new('*', 0, true);
        // firstpos of star node is the same as firstpos of its child
        let mut firstpos: HashSet<u32> = HashSet::new();
        for x in &left.firstpos {
            firstpos.insert(*x);
        }
        new_node.firstpos = firstpos;
        // lastpos is lastpos of child
        let mut lastpos: HashSet<u32> = HashSet::new();
        for x in &left.lastpos {
            lastpos.insert(*x);
        }
        new_node.lastpos = lastpos;
        new_node.add_left_child(left);

        new_node
    }

    fn add_left_child(&mut self, child: Node) {
        self.left = Some(Box::new(child));
    }

    fn add_right_child(&mut self, child: Node) {
        self.right = Some(Box::new(child));
    }

    fn set_nullable(&mut self, nullable: bool) {
        self.nullable = nullable;
    }
}

/*
 * NFA representation
 */
#[derive(Debug, Serialize)]
struct Nfa {
    nfa: HashMap<u32, HashMap<char, HashSet<u32>>>,
    first_state: u32,
    last_state: u32,
}

impl Nfa {
    fn new(
        nfa: HashMap<u32, HashMap<char, HashSet<u32>>>,
        first_state: u32,
        last_state: u32,
    ) -> Nfa {
        Nfa {
            nfa,
            first_state,
            last_state,
        }
    }
}

impl fmt::Display for Nfa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut states = HashSet::new();
        for (key, _) in &self.nfa {
            states.insert(key);
        }
        write!(
            f,
            "ESTADOS = {:?}\nSIMBOLOS = {:?}\nINICIO = {{0}}\nACEPTACION = {{{}}}\nTRANSICION = {:?}",
            states,
            vec!['a', 'b'],
            &self.last_state,
            &self.nfa
        )
    }
}

/*
 * DFA representation
 */
#[derive(Debug, Serialize)]
struct Dfa {
    dfa: HashMap<u32, HashMap<char, u32>>,
    accepting_states: Vec<u32>,
}

impl Dfa {
    fn new(dfa: HashMap<u32, HashMap<char, u32>>, accepting_states: Vec<u32>) -> Dfa {
        Dfa {
            dfa,
            accepting_states,
        }
    }
}

impl fmt::Display for Dfa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut states = HashSet::new();
        for (key, _) in &self.dfa {
            states.insert(key);
        }
        write!(
            f,
            "ESTADOS = {:?}\nSIMBOLOS = {:?}\nINICIO = {{0}}\nACEPTACION = {{{:?}}}\nTRANSICION = {:?}",
            states,
            vec!['a', 'b'],
            &self.accepting_states,
            &self.dfa
        )
    }
}

#[derive(Serialize)]
enum FAType {
    DFA(Dfa),
    NFA(Nfa),
}

#[derive(Serialize)]
struct FAFile {
    alphabet: HashSet<char>,
    regex: String,
    fa: FAType,
}

impl FAFile {
    fn new(alphabet: HashSet<char>, regex: String, fa: FAType) -> FAFile {
        FAFile {
            alphabet,
            regex,
            fa,
        }
    }
}

// ***************************************** Regex Parse and Tree *****************************************
/*
 * loop through input regex to build the Abstract Syntax Tree for the regex.
 * the algorithm used is the one created by Edsger Dijkstra, Shunting Yard Algorithm
 * https://en.wikipedia.org/wiki/Shunting-yard_algorithm
 *
 * Input: a regex
 * Output: the node that represents the root of the tree
 *         the global alphabet is also updated
 * The followpos table is also updated to be used later
 * in the REGEX -> DFA algorithm.
 */
fn parse_regex(
    regex: &String,
    fp_table: &mut HashMap<u32, HashSet<u32>>,
    s_table: &mut HashMap<char, HashSet<u32>>,
) -> Node {
    // holds nodes
    let mut tree_stack: Vec<Node> = Vec::new();
    // holds operators and parentheses
    let mut op_stack: Vec<char> = Vec::new();
    // tree node id
    let mut next_position: u32 = 1;
    // hashmap of operators and precedences
    let mut precedences = HashMap::<char, u8>::new();
    // populate precedences
    precedences.insert('(', 0);
    precedences.insert('|', 1);
    precedences.insert('.', 2);
    precedences.insert('*', 3);

    // for each char in the regex
    for c in regex.chars() {
        if is_valid_regex_symbol(&c) {
            // build node for c and push into tree_stack
            let mut n = Node::new(c, 0, false);
            if c != EPSILON {
                n.position = next_position;
                next_position += 1;
                // firstpos of a symbol node is only its position
                let mut hs = HashSet::new();
                hs.insert(n.position);
                n.firstpos = hs.clone();
                // lastpos of a symbol node is only its position
                n.lastpos = hs;
                // update s_table to save this char position in the tree
                if !s_table.contains_key(&c) {
                    s_table.insert(c, HashSet::new());
                }
                s_table.get_mut(&c).unwrap().insert(n.position);
            } else {
                n.nullable = true;
            }
            // push this node to the stack of nodes
            tree_stack.push(n);
        } else if c == '(' {
            // push into op_stack
            op_stack.push(c);
        } else if c == ')' {
            // pop from op_stack until '(' is found
            // build nodes for the operators popped from stack
            loop {
                // pop until '(' is found
                let op = op_stack.pop().unwrap();
                if op == '(' {
                    break;
                }

                // build node for operator
                let mut n = Node::new(op, 0, false);
                if op == '|' {
                    // OR
                    let right = tree_stack.pop().unwrap();
                    let left = tree_stack.pop().unwrap();
                    n = Node::new_union_node(left, right);
                } else if op == '.' {
                    let right = tree_stack.pop().unwrap();
                    let left = tree_stack.pop().unwrap();
                    // update followpos table
                    for x in &left.lastpos {
                        if !fp_table.contains_key(&x) {
                            fp_table.insert(*x, HashSet::new());
                        }
                        for y in &right.firstpos {
                            fp_table.get_mut(&x).unwrap().insert(*y);
                        }
                    }
                    n = Node::new_concat_node(left, right);
                } else if op == '*' {
                    let left = tree_stack.pop().unwrap();
                    n = Node::new_star_node(left);
                    // update followpos table
                    for x in &n.lastpos {
                        if !fp_table.contains_key(x) {
                            fp_table.insert(*x, HashSet::new());
                        }
                        for y in &n.firstpos {
                            fp_table.get_mut(x).unwrap().insert(*y);
                        }
                    }
                }

                // push new node into tree
                tree_stack.push(n);
            }
        } else if is_op(c) {
            // while top of stack has an operator with greater or equal precedence as 'c',
            // pop from stack and create nodes
            while op_stack.len() > 0 && precedences[op_stack.last().unwrap()] >= precedences[&c] {
                // pop top operator from stack
                let top_op = op_stack.pop().unwrap();
                // create new node for this operator
                let mut n = Node::new(top_op, 0, false);
                if top_op == '|' {
                    // OR
                    let right = tree_stack.pop().unwrap();
                    let left = tree_stack.pop().unwrap();
                    n = Node::new_union_node(left, right);
                } else if top_op == '.' {
                    let right = tree_stack.pop().unwrap();
                    let left = tree_stack.pop().unwrap();
                    // update followpos table
                    for x in &left.lastpos {
                        if !fp_table.contains_key(&x) {
                            fp_table.insert(*x, HashSet::new());
                        }
                        for y in &right.firstpos {
                            fp_table.get_mut(&x).unwrap().insert(*y);
                        }
                    }
                    n = Node::new_concat_node(left, right);
                } else if top_op == '*' {
                    let left = tree_stack.pop().unwrap();
                    n = Node::new_star_node(left);
                    // update followpos table
                    for x in &n.lastpos {
                        if !fp_table.contains_key(x) {
                            fp_table.insert(*x, HashSet::new());
                        }
                        for y in &n.firstpos {
                            fp_table.get_mut(x).unwrap().insert(*y);
                        }
                    }
                }

                // push new node to tree_stack
                tree_stack.push(n);
            }

            // push to op_stack
            op_stack.push(c);
        } else if c == EPSILON {
            continue;
        } else {
            // expression error
            panic!("Invalid character found in expression.");
        }
    }
    // process remaining nodes in op_stack
    while !op_stack.is_empty() {
        let top_op = op_stack.pop().unwrap();
        let mut n = Node::new(top_op, 0, false);
        if top_op == '|' {
            // OR
            let right = tree_stack.pop().unwrap();
            let left = tree_stack.pop().unwrap();
            n = Node::new_union_node(left, right);
        } else if top_op == '.' {
            let right = tree_stack.pop().unwrap();
            let left = tree_stack.pop().unwrap();
            // update followpos table
            for x in &left.lastpos {
                if !fp_table.contains_key(&x) {
                    fp_table.insert(*x, HashSet::new());
                }
                for y in &right.firstpos {
                    fp_table.get_mut(&x).unwrap().insert(*y);
                }
            }
            n = Node::new_concat_node(left, right);
        } else if top_op == '*' {
            let left = tree_stack.pop().unwrap();
            n = Node::new_star_node(left);
            // update followpos table
            for x in &n.lastpos {
                if !fp_table.contains_key(x) {
                    fp_table.insert(*x, HashSet::new());
                }
                for y in &n.firstpos {
                    fp_table.get_mut(x).unwrap().insert(*y);
                }
            }
        }
        // add node to tree stack
        tree_stack.push(n);
    }

    tree_stack.pop().unwrap()
}

// *********************************************** Thompson ***********************************************
/*
 * Thompson algorithm.
 * input: Abstract Syntax Tree
 * output: NFA
 */
fn thompson_algorithm(root: Node, stack: &mut Vec<Nfa>, next_state: u32) -> u32 {
    let mut i = match root.left {
        Some(n) => thompson_algorithm(*n, stack, next_state),
        None => next_state,
    };

    i = match root.right {
        Some(n) => thompson_algorithm(*n, stack, i),
        None => i,
    };

    if is_valid_regex_symbol(&root.symbol) {
        let mut nfa: HashMap<u32, HashMap<char, HashSet<u32>>> = HashMap::new();
        let mut states_set = HashSet::new();
        states_set.insert(i + 1);
        nfa.insert(i, HashMap::new());
        nfa.get_mut(&i).unwrap().insert(root.symbol, states_set);
        stack.push(Nfa::new(nfa, i, i + 1));
        i = i + 2;
    } else if root.symbol == '|' {
        // nfas for children
        let right = stack.pop().unwrap();
        let left = stack.pop().unwrap();
        // new nfa
        let mut nfa: HashMap<u32, HashMap<char, HashSet<u32>>> = HashMap::new();
        let mut map = HashMap::new();
        // new first state -> first states of left and right
        map.insert(EPSILON, hashset(&[left.first_state, right.first_state]));
        nfa.insert(i, map);
        // add left and right nfas
        for (i, m) in left.nfa {
            nfa.insert(i, m);
        }
        for (i, m) in right.nfa {
            nfa.insert(i, m);
        }
        // left last -> new last
        nfa.insert(left.last_state, HashMap::new());
        nfa.get_mut(&left.last_state)
            .unwrap()
            .insert(EPSILON, hashset(&[i + 1]));
        // right last -> new last
        nfa.insert(right.last_state, HashMap::new());
        nfa.get_mut(&right.last_state)
            .unwrap()
            .insert(EPSILON, hashset(&[i + 1]));
        // push new NFA to stack
        stack.push(Nfa::new(nfa, i, i + 1));
        i = i + 2;
    } else if root.symbol == '.' {
        // nfas for children
        let right = stack.pop().unwrap();
        let left = stack.pop().unwrap();
        let mut nfa: HashMap<u32, HashMap<char, HashSet<u32>>> = HashMap::new();
        // left.last --ε-> right.first
        nfa.insert(left.last_state, HashMap::new());
        nfa.get_mut(&left.last_state)
            .unwrap()
            .insert(EPSILON, hashset(&[right.first_state]));
        // add left and right nfas
        for (i, m) in left.nfa {
            nfa.insert(i, m);
        }
        for (i, m) in right.nfa {
            nfa.insert(i, m);
        }
        // push new NFA to stack
        stack.push(Nfa::new(nfa, left.first_state, right.last_state));
    } else if root.symbol == '*' {
        let left = stack.pop().unwrap();
        let mut nfa: HashMap<u32, HashMap<char, HashSet<u32>>> = HashMap::new();
        nfa.insert(left.last_state, HashMap::new());
        // left.last --ε-> left.first
        // left.last --ε-> new.last
        nfa.get_mut(&left.last_state)
            .unwrap()
            .insert(EPSILON, hashset(&[left.first_state, i + 1]));
        nfa.insert(i, HashMap::new());
        // new first --ε-> new last
        // new.first --ε-> left.first
        nfa.get_mut(&i)
            .unwrap()
            .insert(EPSILON, hashset(&[left.first_state, i + 1]));
        // add left nfas
        for (i, m) in left.nfa {
            nfa.insert(i, m);
        }
        // add new nfa
        stack.push(Nfa::new(nfa, i, i + 1));
        i = i + 2;
    }
    // next state id
    i
}

// *********************************************** Subset Construction ***********************************************
/*
 * Epsilon closure.
 * Returns the set of states reachable from some state 's' in 'states'
 * on e-transitions alone.
 */
fn e_closure(states: &[u32], table: &HashMap<u32, HashMap<char, HashSet<u32>>>) -> Vec<u32> {
    // initialize T to states
    let mut res = states.to_vec();
    let mut stack = states.to_vec();

    while !stack.is_empty() {
        let s = stack.pop().unwrap();
        if table.contains_key(&s) {
            let t = &table[&s];
            if t.contains_key(&EPSILON) {
                let ts = &t[&EPSILON];
                for i in ts {
                    if !res.contains(&i) {
                        res.push(*i);
                        stack.push(*i);
                    }
                }
            }
        }
    }

    res
}

/*
 * Move function.
 * Returns the set of states to which there is a transition on input symbol
 * 'symbol' from some state 's' in 'states'.
 */
fn f_move(
    states: &[u32],
    symbol: &char,
    table: &HashMap<u32, HashMap<char, HashSet<u32>>>,
) -> Vec<u32> {
    let mut res = Vec::new();

    for state in states {
        if table.contains_key(&state) {
            if table[&state].contains_key(&symbol) {
                let ts = &table[&state][&symbol];
                for t in ts {
                    if !res.contains(t) {
                        res.push(*t);
                    }
                }
            }
        }
    }

    res
}

fn subset_construction(nfa: &Nfa, alphabet: &HashSet<char>) -> Dfa {
    let mut dfa: HashMap<u32, HashMap<char, u32>> = HashMap::new();
    let mut d_states: HashSet<Vec<u32>> = HashSet::new();
    let mut d_acc_states: Vec<u32> = Vec::new();
    let mut d_states_map: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut unmarked = Vec::new();
    let mut curr_state = 0;

    // push e-closure(start_state)
    let start = e_closure(&[nfa.first_state], &nfa.nfa);
    unmarked.push(start.clone());
    d_states.insert(start.clone());
    d_states_map.insert(start.clone(), curr_state);
    curr_state += 1;

    // main loop
    while !unmarked.is_empty() {
        // pop and mark T
        let state_t = unmarked.pop().unwrap();
        // foreach input symbol
        for a in alphabet.iter() {
            // U = e-clos(move(T, a))
            let mut state_u = e_closure(&f_move(&state_t[..], &a, &nfa.nfa)[..], &nfa.nfa);
            // sort vec because of hash calculation
            state_u.sort();
            // if U is not in d_states
            if state_u.len() > 0 && !d_states.contains(&state_u) {
                d_states.insert(state_u.clone());
                unmarked.push(state_u.clone());
                d_states_map.insert(state_u.clone(), curr_state);
                curr_state += 1;
            }
            // dfa[T, a] = U
            if !dfa.contains_key(&d_states_map[&state_t]) {
                dfa.insert(d_states_map[&state_t], HashMap::new());
            }
            if state_u.len() > 0 {
                dfa.get_mut(&d_states_map[&state_t])
                    .unwrap()
                    .insert(*a, d_states_map[&state_u]);
            }
            // is U an accepting state
            if state_t.contains(&nfa.last_state) {
                if !d_acc_states.contains(&d_states_map[&state_t]) {
                    d_acc_states.push(d_states_map[&state_t]);
                }
            }
        }
    }

    Dfa::new(dfa, d_acc_states)
}

// ************************************************ Regex -> DFA ************************************************
fn regex_dfa(
    fp_table: &HashMap<u32, HashSet<u32>>,
    s_table: &HashMap<char, HashSet<u32>>,
    root: &Node,
    alphabet: &HashSet<char>,
) -> Dfa {
    let mut dfa: HashMap<u32, HashMap<char, u32>> = HashMap::new();
    let mut d_states: HashSet<Vec<u32>> = HashSet::new();
    let mut d_acc_states: Vec<u32> = Vec::new();
    let mut d_states_map: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut unmarked: Vec<Vec<u32>> = Vec::new();
    let mut curr_state = 0;

    // push start state, it is firstpos of the root of the tree
    let start_state: Vec<u32> = root.firstpos.clone().into_iter().collect();
    unmarked.push(start_state);
    // initialize d_states to contain the start state
    let start_vec: Vec<u32> = root.firstpos.iter().map(|a| *a).collect();
    d_states.insert(start_vec.clone());
    d_states_map.insert(start_vec, curr_state);
    curr_state += 1;

    // check if starting state is an accepting state
    if root.firstpos.intersection(&s_table[&'#']).count() > 0 {
        d_acc_states.push(0);
    }

    // main loop
    while !unmarked.is_empty() {
        // pop and mark T
        let state_t = unmarked.pop().unwrap();
        // foreach input symbol
        for a in alphabet {
            if *a == '#' {
                continue;
            }
            // union of followpos of a
            let mut u: HashSet<u32> = HashSet::new();
            // for each position s in state_t
            for p in &state_t {
                // for each position that corresponds to char a
                if s_table[a].contains(&p) {
                    u.extend(&fp_table[&p]);
                }
            }
            let mut u_vec: Vec<u32> = u.clone().into_iter().collect();
            // sort vec so Hash is the same for all vectors
            // containing the same elements
            u_vec.sort();
            // if U is not in Dstates
            if u_vec.len() > 0 && !d_states.contains(&u_vec) {
                d_states.insert(u_vec.clone());
                // save state as unmarked for processing
                unmarked.push(u_vec.clone());
                // update map and current state number
                d_states_map.insert(u_vec.clone(), curr_state);
                curr_state += 1;
            }
            // Update the transition table
            if !dfa.contains_key(&d_states_map[&state_t]) {
                dfa.insert(d_states_map[&state_t], HashMap::new());
            }
            // update transition table to reflect DFA[T, a] = U
            if u_vec.len() > 0 {
                dfa.get_mut(&d_states_map[&state_t])
                    .unwrap()
                    .insert(*a, d_states_map[&u_vec]);
            }
            // check if U is an accepting state
            if u.intersection(&s_table[&'#']).count() > 0 {
                d_acc_states.push(d_states_map[&u_vec]);
            }
        }
    }

    Dfa::new(dfa, d_acc_states)
}

// *********************************************** NFA Simulation ***********************************************
fn nfa_simul(nfa: &Nfa, word: &String) -> bool {
    let mut curr_states = e_closure(&[nfa.first_state], &nfa.nfa);
    for c in word.chars() {
        curr_states = e_closure(&f_move(&curr_states[..], &c, &nfa.nfa)[..], &nfa.nfa);
    }

    hashset(&curr_states[..])
        .intersection(&hashset(&[nfa.last_state]))
        .count()
        > 0
}

// *********************************************** DFA Simulation ***********************************************
fn dfa_simul(dfa: &Dfa, word: &String) -> bool {
    let mut curr_state = 0;
    for c in word.chars() {
        if dfa.dfa[&curr_state].contains_key(&c) {
            curr_state = dfa.dfa[&curr_state][&c];
        }
    }

    dfa.accepting_states.contains(&curr_state)
}

// *********************************************** Main ***********************************************
fn main() {
    // program arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!("usage: ./exec \"<regex>\" \"<word>\"");
        process::exit(1);
    }

    // program arguments
    let regex: &String = &args[1];
    let word: &String = &args[2];

    // replace '?' and '+' operators by the basic operators
    let mut proc_regex = preprocess_regex(&regex);
    // insert explicit concat operator into regex
    proc_regex = regex_insert_concat_op(&proc_regex);
    // build the extended regex used for the regex_dfa algorithm
    let mut ex_proc_regex = proc_regex.clone();
    ex_proc_regex.push_str(".#");

    // create the alphabet using the symbols in the regex
    let mut letters = regex.clone();
    letters.retain(|c| (is_valid_regex_symbol(&c) && c != EPSILON));
    let alphabet: HashSet<char> = letters.chars().into_iter().collect();

    // extended alphabet
    let mut ex_letters = proc_regex.clone();
    ex_letters.retain(|c| (is_valid_regex_symbol(&c) || c == EPSILON));
    let ex_alphabet: HashSet<char> = ex_letters.chars().into_iter().collect();

    // validate word uses the same alphabet as the regex
    let word_alphabet: HashSet<char> = word.clone().chars().into_iter().collect();
    if ex_alphabet.intersection(&word_alphabet).count() != word_alphabet.len() {
        println!("Invalid character found in word.");
        process::exit(1);
    }

    // 1. parse regex
    // followpos table
    let mut fp_table: HashMap<u32, HashSet<u32>> = HashMap::new();
    // saves positions of char in the regex AST, ex.: s_table[a] = {1, 3}
    let mut s_table: HashMap<char, HashSet<u32>> = HashMap::new();
    let tree_root_0 = parse_regex(&proc_regex, &mut fp_table, &mut s_table);
    // TODO: clean
    let mut fp_table: HashMap<u32, HashSet<u32>> = HashMap::new();
    let mut s_table: HashMap<char, HashSet<u32>> = HashMap::new();
    let tree_root_1 = parse_regex(&ex_proc_regex, &mut fp_table, &mut s_table);

    // regex to dfa
    let direct_dfa = regex_dfa(&fp_table, &s_table, &tree_root_1, &alphabet);

    // thompson
    let mut nfa_stack = Vec::new();
    thompson_algorithm(tree_root_0, &mut nfa_stack, 0);

    let nfa = nfa_stack.pop().unwrap();
    let dfa = subset_construction(&nfa, &alphabet);

    // nfa simulation
    let nfa_start = Instant::now();
    let nfa_accepts = nfa_simul(&nfa, &word);
    let nfa_duration = nfa_start.elapsed().as_nanos();

    // dfa simulation
    let dfa_start = Instant::now();
    let dfa_accepts = dfa_simul(&dfa, &word);
    let dfa_duration = dfa_start.elapsed().as_nanos();

    // direct dfa simulation
    let ddfa_start = Instant::now();
    let ddfa_accepts = dfa_simul(&direct_dfa, &word);
    let ddfa_duration = ddfa_start.elapsed().as_nanos();

    // write files
    let dfa_file = FAFile::new(alphabet.clone(), regex.to_string(), FAType::DFA(dfa));
    let serialized = serde_json::to_string(&dfa_file).unwrap();
    fs::write("./dfa.json", serialized).expect("Error writing to file.");
    let nfa_file = FAFile::new(alphabet.clone(), regex.to_string(), FAType::NFA(nfa));
    let serialized = serde_json::to_string(&nfa_file).unwrap();
    fs::write("./nfa.json", serialized).expect("Error writing to file.");
    let ddfa_file = FAFile::new(alphabet.clone(), regex.to_string(), FAType::DFA(direct_dfa));
    let serialized = serde_json::to_string(&ddfa_file).unwrap();
    fs::write("./direct-dfa.json", serialized).expect("Error writing to file.");

    // Info
    println!("************************** Regex Info ******************************");
    println!("Original regex:  {}", regex);
    println!("Processed regex: {}", proc_regex);
    println!("Extended regex:  {}", ex_proc_regex);
    println!("The alphabet found in the regex is: {:?}", alphabet);
    println!("********************** Acceptance of Word **************************");
    println!("NFA accepts        '{}' -> {}", &word, nfa_accepts);
    println!("DFA accepts        '{}' -> {}", &word, dfa_accepts);
    println!("Direct DFA accepts '{}' -> {}", &word, ddfa_accepts);
    println!("**********************  Simulation Timing **************************");
    println!("NFA:        {} nanoseconds", nfa_duration);
    println!("DFA:        {} nanoseconds", dfa_duration);
    println!("Direct DFA: {} nanoseconds", ddfa_duration);
    println!("********************************************************************");
}
