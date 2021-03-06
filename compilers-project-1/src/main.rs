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
const EPSILON: char = 'ε';

/*
 * Utility function to create hashset from u8 slice.
 */
fn hashset(data: &[u32]) -> HashSet<u32> {
    HashSet::from_iter(data.iter().cloned())
}

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
    c.is_ascii_alphabetic() || *c == '#'
}

fn depth_first_debug(root: Node) {
    match root.left {
        Some(n) => depth_first_debug(*n),
        _ => (),
    }
    match root.right {
        Some(n) => depth_first_debug(*n),
        _ => (),
    }
    println!(
        "symbol: {} | pos: {} | nullable: {} | firstpos: {:?} | lastpos: {:?} | followpos: {:?}",
        root.symbol, root.position, root.nullable, root.firstpos, root.lastpos, root.followpos
    );
}

/*
 * AST node representation
 */
#[derive(Debug)]
struct Node {
    symbol: char,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    position: u32,
    nullable: bool,
    firstpos: HashSet<u32>,
    lastpos: HashSet<u32>,
    followpos: HashSet<u32>,
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
            followpos: HashSet::new(),
        }
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
fn parse_regex(regex: &String, fp_table: &HashMap<u32, HashSet<u32>>) -> Node {
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
    precedences.insert('?', 3);
    precedences.insert('+', 3);

    // for each char in the regex
    for c in regex.chars() {
        if is_valid_regex_symbol(&c) {
            // build node for c and push into tree_stack
            let mut n = Node::new(c, next_position, false);
            // firstpos of a symbol node is only its position
            let mut hs = HashSet::new();
            hs.insert(n.position);
            n.firstpos = hs.clone();
            // lastpos of a symbol node is only its position
            n.lastpos = hs;
            tree_stack.push(n);
            next_position += 1;
        } else if c == EPSILON {
            // insert node without position
            tree_stack.push(Node::new(c, 0, true));
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
                    // compute nullable
                    let nullable = left.nullable || right.nullable;
                    n.set_nullable(nullable);
                    // compute firstpos
                    let union: HashSet<&u32> = left.firstpos.union(&right.firstpos).collect();
                    let mut hs: HashSet<u32> = HashSet::new();
                    for x in union {
                        hs.insert(*x);
                    }
                    n.firstpos = hs.clone();
                    // lastpos of OR node is the same as firstpos
                    n.lastpos = hs;
                    // add children
                    n.add_left_child(left);
                    n.add_right_child(right);
                } else if op == '.' {
                    // CONCAT
                    // pop children from stack
                    let right = tree_stack.pop().unwrap();
                    let left = tree_stack.pop().unwrap();
                    // compute and set nullable
                    let nullable = left.nullable && right.nullable;
                    n.set_nullable(nullable);
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
                    n.firstpos = firstpos;
                    // compute lastpos
                    let mut lastpos: HashSet<u32> = HashSet::new();
                    if left.nullable {
                        let union: HashSet<&u32> = left.lastpos.union(&right.lastpos).collect();
                        for x in union {
                            lastpos.insert(*x);
                        }
                    } else {
                        for x in &left.lastpos {
                            lastpos.insert(*x);
                        }
                    }
                    n.lastpos = lastpos;
                    // add children
                    n.add_left_child(left);
                    n.add_right_child(right);
                } else if op == '*' {
                    // KLEENE
                    let left = tree_stack.pop().unwrap();
                    // firstpos of star node is the same as firstpos of its child
                    let mut firstpos: HashSet<u32> = HashSet::new();
                    for x in &left.firstpos {
                        firstpos.insert(*x);
                    }
                    n.firstpos = firstpos;
                    // lastpos is lastpos of child
                    let mut lastpos: HashSet<u32> = HashSet::new();
                    for x in &left.lastpos {
                        lastpos.insert(*x);
                    }
                    n.lastpos = lastpos;
                    // a star node is always nullable
                    n.set_nullable(true);
                    // add only child
                    n.add_left_child(left);
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
                    // compute nullable
                    let nullable = left.nullable || right.nullable;
                    n.set_nullable(nullable);
                    // compute firstpos
                    let union: HashSet<&u32> = left.firstpos.union(&right.firstpos).collect();
                    let mut hs: HashSet<u32> = HashSet::new();
                    for x in union {
                        hs.insert(*x);
                    }
                    n.firstpos = hs.clone();
                    // lastpos of OR node is the same as firstpos
                    n.lastpos = hs;
                    n.add_left_child(left);
                    n.add_right_child(right);
                } else if top_op == '.' {
                    // CONCAT
                    let right = tree_stack.pop().unwrap();
                    let left = tree_stack.pop().unwrap();
                    // compute nullable
                    let nullable = left.nullable && right.nullable;
                    n.set_nullable(nullable);
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
                    n.firstpos = firstpos;
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
                    n.lastpos = lastpos;
                    // compute followpos
                    // add children
                    n.add_left_child(left);
                    n.add_right_child(right);
                } else if top_op == '*' {
                    // KLEENE
                    let left = tree_stack.pop().unwrap();
                    // star node is always nullable
                    n.set_nullable(true);
                    // firstpos of star node is the same as firstpos of its child
                    let mut firstpos: HashSet<u32> = HashSet::new();
                    for x in &left.firstpos {
                        firstpos.insert(*x);
                    }
                    n.firstpos = firstpos;
                    // lastpos is lastpos of child
                    let mut lastpos: HashSet<u32> = HashSet::new();
                    for x in &left.lastpos {
                        lastpos.insert(*x);
                    }
                    n.lastpos = lastpos;
                    // add only child
                    n.add_left_child(left);
                }

                // push new node to tree_stack
                tree_stack.push(n);
            }

            // push to op_stack
            op_stack.push(c);
        } else {
            // expression error
            panic!("Invalid charcter found in expression.");
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
            // compute nullable
            let nullable = left.nullable || right.nullable;
            n.set_nullable(nullable);
            // compute firstpos
            let union: HashSet<&u32> = left.firstpos.union(&right.firstpos).collect();
            let mut hs: HashSet<u32> = HashSet::new();
            for x in union {
                hs.insert(*x);
            }
            n.firstpos = hs.clone();
            // lastpos of OR node is the same as firstpos
            n.lastpos = hs;
            n.add_left_child(left);
            n.add_right_child(right);
        } else if top_op == '.' {
            // CONCAT
            let right = tree_stack.pop().unwrap();
            let left = tree_stack.pop().unwrap();
            // compute nullable
            let nullable = left.nullable && right.nullable;
            n.set_nullable(nullable);
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
            n.firstpos = firstpos;
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
            n.lastpos = lastpos;
            n.add_left_child(left);
            n.add_right_child(right);
        } else if top_op == '*' {
            // KLEENE
            let left = tree_stack.pop().unwrap();
            // star node is always nullable
            n.set_nullable(true);
            // firstpos of star node is the same as firstpos of its child
            let mut firstpos: HashSet<u32> = HashSet::new();
            for x in &left.firstpos {
                firstpos.insert(*x);
            }
            n.firstpos = firstpos;
            // lastpos is lastpos of child
            let mut lastpos: HashSet<u32> = HashSet::new();
            for x in &left.lastpos {
                lastpos.insert(*x);
            }
            n.lastpos = lastpos;
            // add only child
            n.add_left_child(left);
        }
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
                    res.push(*t);
                }
            }
        }
    }

    res
}

fn subset_construction(
    nfa: &Nfa,
    // nfa: &HashMap<u32, HashMap<char, HashSet<u32>>>,
    // start_state: u32,
    // final_state: u32,
    alphabet: &HashSet<char>,
) -> Dfa {
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
            let state_u = e_closure(&f_move(&state_t[..], &a, &nfa.nfa)[..], &nfa.nfa);
            // if U not in d_states
            if !d_states.contains(&state_u) {
                d_states.insert(state_u.clone());
                unmarked.push(state_u.clone());
                d_states_map.insert(state_u.clone(), curr_state);
                curr_state += 1;
            }
            // dfa[T, a] = U
            let mut h_map = HashMap::new();
            h_map.insert(a, state_u.clone());
            if !dfa.contains_key(&d_states_map[&state_t]) {
                dfa.insert(d_states_map[&state_t], HashMap::new());
            }
            dfa.get_mut(&d_states_map[&state_t])
                .unwrap()
                .insert(*a, d_states_map[&state_u]);

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
        curr_state = dfa.dfa[&curr_state][&c];
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

    // insert explicit concat operator into regex
    println!("original regex {}", regex);
    let regex = regex_insert_concat_op(&regex);
    println!("regex: {}", regex);

    // global alphabet
    let mut letters = regex.clone();
    letters.retain(|c| (is_valid_regex_symbol(&c) && c != EPSILON));
    let alphabet: HashSet<char> = letters.chars().into_iter().collect();

    // validate word uses the same alphabet as the regex
    let word_alphabet: HashSet<char> = word.clone().chars().into_iter().collect();
    if alphabet.intersection(&word_alphabet).count() != word_alphabet.len() {
        println!("Invalid character found in word");
        process::exit(1);
    }

    // 1. parse regex
    let mut fp_table: HashMap<u32, HashSet<u32>> = HashMap::new();
    let tree_root = parse_regex(&regex, &fp_table);
    depth_first_debug(tree_root);
    println!("{:?}", fp_table);

    // // thompson
    // let mut nfa_stack = Vec::new();
    // // thompson_algorithm(tree_stack.pop().unwrap(), &mut nfa_stack, 0);
    // thompson_algorithm(tree_root, &mut nfa_stack, 0);

    // let nfa = nfa_stack.pop().unwrap();
    // let dfa = subset_construction(&nfa, &alphabet);

    // // serialize DFA to json and write to file
    // let serialized = serde_json::to_string(&nfa).unwrap();
    // fs::write("./nfa-graph.json", serialized).expect("Error writing to file.");

    // // serialize DFA to json and write to file
    // let serialized = serde_json::to_string(&dfa).unwrap();
    // fs::write("./dfa-graph.json", serialized).expect("Error writing to file.");

    // // nfa simulation
    // let nfa_start = Instant::now();
    // let nfa_accepts = nfa_simul(&nfa, &word);
    // let nfa_duration = nfa_start.elapsed().as_micros();

    // let dfa_start = Instant::now();
    // let dfa_accepts = dfa_simul(&dfa, &word);
    // let dfa_duration = dfa_start.elapsed().as_micros();

    // // Info
    // println!("The alphabet found in the regex is: {:?}", alphabet);
    // println!("NFA accepts '{}' -> {}'", &word, nfa_accepts);
    // // simulate dfa with word
    // println!("DFA accepts '{}' -> {}'", &word, dfa_accepts);
    // println!("**************************** Timing ****************************");
    // println!("NFA: {} μs", nfa_duration);
    // println!("DFA: {} μs", dfa_duration);
    // println!("****************************************************************");

    // // file for each automaton
    // println!("{}", &nfa.to_string());
    // println!("");
    // println!("{}", &dfa.to_string());
}
