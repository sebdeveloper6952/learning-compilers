/*
 * A NFA is represented as: HashMap<u8, HashMap<char, HashSet<u8>>>
 * A DFA is represented as: HashMap<u8, HashMap<char, u8>>
 */

use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

const ALPHABET: [char; 2] = ['a', 'b'];

/*
 * Utility function to create hashset from u8 slice.
 */
fn hashset(data: &[u8]) -> HashSet<u8> {
    HashSet::from_iter(data.iter().cloned())
}

/*
 * Epsilon closure.
 * Returns the set of states reachable from some state 's' in 'states'
 * on e-transitions alone.
 */
fn e_closure(states: &[u8], table: &HashMap<u8, HashMap<char, HashSet<u8>>>) -> Vec<u8> {
    // initialize T to states
    let mut res = states.to_vec();
    let mut stack = states.to_vec();

    while !stack.is_empty() {
        let s = stack.pop().unwrap();
        if table.contains_key(&s) {
            let t = &table[&s];
            if t.contains_key(&'.') {
                let ts = &t[&'.'];
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
    states: &[u8],
    symbol: &char,
    table: &HashMap<u8, HashMap<char, HashSet<u8>>>,
) -> Vec<u8> {
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

/*
 * NFA -> DFA
 */
fn subset_construction(
    nfa: &HashMap<u8, HashMap<char, HashSet<u8>>>,
) -> HashMap<u8, HashMap<char, u8>> {
    let mut dfa: HashMap<u8, HashMap<char, u8>> = HashMap::new();
    let mut d_states: HashSet<Vec<u8>> = HashSet::new();
    let mut d_states_map: HashMap<Vec<u8>, u8> = HashMap::new();
    let mut unmarked = Vec::new();
    let mut curr_state = 0;

    // push e-closure(start_state)
    let start = e_closure(&vec![0], &nfa);
    unmarked.push(start.clone());
    d_states.insert(start.clone());
    d_states_map.insert(start.clone(), curr_state);
    curr_state += 1;

    // main loop
    while !unmarked.is_empty() {
        // pop and mark T
        let state_t = unmarked.pop().unwrap();
        // foreach input symbol
        for a in ALPHABET.iter() {
            // U = e-clos(move(T, a))
            let state_u = e_closure(&f_move(&state_t[..], &a, &nfa)[..], &nfa);
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
        }
    }

    dfa
}

fn main() {
    // hardcoded nfa
    /*      e      a       b
     *  0 {1,7}   {}       {}
     *  1 {2,4}   {}       {}
     *  2 {}      {3}      {}
     *  3 {6}     {}       {}
     *  4 {}      {}       {5}
     *  5 {6}     {}       {}
     *  6 {1,7}   {}       {}
     *  7 {}      {8}      {}
     *  8 {}      {}       {}
     */
    let mut nfa_states = HashMap::<u8, HashMap<char, HashSet<u8>>>::new();
    for i in 0..10 {
        nfa_states.insert(i, HashMap::new());
    }
    // state 0
    nfa_states
        .get_mut(&0)
        .unwrap()
        .insert('.', hashset(&vec![1, 7]));
    // state 1
    nfa_states
        .get_mut(&1)
        .unwrap()
        .insert('.', hashset(&vec![2, 4]));
    // state 2
    nfa_states
        .get_mut(&2)
        .unwrap()
        .insert('a', hashset(&vec![3]));
    // state 3
    nfa_states
        .get_mut(&3)
        .unwrap()
        .insert('.', hashset(&vec![6]));
    // state 4
    nfa_states
        .get_mut(&4)
        .unwrap()
        .insert('b', hashset(&vec![5]));
    // state 5
    nfa_states
        .get_mut(&5)
        .unwrap()
        .insert('.', hashset(&vec![6]));
    // state 6
    nfa_states
        .get_mut(&6)
        .unwrap()
        .insert('.', hashset(&vec![1, 7]));
    // state 7
    nfa_states
        .get_mut(&7)
        .unwrap()
        .insert('a', hashset(&vec![8]));
    // state 8
    nfa_states
        .get_mut(&8)
        .unwrap()
        .insert('b', hashset(&vec![9]));
    // state 9
    nfa_states
        .get_mut(&9)
        .unwrap()
        .insert('b', hashset(&vec![10]));
    //  dfa construction test
    let dfa = subset_construction(&nfa_states);
    for a in dfa {
        println!("{:?}", a);
    }
}
