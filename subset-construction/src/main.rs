use std::collections::{HashMap, HashSet};

fn main() {
    // how to represent a nfa
    // - set of symbols (Vec, hardcoded for now)
    // - set of states (HashSet)
    // - starting state (u32)
    // - set of accepting states (HashSet)
    // - transition function ()
    //   - [state][symbol] -> set of states
    let mut nfa_states = HashMap::<i32, HashMap<char, HashSet<i32>>>::new();
    // this represents [0]['a'] -> {0, 1, 2}
    nfa_states.insert(0, HashMap::new());
    let mut set = HashSet::new();
    set.insert(0);
    set.insert(1);
    set.insert(2);
    nfa_states.get_mut(&0).unwrap().insert('a', set);
    println!("NFA[0]['a'] -> {:?}", nfa_states[&0][&'a']);

    // how to represent a dfa
    // - set of symbols (Vec, hardcoded for now)
    // - set of states (HashSet)
    // - starting state (u32)
    // - set of accepting states (HashSet)
    // - transition function ()
    //   - [state][symbol] -> state
    let mut dfa_states = HashMap::<i32, HashMap<char, i32>>::new();
    // this represents [0]['a'] -> 1
    dfa_states.insert(0, HashMap::new());
    dfa_states.get_mut(&0).unwrap().insert('a', 1);
    println!("DFA[0]['a'] -> {:?}", dfa_states[&0][&'a']);
}
