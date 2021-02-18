use std::io::{self, Write};
use std::collections::HashMap;

fn is_op(c:char) -> bool {
    match c {
        '+' => true,
        '-' => true,
        '*' => true,
        '/' => true,
        _ => false,
    }
}

fn main() {
    let mut buffer = String::new();
    let mut op_stack = Vec::<i32>::new();
    let mut prec = HashMap::<&str, u8>::new();

    // populate precedences
    prec.insert("+", 0);
    prec.insert("-", 0);
    prec.insert("*", 1);
    prec.insert("/", 1);

    // read input and save to buffer
    print!("Input expression: ");
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut buffer).unwrap();

    // reemove whitespace
    buffer.retain(|c| !c.is_whitespace());

    // loop through input, converting to postfix
    for s in buffer.chars() {
        if is_op(s) {
            println!("Found an operator.");
        } else if s.is_ascii_digit() {
            println!("Found a digit.");
        } else if c == '(' {
            println!("Opening parentheses.");
        } else if c == ')' {
            println!("Closing parentheses.")
        }
    }
}
