use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

const TAG_NUM: u32 = 256;
const TAG_WORD: u32 = 257;
const TAG_TRUE: u32 = 258;
const TAG_FALSE: u32 = 259;
const TAG_TOKEN: u32 = 260;

/**
 * Token
 */
#[derive(Debug)]
struct Token {
    tag: u32,
    lexeme: Option<String>,
    value: Option<u32>,
}

impl Token {
    fn new(tag: u32, lexeme: Option<String>, value: Option<u32>) -> Token {
        Token { tag, lexeme, value }
    }
}

struct Lexer {
    line: u32,
    words_table: HashMap<String, Token>,
    chars: Vec<char>,
    index: usize,
}

impl Lexer {
    fn new(buffer: String) -> Lexer {
        let table = HashMap::new();
        let mut chars: Vec<char> = buffer.chars().collect();
        chars.retain(|&c| c != ' ' || c != '\t');

        Lexer {
            line: 1,
            words_table: table,
            chars: chars,
            index: 0,
        }
    }
    fn reserve(&mut self, word: Token) {
        self.words_table.insert(String::from("key"), word);
    }
    fn scan(&mut self) -> Option<Token> {
        if self.index == self.chars.len() {
            return None;
        }
        let mut peek;
        // skip whitespace and new lines
        loop {
            peek = self.chars[self.index];
            if peek == ' ' || peek == '\t' {
                self.index += 1;
                continue;
            } else if peek == '\n' {
                self.line = self.line + 1;
                self.index += 1;
            } else {
                break;
            }
        }

        // process current character and next to form a token
        if peek.is_ascii_digit() {
            let mut value = 0;
            loop {
                value = 10 * value + peek.to_digit(10).unwrap();
                self.index += 1;
                if self.index == self.chars.len() {
                    break;
                }
                peek = self.chars[self.index];
                if !peek.is_ascii_digit() {
                    break;
                }
            }

            return Some(Token {
                tag: TAG_NUM,
                lexeme: None,
                value: Some(value),
            });
        } else if peek.is_ascii_alphabetic() {
            let mut s = String::new();
            s.push(peek);
            loop {
                self.index += 1;
                if self.index == self.chars.len() {
                    break;
                }
                peek = self.chars[self.index];
                if !peek.is_ascii_alphabetic() {
                    break;
                }
                s.push(peek);
            }
            let word = self.words_table.get(&s);
            return match word {
                Some(w) => {
                    let s = w.lexeme.as_ref().unwrap().clone();
                    return Some(Token {
                        tag: TAG_WORD,
                        lexeme: Some(s),
                        value: None,
                    });
                }
                _ => Some(Token {
                    tag: TAG_WORD,
                    lexeme: Some(String::from(s)),
                    value: None,
                }),
            };
        }

        // increment index for next scan
        self.index += 1;

        // return a token that is not a number or a word
        Some(Token {
            tag: TAG_TOKEN,
            lexeme: Some(String::from(peek)),
            value: None,
        })
    }
}

fn main() {
    // read src file
    let mut f = BufReader::new(File::open("input.c").expect("File::open failed."));
    let mut buffer = String::new();
    f.read_to_string(&mut buffer)
        .expect("BufReader::read_to_string failed.");

    // create lexer and fill table with reserved words
    let mut lexer = Lexer::new(buffer);
    lexer.reserve(Token::new(TAG_TRUE, Some(String::from("TRUE")), None));
    lexer.reserve(Token::new(TAG_FALSE, Some(String::from("FALSE")), None));

    while let Some(t) = lexer.scan() {
        println!("{:?}", t);
    }

    println!("Lexer found {} lines in the source program.", lexer.line);
}
