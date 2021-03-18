# Compilers Course 1 - Project 1

### How to run

This project consists of two programs. A finite automata generator and a web application to visualize the generated finite automata.
Please install Docker to run both programs.

1. cd into the `compilers-project-1` directory.
2. run the following command to create the project executable: `docker run --rm --user "$(id -u)":"$(id -g)" -v "$PWD:/usr/src/myapp" -w /usr/src/myapp rust cargo build --release`
3. run the following command to run the executable (replace regex and word with a regular expression and a word, both enclosed in \" \"): `docker run --rm --user "$(id -u)":"$(id -g)" -v "$PWD:/usr/src/myapp" -w /usr/src/myapp rust ./target/release/compilers-project-1 "<regex>" "<word>"`
4. One JSON file for each generated automata will be written to the current directory. These JSON files can be loaded by the web application to visualize the automata.
5. cd into the `js-grap-draw` directory
6. to install the webapp dependencies, run: `docker run -it --rm --name my-running-script -p 8080:80 -v "$PWD":/usr/src/app -w /usr/src/app node:14 npm install`
7. to run the webapp, run the following command: `docker run -it --rm --name my-running-script -p 8080:80 -v "$PWD":/usr/src/app -w /usr/src/app -p 8080:8080 node:14 npm run serve`
8. visit `localhost:8080` on your web browser to use the web application.
