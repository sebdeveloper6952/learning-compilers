# js-graph-draw

## Project Setup with Docker

1. to install the webapp dependencies, run: `docker run -it --rm --name js-graph-draw-webapp -v "$PWD":/usr/src/app -w /usr/src/app node:14 npm install`
2. to run the webapp, run the following command: `docker run -it --rm --name js-graph-draw-webapp -p 8080:80 -v "$PWD":/usr/src/app -w /usr/src/app -p 8080:8080 node:14 npm run serve`
3. visit `localhost:8080` on your web browser to use the web application.

## Project Setup (without Docker)

```
npm install
```

### Compiles and hot-reloads for development

```
npm run serve
```

### Compiles and minifies for production

```
npm run build
```

### Customize configuration

See [Configuration Reference](https://cli.vuejs.org/config/).
