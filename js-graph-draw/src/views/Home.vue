<template>
  <div class="container mt-4">
    <h1 class="title">Compis Proyecto 1</h1>
    <h1 class="subtitle">Sebastián Arriola</h1>
    <div class="columns is-centered">
      <b-field class="file is-info" :class="{ 'has-name': !!file }">
        <b-upload v-model="file" class="file-label" @input="onFileChange">
          <span class="file-cta">
            <b-icon class="file-icon" icon="upload"></b-icon>
            <span class="file-label">Escoger archivo de automata...</span>
          </span>
          <span class="file-name" v-if="file">
            {{ file.name }}
          </span>
        </b-upload>
      </b-field>
    </div>
    <div class="columns is-centered">
      <div class="column is-4">
        <div class="card blue">
          <div class="card-content">
            <p class="title" style="color:white">
              Los nodos azules son estados iniciales.
            </p>
          </div>
        </div>
      </div>
      <div class="column is-4">
        <div class="card red">
          <div class="card-content">
            <p class="title" style="color:white">
              Los nodos rojos son estados finales.
            </p>
          </div>
        </div>
      </div>
    </div>
    <div class="columns is-centered mt-4">
      <p class="title">{{ regex }}</p>
    </div>
    <div class="columns is-centered mt-4">
      <div id="mynetwork"></div>
    </div>
  </div>
</template>

<style lang="scss" scoped>
#mynetwork {
  border: 1px solid black;
}

.red {
  background-color: #ff5733;
}

.blue {
  background-color: #3377ff;
}
</style>

<script>
export default {
  name: "Home",
  mounted() {},
  data() {
    return {
      file: {},
      graphJsonFile: {},
      nodes: [],
      edges: [],
      alphabet: [],
      regex: "",
    };
  },
  methods: {
    onFileChange(file) {
      const reader = new FileReader();
      reader.onload = function(event) {
        var contents = event.target.result;
        this.graphJsonFile = JSON.parse(contents);
        this.parseJsonIntoGraph();
      }.bind(this);
      reader.readAsText(file);
    },
    parseJsonIntoGraph() {
      this.alphabet = [];
      this.regex = "";
      this.nodes = [];
      this.edges = [];
      this.alphabet = this.graphJsonFile.alphabet;
      this.alphabet.push("$");
      this.regex = this.graphJsonFile.regex;
      this.file = {};
      if (this.graphJsonFile.fa.DFA) this.parseDfa();
      else if (this.graphJsonFile.fa.NFA) this.parseNfa();
    },
    parseDfa() {
      const dfa = this.graphJsonFile.fa.DFA.dfa;
      const acceptingStates = this.graphJsonFile.fa.DFA.accepting_states;
      for (const [key, value] of Object.entries(dfa)) {
        const color =
          key == 0
            ? "#3377ff"
            : acceptingStates.includes(Number(key))
            ? "#ff5733"
            : "black";
        this.nodes.push({
          id: key,
          label: key,
          color: color,
          font: {
            color: "white",
          },
        });
        this.alphabet.forEach((a) => {
          if (value[a]) {
            this.edges.push({
              from: key,
              to: value[a],
              label: a,
              color: "black",
            });
          }
        });
      }

      // create a network
      const container = document.getElementById("mynetwork");
      const data = {
        nodes: this.nodes,
        edges: this.edges,
      };
      const options = {
        edges: {
          arrows: {
            to: {
              enabled: true,
              type: "arrow",
            },
          },
        },
      };

      // create network
      const network = new vis.Network(container, data, options);
    },

    parseNfa() {
      const nfa = this.graphJsonFile.fa.NFA.nfa;
      const startState = this.graphJsonFile.fa.NFA.first_state;
      const lastState = this.graphJsonFile.fa.NFA.last_state;
      for (const [key, value] of Object.entries(nfa)) {
        const color =
          startState == key
            ? "#3377ff"
            : lastState == key
            ? "#ff5733"
            : "black";
        this.nodes.push({
          id: key,
          label: key,
          color: color,
          font: {
            color: "white",
          },
        });

        const obj = nfa[key];
        this.alphabet.forEach((a) => {
          if (obj[a]) {
            obj[a].forEach((to) => {
              this.edges.push({
                from: key,
                to: to,
                label: a,
                color: "black",
              });
            });
          }
        });
      }

      // node for last state
      this.nodes.push({
        id: lastState,
        label: String(lastState),
        color: "#ff5733",
        font: {
          color: "white",
        },
      });

      // create a network
      const container = document.getElementById("mynetwork");
      const data = {
        nodes: this.nodes,
        edges: this.edges,
      };
      const options = {
        edges: {
          arrows: {
            to: {
              enabled: true,
              type: "arrow",
            },
          },
        },
      };
      const network = new vis.Network(container, data, options);
    },
  },
};
</script>
