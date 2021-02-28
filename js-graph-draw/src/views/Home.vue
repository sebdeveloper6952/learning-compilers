<template>
  <div class="container mt-4">
    <h1 class="title">Compis Proyecto 1</h1>
    <h1 class="subtitle">Sebastian Arriola</h1>
    <div class="columns is-centered">
      <b-field class="file is-primary" :class="{ 'has-name': !!file }">
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
    <div class="columns is-centered mt-4">
      <div id="mynetwork"></div>
    </div>
  </div>
</template>

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
      this.nodes = [];
      this.edges = [];
      if (this.graphJsonFile.dfa) this.parseDfa();
      else if (this.graphJsonFile.nfa) this.parseNfa();
      this.file = {};
    },
    parseDfa() {
      for (const [key, value] of Object.entries(this.graphJsonFile.dfa)) {
        const color = this.graphJsonFile.accepting_states.includes(Number(key))
          ? "red"
          : "black";
        this.nodes.push({
          id: key,
          label: key,
          color: color,
          font: {
            color: "white",
          },
        });
        this.edges.push({
          from: key,
          to: value["a"],
          label: "a",
          color: "black",
        });
        this.edges.push({
          from: key,
          to: value["b"],
          label: "b",
          color: "black",
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
      const network = new vis.Network(container, data, options);
    },

    parseNfa() {
      for (const [key, value] of Object.entries(this.graphJsonFile.nfa)) {
        const color =
          this.graphJsonFile.first_state == key
            ? "blue"
            : this.graphJsonFile.last_state == key
            ? "red"
            : "black";
        this.nodes.push({
          id: key,
          label: key,
          color: color,
          font: {
            color: "white",
          },
        });

        const alphabet = ["a", "b", "ε"];
        const obj = this.graphJsonFile.nfa[key];
        alphabet.forEach((a) => {
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
