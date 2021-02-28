<template>
  <div class="container">
    <h1 class="title">Compis Proyecto 1</h1>
    <b-field class="file is-primary" :class="{ 'has-name': !!file }">
      <b-upload v-model="file" class="file-label" @input="onFileChange">
        <span class="file-cta">
          <b-icon class="file-icon" icon="upload"></b-icon>
          <span class="file-label">Escoger archivo de DFA...</span>
        </span>
        <span class="file-name" v-if="file">
          {{ file.name }}
        </span>
      </b-upload>
    </b-field>
    <div id="mynetwork"></div>
  </div>
</template>

<script>
import graph from "../assets/graph.json";

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
      console.log(this.graphJsonFile);
      for (const [key, value] of Object.entries(this.graphJsonFile.dfa)) {
        const color = graph.accepting_states.includes(Number(key))
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
  },
};
</script>
