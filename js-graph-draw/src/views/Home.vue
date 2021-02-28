<template>
  <div class="container" id="mynetwork"></div>
</template>

<style scoped lang="scss">
.container {
  display: flex;
  flex-direction: row;
  justify-content: center;
  width: 100vw;
}
</style>

<script>
import graph from "../assets/graph.json";

export default {
  name: "Home",
  mounted() {
    for (const [key, value] of Object.entries(graph.dfa)) {
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
  data() {
    return {
      nodes: [],
      edges: [],
    };
  },
};
</script>
