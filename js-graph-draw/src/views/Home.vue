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
            <p class="subtitle" style="color:white">
              Los estados azules son estados iniciales.
            </p>
          </div>
        </div>
      </div>
      <div class="column is-4">
        <div class="card red">
          <div class="card-content">
            <p class="subtitle" style="color:white">
              Los estados rojos son estados finales.
            </p>
          </div>
        </div>
      </div>
    </div>
    <div class="columns is-centered">
      <div class="column is-8">
        <b-notification v-if="faType" type="is-info" has-icon :closable="false">
          {{ faType }} para
          <p class="title">{{ regex }}</p>
        </b-notification>
      </div>
    </div>
    <div class="columns is-centered mt-4">
      <div class="card custom-card">
        <div class="card-content">
          <div style="border-radius: 5px;" id="mynetwork"></div>
        </div>
      </div>
    </div>
    <div class="columns is-centered" v-if="faType == 'DFA'">
      <div class="column is-8">
        <b-field label="Introducir Palabra">
          <b-input v-model="word"></b-input>
        </b-field>
        <b-button expanded type="is-info" @click="simulateWord"
          >Simular</b-button
        >
      </div>
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

.custom-card {
  background-color: #e9e9e9;
  padding: 2px;
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
      faType: false,
      network: {},
      originalCurrNode: 0,
      currentNode: 0,
      word: "",
      acceptingStates: [],
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
      this.faType = "DFA";
      const dfa = this.graphJsonFile.fa.DFA.dfa;
      this.acceptingStates = this.graphJsonFile.fa.DFA.accepting_states;
      for (const [key, value] of Object.entries(dfa)) {
        const color =
          key == 0
            ? "#3377ff"
            : this.acceptingStates.includes(Number(key))
            ? "#ff5733"
            : "black";
        this.nodes.push({
          id: +key,
          label: key,
          color: color,
          font: {
            color: "white",
          },
        });
        this.alphabet.forEach((a) => {
          if (value[a]) {
            this.edges.push({
              from: +key,
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
      this.network = new vis.Network(container, data, options);
      this.network.focus(String(this.currentNode));
      this.network.selectNodes([String(this.currentNode)], false);
    },

    parseNfa() {
      this.faType = "NFA";
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
      this.network = new vis.Network(container, data, options);
    },
    sleep(ms) {
      return new Promise((r) => setTimeout(r, ms));
    },
    async simulateWord() {
      this.currentNode = this.originalCurrNode;
      const options = {
        scale: 2,
        animation: true,
      };

      try {
        for (let i = 0; i < this.word.length; i++) {
          const edges = this.edges.filter((e) => e.from == this.currentNode);
          let found = false;
          for (let j = 0; j < edges.length; j++) {
            if (edges[j].label == this.word[i]) {
              // match
              found = true;
              this.network.selectEdges([edges[j].id]);
              this.currentNode = edges[j].to;
              this.network.focus(this.currentNode, options);
              await this.sleep(1000);
            }
          }
          if (!found) throw "error";
        }
        if (this.acceptingStates.includes(this.currentNode)) {
          this.$buefy.notification.open({
            duration: 5000,
            message: `La palabra es aceptada por la expresión regular.`,
            position: "is-bottom-right",
            type: "is-success",
            hasIcon: true,
          });
        } else throw "error";
      } catch {
        this.$buefy.notification.open({
          duration: 5000,
          message: `La palabra no es aceptada por la expresión regular.`,
          position: "is-bottom-right",
          type: "is-danger",
          hasIcon: true,
        });
      }
    },
  },
};
</script>
