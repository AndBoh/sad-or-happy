class Input {
  constructor(neuron, weight, id) {
    Object.defineProperty(this, 'neuron', {
      enumerable: false,
      value: neuron,
    });
    this.weight = weight;
    this.id = id;
  }
}

class Neuron {
  constructor(layer, previousLayer, id) {
    Object.defineProperty(this, '_layer', {
      enumerable: false,
      value: layer,
    });

    this.id = id;

    this.inputs = previousLayer
      ? previousLayer.neurons.map((neuron) => new Input(neuron, Math.random() - 0.5, `${neuron.id.split(':')[1]}:${this.id}`))
      : [0];
  }

  get inputSum() {
    return this.inputs.reduce((sum, input) => {
      return sum + input.neuron.value * input.weight;
    }, 0);
  }

  get $isFirstLayerNeuron() {
    return !(this.inputs[0] instanceof Input)
  }

  get value() {
    return this.$isFirstLayerNeuron
      ? this.inputs[0]
      : this._layer._network.activationFunction(this.inputSum);
  }

  set input(val) {
    if (!this.$isFirstLayerNeuron) {
      return;
    }

    this.inputs[0] = val;
  }

  set error(error) {
    if (this.$isFirstLayerNeuron) {
      return;
    }

    const wDelta = error * this._layer._network.derivativeFunction(this.inputSum);

    this.inputs.forEach((input) => {
      input.weight -= input.neuron.value * wDelta * this._layer._network.learningRate;
      input.neuron.error = input.weight * wDelta;
    });
  }
}

class Layer {
  constructor(neuronsCount, previousLayer, network) {
    Object.defineProperty(this, '_network', {
      enumerable: false,
      value: network,
    });
    this.id = previousLayer ? previousLayer.id + 1 : 0;
    this.neurons = [];
    for (let i = 0; i < neuronsCount; i++) {
      this.neurons.push(new Neuron(this, previousLayer, `${this.id}:${i}`));
    }
  }

  get $isFirstLayer() {
    return this.neurons[0].$isFirstLayerNeuron;
  }

  set input(val) {
    if (!this.$isFirstLayer) {
      return;
    }

    if (!Array.isArray(val)) {
      return;
    }

    if (val.length !== this.neurons.length) {
      return;
    }

    val.forEach((v, i) => this.neurons[i].input = v);
  }
}

class Network {
  static  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  static sigmoidDerivative(x) {
    return Network.sigmoid(x) * (1 - Network.sigmoid(x));
  }

  constructor(inputSize, outputSize, hiddenLayersCount = 1, learningRate = 0.5) {
    this.activationFunction = Network.sigmoid;
    this.derivativeFunction = Network.sigmoidDerivative;
    this.learningRate = learningRate;

    this.layers = [new Layer(inputSize, null, this)];

    for (let i = 0; i < hiddenLayersCount; i++) {
      const layerSize = Math.min(inputSize * 2 - 1, Math.ceil((inputSize * 2 / 3) + outputSize));
      this.layers.push(new Layer(layerSize, this.layers[this.layers.length - 1], this));
    }

    this.layers.push(new Layer(outputSize, this.layers[this.layers.length - 1], this));
  }

  get prediction() {
    return this.layers[this.layers.length - 1].neurons.map((neuron) => neuron.value);
  }

  set input(val) {
    this.layers[0].input = val;
  }

  trainOnce(dataSet) {
    if (!Array.isArray(dataSet)) {
      return;
    }

    dataSet.forEach((dataCase) => {
      const [input, expected] = dataCase;

      this.input = input;
      n.prediction.forEach((r, i) => {
        this.layers[this.layers.length - 1].neurons[i].error = r - expected[i];
      });
    });
  }

  train(dataSet, epochs = 100000) {
    return new Promise(resolve => {
      for (let i = 0; i < epochs; i++) {
        this.trainOnce(dataSet);
      }
      resolve();
    });
  }

  get weights() {
    return JSON.stringify(this.layers.map(l => l.neurons.map(n => n.inputs)).flat(2)
      .filter((input) => input instanceof Input)
      .map(w => [w.id, w.weight]));
  }

  set weights(modelStr) {
    let weights = []

    try {
      weights = JSON.parse(modelStr);
      weights.forEach((w) => {
        const [fromNeuronIndex, layerIndex, toNeuronIndex] = w[0].split(':');
        this.layers[+layerIndex].neurons[+toNeuronIndex].inputs[+fromNeuronIndex].weight = w[1];
      });
    } catch (e) {
      console.log(e.message);
      return;
    }
  }

  get model() {
    return [
      this.layers[0].neurons.length,
      this.layers[this.layers.length - 1].neurons.length,
      this.layers.length - 2,
      this.weights
    ].join('*');
  }

  static fromModel(modelStr) {
    const [inputSize, outputSize, hiddenLayersCount, weights] = modelStr.split('*');
    const network = new Network(+inputSize, +outputSize, +hiddenLayersCount);
    network.weights = weights;
    return network;
  }
}

const n = new Network(2, 1);

// const data = [
//   [[0, 0], [0]],
//   [[0, 1], [1]],
//   [[1, 0], [1]],
//   [[1, 1], [0]],
// ];
//
// n.train(data).then(() => {
//   n.input = [1, 0];
//   console.log(n.prediction);
// })
//
// console.log(n)
