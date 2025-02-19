require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require('./logistic-regression')
const loadCSV = require("../load-csv");
const plot = require('node-remote-plot');

const { features, labels, testFeatures, testLabels } = loadCSV(
  "../data/cars.csv",
  {
    dataColumns: ["horsepower", "displacement", "weight"],
    labelColumns: ["passedemissions"],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: (value) => {
            return value === 'TRUE'? 1 : 0;
        }
    }
  }
);

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.6
});

regression.train();

console.log(regression.test(testFeatures, testLabels));

plot({
  x: regression.costHistory.reverse()
})
