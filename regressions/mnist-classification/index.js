require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logistic-regression");
const loadCSV = require("../load-csv");
const plot = require("node-remote-plot");
const _ = require("lodash");
const mnist = require('mnist-data');

const mnistData = mnist.training(0,1000);


const features = mnistData.images.values.map(image => _.flatMap(image)); //flatMap remove 1 layer of nesting

const encodedLabels = mnistData.labels.values.map(label => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row
});

const regression = new LogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 5,
  batchSize: 100
});

regression.train();

const testMnistData = mnist.testing(0, 100);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row
});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log('Acc is :>> ', accuracy);