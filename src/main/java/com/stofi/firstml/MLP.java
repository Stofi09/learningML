package com.stofi.firstml;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;


import java.io.File;
import java.io.IOException;

public class MLP {

    public static void main(String[] args) throws IOException, InterruptedException {

        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 30;
        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        // load training data
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("linear_data_train.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

        // load the test-evaluation data
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("linear_data_train.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(learningRate, 0.9))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        System.out.println(conf.toJson());
    }
}
