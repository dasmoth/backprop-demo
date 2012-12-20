library backprop;

import 'dart:math';

double logit(double x) {
  return 1.0 / (1.0+exp(-x));
}

class Matrix {
  List<double> values;
  final int rows;
  final int columns;
  
  Matrix(int this.rows, int this.columns) {
    values = new List<double>(rows*columns);
  }
  
  double get(int r, int c) {
    return values[(r*columns) + c];
  }
  
  void set(int r, int c, double v) {
    values[(r*columns) + c] = v;
  }
}

abstract class Layer {
  static final Random rng = new Random(); 
  
  Matrix weights;
  List<double> values;
  int minNode, maxNode;
  
  Layer(int nodes, bool hasBias) {
    minNode = hasBias ? 0 : 1;
    maxNode = nodes;
    values = new List<double>(maxNode+1);
    for (int n = minNode; n <= maxNode; ++n) {
      values[n] = (n==0) ? 1.0 : 0.0;
    }
  }
}

abstract class NonInputLayer extends Layer {
  NonInputLayer(int nodes, bool hasBias) : super(nodes, hasBias);
  
  List<double> backpropagate(List<double> outputs, double rate, double momentum);
  void update(Layer prev);
}

class InputLayer extends Layer {
  NonInputLayer nextLayer;
  Matrix weights;
  Matrix weightChanges;
  
  InputLayer(int nodes, NonInputLayer this.nextLayer) : super(nodes, true){
    weights = new Matrix(maxNode + 1, nextLayer.maxNode + 1);
    weightChanges = new Matrix(maxNode + 1, nextLayer.maxNode + 1);
    
    for (int own = minNode; own <= maxNode; ++own) {
      for (int other = nextLayer.minNode; other <= nextLayer.maxNode; ++other) {
        weights.set(own, other, (4.0*Layer.rng.nextDouble())-2);
        weightChanges.set(own, other, (4.0*Layer.rng.nextDouble())-2);
      }
    }
  }
  
  void update(List<double> inputs) {
    for (int n = 0; n < inputs.length; ++n) {
      values[n+1] = inputs[n];
    }
    nextLayer.update(this);
  }
  
  void backpropagate(List<double> outputs, double rate, double momentum) {
    List<double> nextErrors = nextLayer.backpropagate(outputs, rate, momentum);
    for (int n = minNode; n <= maxNode; ++n) {
      for (int m = nextLayer.minNode; m <= nextLayer.maxNode; ++m) {
        double errorDeriv = nextErrors[m] * values[n];
        double delta = rate * errorDeriv + momentum * weightChanges.get(n, m);
        weights.set(n, m, weights.get(n, m) + delta);
        weightChanges.set(n,m, delta);
      }
    }
  }
}

class HiddenLayer extends NonInputLayer {
  NonInputLayer nextLayer;
  Matrix weights;
  Matrix weightChanges;
  
  HiddenLayer(int nodes, NonInputLayer this.nextLayer) : super(nodes, true) {
    weights = new Matrix(maxNode + 1, nextLayer.maxNode + 1);
    weightChanges = new Matrix(maxNode + 1, nextLayer.maxNode + 1);
    
    for (int own = minNode; own <= maxNode; ++own) {
      for (int other = nextLayer.minNode; other <= nextLayer.maxNode; ++other) {
        weights.set(own, other, (4.0*Layer.rng.nextDouble())-2);
        weightChanges.set(own, other, (4.0*Layer.rng.nextDouble())-2);
      }
    }
  }
  
  void update(Layer prev) {
    for (int n = 1; n <= maxNode; ++n) {
      double sum = 0.0;
      for (int m = prev.minNode; m <= prev.maxNode; ++m) {
        sum += prev.values[m] * prev.weights.get(m, n);
      }
      values[n] = logit(sum);
    }
    nextLayer.update(this);
  }
  
  List<double> backpropagate(List<double> outputs, double rate, double momentum) {
    List<double> nextErrors = nextLayer.backpropagate(outputs, rate, momentum);
    for (int n = minNode; n <= maxNode; ++n) {
      for (int m = nextLayer.minNode; m <= nextLayer.maxNode; ++m) {
        double errorDeriv = nextErrors[m] * values[n];
        double delta = rate * errorDeriv + momentum * weightChanges.get(n, m);
        weights.set(n, m, weights.get(n, m) + delta);
        weightChanges.set(n,m, delta);
      }
    }
    
    List<double> errors = new List<double>(maxNode + 1);
    for (int n = minNode; n <= maxNode; ++n) {
      double prevError = 0.0;
      for (int m = nextLayer.minNode; m <= nextLayer.maxNode; ++m) {
        prevError += nextErrors[m] * weights.get(n, m);
      }
      double corrFactor = values[n] * (1-values[n]);
      errors[n] = prevError * corrFactor;
    }
    
    return errors;
  }
}

class OutputLayer extends NonInputLayer {
  OutputLayer(int nodes) : super(nodes, false) {
  }
  
  void update(Layer prev) {
    for (int n = 1; n <= maxNode; ++n) {
      double sum = 0.0;
      for (int m = prev.minNode; m <= prev.maxNode; ++m) {
        sum += prev.values[m] * prev.weights.get(m, n);
      }
      values[n] = logit(sum);
    }
  }

  List<double> backpropagate(List<double> outputs, double rate, double momentum) {
    List<double> errors = new List<double>(maxNode + 1);
    for (int n = minNode; n <= maxNode; ++n) {
      double calc = values[n], real = outputs[n-1];
      errors[n] = ((real-calc) * calc * (1-calc));
    }
    
    return errors;
  }
  
  double getError(double real, int n) {
    double calc = values[n];
    return 0.5 * pow(real-calc, 2);
  }
}

class Example {
  List<double> inputs;
  List<double> outputs;

  Example(List<double> this.inputs, List<double> this.outputs);
}

class BasicNetwork {
  InputLayer input;
  OutputLayer output;
  int numIterations = 0;
  
  BasicNetwork(InputLayer this.input, OutputLayer this.output);
  
  void train(List<Example> trainingExamples, List<Example> testExamples, double rate, double momentum) {
    int numIterations = 0;
    while (true) {
      trainStep(trainingExamples, testExamples, rate, momentum);
      
      if (numIterations > 1000) {
        break;
      }
    }
  }
  
  void trainStep(List<Example> trainingExamples, List<Example> testExamples, double rate, double momentum) {
      ++numIterations;
      double trainingError = 0.0;
      for (Example e in trainingExamples) {
        input.update(e.inputs);
        input.backpropagate(e.outputs, rate, momentum);
        
        for (int n = 0; n < e.outputs.length; ++n) {
          trainingError += output.getError(e.outputs[n], n+1);
        }
      }
      
      double validationError = 0.0;
      for (Example e in testExamples) {
        input.update(e.inputs);
        for (int n = 0; n < e.outputs.length; ++n) {
          validationError += output.getError(e.outputs[n], n+1);
        }
      }
      
      print('trainError=$trainingError; validationEror=$validationError.');
  }
}