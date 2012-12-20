library xordemo;

import 'dart:html';
import 'dart:math';

import 'backprop.dart';

class ViewerRow {
  List inputCells = [];
  List truthCells = [];
  List outputCells = [];
  var row;
  
  ViewerRow(int inputs, int outputs) {
    row = new TableRowElement();
    
    for (int i = 0; i < inputs; ++i) {
      var c = new TableCellElement();
      inputCells.add(c);
      row.elements.add(c);
    }
    row.nodes.add(new TableCellElement());    
    for (int i = 0; i < outputs; ++i) {
      var c = new TableCellElement();
      truthCells.add(c);
      row.elements.add(c);
    }
    
    for (int i = 0; i < outputs; ++i) {
      var c = new TableCellElement();
      outputCells.add(c);
      row.elements.add(c);
    }
  }
}

List<Example> examples = <Example>[new Example([0.0,0.0], [0.0]),
                                   new Example([0.0,1.0], [1.0]),
                                   new Example([1.0,0.0], [1.0]),
                                   new Example([1.0,1.0], [0.0])];
List<ViewerRow> viewerRows = new List<ViewerRow>(examples.length);


double clamp(double min, double x, double max) {
  if (x < min) {
    return min;
  } else if (x > max) {
    return  max;
  } else {
    return x;
  }
}

void colorCell(Element e, double score) {
  int c = (clamp(0.0, 1.0-score, 1.0) * 255).toInt();
  e.style.backgroundColor = 'rgb($c,$c,$c)';
}

void colorCells(List<Element> el, List<double> scores) {
  for (int i = 0; i < el.length; ++i) {
    colorCell(el[i], scores[i]);
  }
}

void main() {
  Element viewer = query('#viewer');
  ViewerRow header = new ViewerRow(2, 1);
  for (int i = 0; i < header.inputCells.length; ++i) {
    header.inputCells[i].text = 'I$i';
  }
  for (int i = 0; i < header.truthCells.length; ++i) {
    header.truthCells[i].text = 'T$i';
  }
  for (int i = 0; i < header.outputCells.length; ++i) {
    header.outputCells[i].text = 'O$i';
  }
  viewer.elements.add(header.row);
  
  for (int ei = 0; ei < examples.length; ++ei) {
    Example e = examples[ei];
    ViewerRow r = new ViewerRow(2, 1);
    colorCells(r.inputCells, e.inputs);
    colorCells(r.truthCells, e.outputs);
    viewer.elements.add(r.row);
    viewerRows[ei] = r;
  }
  
  query("#text")
    ..text = "Click to start training"
    ..on.click.add(nnTest);
}

BasicNetwork network;

void nnTest(Event event) {
  OutputLayer output = new OutputLayer(1);
  HiddenLayer hidden = new HiddenLayer(3, output);
  // HiddenLayer hidden2 = new HiddenLayer(3, hidden);
  InputLayer input = new InputLayer(2, hidden);
  network = new BasicNetwork(input, output);
  
  window.requestAnimationFrame(tick);
}

void tick(num time) {
  network.trainStep(examples, [], 0.5, 0.1);
  for (int i = 0; i < examples.length; ++i) {
    Example e = examples[i];
    network.input.update(e.inputs);
    colorCells(viewerRows[i].outputCells, network.output.values.getRange(1, e.outputs.length));
  }
  window.requestAnimationFrame(tick);
}