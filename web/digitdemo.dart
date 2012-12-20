library digitdemo;

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

List<String> digits = <String>[
"""
 *** 
*   *
*   *
*   *
 *** 
""",
"""
  *  
 **  
  *  
  *  
 *** 
""",
"""
 *** 
    *
 *** 
*    
 ****
""",
"""
**** 
    *
 *** 
    *
**** 
""",
"""
*    
* *  
*****
  *  
  *  
""",
"""
*****
*    
**** 
    *
**** 
""",
"""
  *  
 *   
 *** 
*   *
 *** 
""",
"""
*****
    *
   * 
  *  
  *  
""",
"""
 *** 
*   *
 *** 
*   *
 *** 
""",
"""
 *** 
*   *
 *** 
  *  
 *   
"""];
 
List<Example> examples = <Example>[];
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
  int charSpace = ' '.charCodes[0], charStar = '*'.charCodes[0];
  for (int d = 0; d < digits.length; ++d) {
    List<double> inputs = <double>[];
    for (int c in digits[d].charCodes) {
      if (c == charSpace) {
        inputs.add(0.0);
      } else if (c == charStar) {
        inputs.add(1.0);
      }
    }
    List<double> outputs = new List<double>(digits.length);
    for (int o = 0; o < outputs.length; ++o) {
        outputs[o] = o == d ? 1.0 : 0.0;
    }
    examples.add(new Example(inputs, outputs));
  }
  
  Element viewer = query('#viewer');
  ViewerRow header = new ViewerRow(25, 10);
  
  /*
  for (int i = 0; i < header.inputCells.length; ++i) {
    header.inputCells[i].text = 'I$i';
  }
  for (int i = 0; i < header.truthCells.length; ++i) {
    header.truthCells[i].text = 'T$i';
  }
  for (int i = 0; i < header.outputCells.length; ++i) {
    header.outputCells[i].text = 'O$i';
  }
  */
  
  viewer.elements.add(header.row);
  
  for (int ei = 0; ei < examples.length; ++ei) {
    Example e = examples[ei];
    ViewerRow r = new ViewerRow(25, 10);
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
  OutputLayer output = new OutputLayer(10);
  HiddenLayer hidden = new HiddenLayer(20, output);
  HiddenLayer hidden2 = new HiddenLayer(40, hidden);
  InputLayer input = new InputLayer(25, hidden);
  network = new BasicNetwork(input, output);
  
  window.requestAnimationFrame(tick);
}

void tick(num time) {
  network.trainStep(examples, [], 0.02, 0.05);
  for (int i = 0; i < examples.length; ++i) {
    Example e = examples[i];
    network.input.update(e.inputs);
    colorCells(viewerRows[i].outputCells, network.output.values.getRange(1, e.outputs.length));
  }
  window.requestAnimationFrame(tick);
}