<<<<<<< HEAD
/// <reference path="../../../typings/tsd.d.ts" />
/// <reference path="common.ts" />
=======
/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
>>>>>>> tensorflow/master
module tf.graph.parser {

/**
 * Parses a native js value, which can be either a string, boolean or number.
 *
 * @param value The value to be parsed.
 */
function parseValue(value: string): string|number|boolean {
  if (value === "true") {
    return true;
  }
  if (value === "false") {
    return false;
  }
  let firstChar = value[0];
  if (firstChar === "\"") {
    return value.substring(1, value.length - 1);
  }
  let num = parseFloat(value);
  return isNaN(num) ? value : num;
}

/**
 * Fetches a text file and returns a promise of the result.
 */
export function readPbTxt(filepath: string): Promise<string> {
  return new Promise<string>(function(resolve, reject) {
    d3.text(filepath, function(error, text) {
      if (error) {
        reject(error);
        return;
      }
      resolve(text);
    });
  });
}

/**
 * Fetches and parses a json file and returns a promise of the result.
 */
export function readJson(filepath: string): Promise<Object> {
  return new Promise<Object>(function(resolve, reject) {
    d3.json(filepath, function(error, text) {
      if (error) {
        reject(error);
        return;
      }
      resolve(text);
    });
  });
}

/**
 * Reads the graph and stats file (if available), parses them and returns a
 * promise of the result.
 */
export function readAndParseData(dataset: {path: string, statsPath: string},
<<<<<<< HEAD
    pbTxtContent: string, tracker: ProgressTracker):
    Promise<{ nodes: TFNode[], statsJson: Object }|void> {
  let graphPbTxt;
  let statsJson;
  return runAsyncTask("Reading graph.pbtxt", 20, () => {
    return pbTxtContent || readPbTxt(dataset.path);
  }, tracker)
  .then(function(text) {
    graphPbTxt = text;
    return runAsyncTask("Reading stats.pbtxt", 20, () => {
=======
    pbTxtFile: Blob, tracker: ProgressTracker):
    Promise<{ nodes: TFNode[], statsJson: Object }|void> {
  let graphPbTxt: Blob;
  let statsJson: Object;
  return runTask("Reading graph.pbtxt", 20, () => {
    return pbTxtFile ?
      Promise.resolve(pbTxtFile) :
      readPbTxt(dataset.path).then(text => new Blob([text]));
  }, tracker)
  .then(blob => {
    graphPbTxt = blob;
    return runTask("Reading stats.pbtxt", 20, () => {
>>>>>>> tensorflow/master
      return (dataset != null && dataset.statsPath != null) ?
          readJson(dataset.statsPath) : null;
    }, tracker);
  })
<<<<<<< HEAD
  .then(function(json) {
    statsJson = json;
    return runAsyncTask("Parsing graph.pbtxt", 60, () => {
      return parsePbtxt(graphPbTxt);
    }, tracker);
  })
  .then(function(nodes) {
=======
  .then(json => {
    statsJson = json;
    return runTask("Parsing graph.pbtxt", 60, () => {
      return parsePbtxtFile(graphPbTxt);
    }, tracker);
  })
  .then(nodes => {
>>>>>>> tensorflow/master
    return {
      nodes: nodes,
      statsJson: statsJson
    };
<<<<<<< HEAD
    })
  .catch(function(reason) {
    throw new Error("Failure parsing graph definition");
=======
>>>>>>> tensorflow/master
  });
}

/**
<<<<<<< HEAD
 * Parses a proto txt file into a javascript object.
 *
 * @param input The string contents of the proto txt file.
 * @return The parsed object.
 */
export function parsePbtxt(input: string): TFNode[] {
=======
 * Parse a file object in a streaming fashion line by line (or custom delim).
 * Can handle very large files.
 * @param input The file object
 * @param callback The callback called on each line
 * @param chunkSize The size of each read chunk. (optional)
 * @param delim The delimiter used to split a line. (optional)
 * @returns A promise for when it is finished.
 */
export function streamParse(file: Blob, callback: (string) => void,
    chunkSize: number = 1000000, delim: string = "\n"): Promise<boolean> {
  return new Promise<boolean>(function(resolve, reject) {
    let offset = 0;
    let fileSize = file.size - 1;
    let data = "";

    function readHandler(evt) {
      if (evt.target.error == null) {
        offset += evt.target.result.length;
        let str = evt.target.result;
        let parts = str.split(delim);
        let first = data + parts[0];
        if (parts.length === 1) {
          data = first;
          readChunk(offset, chunkSize);
          return;
        }
        data = parts[parts.length - 1];
        callback(first);
        for (let i = 1; i < parts.length - 1; i++) {
          callback(parts[i]);
        }
      } else {
        // read error
        reject(evt.target.error);
        return;
      }
      if (offset >= fileSize) {
        if (data) {
          callback(data);
        }
        resolve(true);
        return;
      }
      readChunk(offset, chunkSize);
    }

    function readChunk(offset: number, size: number) {
      var reader = new FileReader();
      var blob = file.slice(offset, offset + size);
      reader.onload = readHandler;
      reader.readAsText(blob);
    }

    readChunk(offset, chunkSize);
  });
}

/**
 * Parses a proto txt file or blob into javascript object.
 *
 * @param input The Blob or file object implementing slice.
 * @returns The parsed object.
 */
export function parsePbtxtFile(input: Blob): Promise<TFNode[]> {
>>>>>>> tensorflow/master
  let output: { [name: string]: any; } = { node: [] };
  let stack = [];
  let path: string[] = [];
  let current: { [name: string]: any; } = output;

  function splitNameAndValueInAttribute(line: string) {
    let colonIndex = line.indexOf(":");
    let name = line.substring(0, colonIndex).trim();
    let value = parseValue(line.substring(colonIndex + 2).trim());
    return {
      name: name,
      value: value
    };
  }

  /**
   * Since proto-txt doesn't explicitly say whether an attribute is repeated
   * (an array) or not, we keep a hard-coded list of attributes that are known
   * to be repeated. This list is used in parsing time to convert repeated
   * attributes into arrays even when the attribute only shows up once in the
   * object.
   */
<<<<<<< HEAD
  let ARRAY_ATTRIBUTES: {[attrPath: string] : boolean} = {
=======
  let ARRAY_ATTRIBUTES: {[attrPath: string]: boolean} = {
>>>>>>> tensorflow/master
    "node": true,
    "node.input": true,
    "node.attr": true,
    "node.attr.value.list.type": true,
    "node.attr.value.shape.dim": true,
    "node.attr.value.tensor.string_val": true,
<<<<<<< HEAD
    "node.attr.value.tensor.tensor_shape.dim": true
=======
    "node.attr.value.tensor.tensor_shape.dim": true,
    "node.attr.value.list.shape": true,
    "node.attr.value.list.shape.dim": true,
    "node.attr.value.list.s": true
>>>>>>> tensorflow/master
  };

  /**
   * Adds a value, given the attribute name and the host object. If the
   * attribute already exists, but is not an array, it will convert it to an
   * array of values.
   *
   * @param obj The host object that holds the attribute.
   * @param name The attribute name (key).
   * @param value The attribute value.
   * @param path A path that identifies the attribute. Used to check if
   *     an attribute is an array or not.
   */
  function addAttribute(obj: Object, name: string,
      value: Object|string|number|boolean, path: string[]): void {
    // We treat "node" specially since it is done so often.
    let existingValue = obj[name];
    if (existingValue == null) {
      obj[name] = path.join(".") in ARRAY_ATTRIBUTES ? [value] : value;
    } else if (Array.isArray(existingValue)) {
      existingValue.push(value);
    } else {
      obj[name] = [existingValue, value];
    }
  }

  // Run through the file a line at a time.
<<<<<<< HEAD
  let startPos = 0;
  while (startPos < input.length) {
    let endPos = input.indexOf("\n", startPos);
    if (endPos === -1) {
      endPos = input.length;
    }
    let line = input.substring(startPos, endPos);
    startPos = endPos + 1;
    if (!line) {
      continue;
=======
  return streamParse(input, function(line: string) {
    if (!line) {
      return;
>>>>>>> tensorflow/master
    }
    switch (line[line.length - 1]) {
      case "{": // create new object
        let name = line.substring(0, line.length - 2).trim();
        let newValue: { [name: string]: any; } = {};
        stack.push(current);
        path.push(name);
        addAttribute(current, name, newValue, path);
        current = newValue;
        break;
      case "}":
        current = stack.pop();
        path.pop();
        break;
      default:
        let x = splitNameAndValueInAttribute(line);
        addAttribute(current, x.name, x.value, path.concat(x.name));
        break;
    }
<<<<<<< HEAD
  }

  return output["node"];
=======
  }).then(function() {
    return output["node"];
  });
}

/**
 * Parses a proto txt file into a javascript object.
 *
 * @param input The string contents of the proto txt file.
 * @return The parsed object.
 */
export function parsePbtxt(input: string): Promise<TFNode[]> {
  let blob = new Blob([input]);
  return parsePbtxtFile(blob);
>>>>>>> tensorflow/master
}

} // Close module tf.graph.parser.
