<<<<<<< HEAD
/// <reference path="../graph.ts" />
/// <reference path="../render.ts" />
/// <reference path="scene.ts" />
/// <reference path="edge.ts" />

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
module tf.graph.scene.annotation {

/**
 * Populate a given annotation container group
 *
 *     <g class="{in|out}-annotations"></g>
 *
 * with annotation group of the following structure:
 *
 * <g class="annotation">
 *   <g class="annotation-node">
 *   <!--
 *   Content here determined by Scene.node.buildGroup.
 *   -->
 *   </g>
 * </g>
 *
 * @param container selection of the container.
 * @param annotationData node.{in|out}Annotations
 * @param d node to build group for.
<<<<<<< HEAD
 * @param sceneBehavior polymer scene element.
 * @return selection of appended objects
 */
export function buildGroup(container, annotationData: render.AnnotationList,
  d: render.RenderNodeInformation, sceneBehavior) {
=======
 * @param sceneElement <tf-graph-scene> polymer element.
 * @return selection of appended objects
 */
export function buildGroup(container, annotationData: render.AnnotationList,
  d: render.RenderNodeInfo, sceneElement) {
>>>>>>> tensorflow/master
  // Select all children and join with data.
  let annotationGroups = container.selectAll(function() {
       // using d3's selector function
       // See https://github.com/mbostock/d3/releases/tag/v2.0.0
       // (It's not listed in the d3 wiki.)
         return this.childNodes;
       })
       .data(annotationData.list, d => { return d.node.name; });

  annotationGroups.enter()
    .append("g")
    .attr("data-name", a => { return a.node.name; })
    .each(function(a) {
      let aGroup = d3.select(this);

      // Add annotation to the index in the scene
<<<<<<< HEAD
      sceneBehavior.addAnnotationGroup(a, d, aGroup);
=======
      sceneElement.addAnnotationGroup(a, d, aGroup);
>>>>>>> tensorflow/master
      // Append annotation edge
      let edgeType = Class.Annotation.EDGE;
      let metaedge = a.renderMetaedgeInfo && a.renderMetaedgeInfo.metaedge;
      if (metaedge && !metaedge.numRegularEdges) {
        edgeType += " " + Class.Annotation.CONTROL_EDGE;
      }
      // If any edges are reference edges, add the reference edge class.
      if (metaedge && metaedge.numRefEdges) {
        edgeType += " " + Class.Edge.REF_LINE;
      }
<<<<<<< HEAD
      edge.appendEdge(aGroup, a, sceneBehavior, edgeType);

      if (a.annotationType !== tf.graph.render.AnnotationType.ELLIPSIS) {
        addAnnotationLabelFromNode(aGroup, a);
        buildShape(aGroup, a, sceneBehavior);
=======
      edge.appendEdge(aGroup, a, sceneElement, edgeType);

      if (a.annotationType !== render.AnnotationType.ELLIPSIS) {
        addAnnotationLabelFromNode(aGroup, a);
        buildShape(aGroup, a);
>>>>>>> tensorflow/master
      } else {
        addAnnotationLabel(aGroup, a.node.name, a, Class.Annotation.ELLIPSIS);
      }
    });

  annotationGroups
    .attr("class", a => {
      return Class.Annotation.GROUP + " " +
        annotationToClassName(a.annotationType) +
        " " + node.nodeClass(a);
    })
    .each(function(a) {
      let aGroup = d3.select(this);
<<<<<<< HEAD
      update(aGroup, d, a, sceneBehavior);
      if (a.annotationType !== tf.graph.render.AnnotationType.ELLIPSIS) {
        addInteraction(aGroup, d, sceneBehavior);
=======
      update(aGroup, d, a, sceneElement);
      if (a.annotationType !== render.AnnotationType.ELLIPSIS) {
        addInteraction(aGroup, d, a, sceneElement);
>>>>>>> tensorflow/master
      }
    });

  annotationGroups.exit()
    .each(function(a) {
      let aGroup = d3.select(this);

      // Remove annotation from the index in the scene
<<<<<<< HEAD
      sceneBehavior.removeAnnotationGroup(a, d, aGroup);
=======
      sceneElement.removeAnnotationGroup(a, d, aGroup);
>>>>>>> tensorflow/master
    })
    .remove();
  return annotationGroups;
};

/**
 * Maps an annotation enum to a class name used in css rules.
 */
function annotationToClassName(annotationType: render.AnnotationType) {
<<<<<<< HEAD
  return (tf.graph.render.AnnotationType[annotationType] || "")
      .toLowerCase() || null;
}

function buildShape(aGroup, a: render.Annotation, sceneBehavior) {
  if (a.annotationType === tf.graph.render.AnnotationType.SUMMARY) {
    let image = scene.selectOrCreateChild(aGroup, "image");
    image.attr({
      "xlink:href": sceneBehavior.resolveUrl("../../lib/svg/summary-icon.svg"),
      "height": "12px",
      "width": "12px",
=======
  return (render.AnnotationType[annotationType] || "")
      .toLowerCase() || null;
}

function buildShape(aGroup, a: render.Annotation) {
  if (a.annotationType === render.AnnotationType.SUMMARY) {
    let summary = selectOrCreateChild(aGroup, "use");
    summary.attr({
      "class": "summary",
      "xlink:href": "#summary-icon",
>>>>>>> tensorflow/master
      "cursor": "pointer"
    });
  } else {
    let shape = node.buildShape(aGroup, a, Class.Annotation.NODE);
    // add title tag to get native tooltips
<<<<<<< HEAD
    scene.selectOrCreateChild(shape, "title").text(a.node.name);
=======
    selectOrCreateChild(shape, "title").text(a.node.name);
>>>>>>> tensorflow/master
  }
}

function addAnnotationLabelFromNode(aGroup, a: render.Annotation) {
  let namePath = a.node.name.split("/");
  let text = namePath[namePath.length - 1];
  let shortenedText = text.length > 8 ? text.substring(0, 8) + "..." : text;
  return addAnnotationLabel(aGroup, shortenedText, a, null, text);
}

function addAnnotationLabel(aGroup, label, a, additionalClassNames,
    fullLabel?) {
  let classNames = Class.Annotation.LABEL;
  if (additionalClassNames) {
    classNames += " " + additionalClassNames;
  }
  let titleText = fullLabel ? fullLabel : label;
  return aGroup.append("text")
                .attr("class", classNames)
                .attr("dy", ".35em")
                .attr("text-anchor", a.isIn ? "end" : "start")
                .text(label)
                .append("title").text(titleText);
}

<<<<<<< HEAD
function addInteraction(selection, d: render.RenderNodeInformation,
    sceneBehavior) {
  selection
    .on("mouseover", a => {
      sceneBehavior.fire("annotation-highlight", {
=======
function addInteraction(selection, d: render.RenderNodeInfo,
    annotation: render.Annotation, sceneElement) {
  selection
    .on("mouseover", a => {
      sceneElement.fire("annotation-highlight", {
>>>>>>> tensorflow/master
        name: a.node.name,
        hostName: d.node.name
      });
    })
    .on("mouseout", a => {
<<<<<<< HEAD
      sceneBehavior.fire("annotation-unhighlight", {
=======
      sceneElement.fire("annotation-unhighlight", {
>>>>>>> tensorflow/master
        name: a.node.name,
        hostName: d.node.name
      });
    })
    .on("click", a => {
      // Stop this event"s propagation so that it isn't also considered a
      // graph-select.
      (<Event>d3.event).stopPropagation();
<<<<<<< HEAD
      sceneBehavior.fire("annotation-select", {
=======
      sceneElement.fire("annotation-select", {
>>>>>>> tensorflow/master
        name: a.node.name,
        hostName: d.node.name
      });
    });
<<<<<<< HEAD
=======
  if (annotation.annotationType !== render.AnnotationType.SUMMARY &&
      annotation.annotationType !== render.AnnotationType.CONSTANT) {
    selection.on("contextmenu", contextmenu.getMenu(
      node.getContextMenu(annotation.node, sceneElement)));
  }
>>>>>>> tensorflow/master
};

/**
 * Adjust annotation's position.
 *
 * @param aGroup selection of a "g.annotation" element.
 * @param d Host node data.
 * @param a annotation node data.
<<<<<<< HEAD
 * @param scene Polymer scene element.
 */
function update(aGroup, d: render.RenderNodeInformation, a: render.Annotation,
    sceneBehavior) {
=======
 * @param scene <tf-graph-scene> polymer element.
 */
function update(aGroup, d: render.RenderNodeInfo, a: render.Annotation,
    sceneElement) {
  let cx = layout.computeCXPositionOfNodeShape(d);
>>>>>>> tensorflow/master
  // Annotations that point to embedded nodes (constants,summary)
  // don't have a render information attached so we don't stylize these.
  // Also we don't stylize ellipsis annotations (the string "... and X more").
  if (a.renderNodeInfo &&
<<<<<<< HEAD
      a.annotationType !== tf.graph.render.AnnotationType.ELLIPSIS) {
    node.stylize(aGroup, a.renderNodeInfo, sceneBehavior,
      Class.Annotation.NODE);
  }

  if (a.annotationType === tf.graph.render.AnnotationType.SUMMARY) {
=======
      a.annotationType !== render.AnnotationType.ELLIPSIS) {
    node.stylize(aGroup, a.renderNodeInfo, sceneElement,
      Class.Annotation.NODE);
  }

  if (a.annotationType === render.AnnotationType.SUMMARY) {
>>>>>>> tensorflow/master
    // Update the width of the annotation to give space for the image.
    a.width += 10;
  }

  // label position
  aGroup.select("text." + Class.Annotation.LABEL).transition().attr({
<<<<<<< HEAD
    x: d.x + a.dx + (a.isIn ? -1 : 1) * (a.width / 2 + a.labelOffset),
=======
    x: cx + a.dx + (a.isIn ? -1 : 1) * (a.width / 2 + a.labelOffset),
>>>>>>> tensorflow/master
    y: d.y + a.dy
  });

  // Some annotations (such as summary) are represented using a 12x12 image tag.
<<<<<<< HEAD
  // Purposely ommited units (e.g. pixels) since the images are vector graphics.
  // If there is an image, we adjust the location of the image to be vertically
  // centered with the node and horizontally centered between the arrow and the
  // text label.
  aGroup.select("image").transition().attr({
    x: d.x + a.dx - 3,
=======
  // Purposely omitted units (e.g. pixels) since the images are vector graphics.
  // If there is an image, we adjust the location of the image to be vertically
  // centered with the node and horizontally centered between the arrow and the
  // text label.
  aGroup.select("use.summary").transition().attr({
    x: cx + a.dx - 3,
>>>>>>> tensorflow/master
    y: d.y + a.dy - 6
  });

  // Node position (only one of the shape selection will be non-empty.)
<<<<<<< HEAD
  scene.positionEllipse(aGroup.select("." + Class.Annotation.NODE + " ellipse"),
                        d.x + a.dx, d.y + a.dy, a.width, a.height);
  scene.positionRect(aGroup.select("." + Class.Annotation.NODE + " rect"),
                     d.x + a.dx, d.y + a.dy, a.width, a.height);
  scene.positionRect(aGroup.select("." + Class.Annotation.NODE + " use"),
                     d.x + a.dx, d.y + a.dy, a.width, a.height);
=======
  positionEllipse(aGroup.select("." + Class.Annotation.NODE + " ellipse"),
                        cx + a.dx, d.y + a.dy, a.width, a.height);
  positionRect(aGroup.select("." + Class.Annotation.NODE + " rect"),
                     cx + a.dx, d.y + a.dy, a.width, a.height);
  positionRect(aGroup.select("." + Class.Annotation.NODE + " use"),
                     cx + a.dx, d.y + a.dy, a.width, a.height);
>>>>>>> tensorflow/master

  // Edge position
  aGroup.select("path." + Class.Annotation.EDGE).transition().attr("d", a => {
        // map relative position to absolute position
        let points = a.points.map(p => {
<<<<<<< HEAD
          return {x: p.dx + d.x, y: p.dy + d.y};
=======
          return {x: p.dx + cx, y: p.dy + d.y};
>>>>>>> tensorflow/master
        });
        return edge.interpolate(points);
      });
};

} // close module
