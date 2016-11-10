/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

import {DataPoint, DataSet, State, stateGetAccessorDimensions} from './data';

/**
 * Helper method that makes a list of points given an array of
 * trace indexes.
 *
 * @param traces The i-th entry holds the 'next' attribute for the i-th point.
 */
function makePointsWithTraces(traces: number[]) {
  let nextAttr = '__next__';
  let points: DataPoint[] = [];
  traces.forEach((t, i) => {
    let metadata: {[key: string]: any} = {};
    metadata[nextAttr] = t >= 0 ? t : null;
    points.push({
      vector: new Float32Array(0),
      metadata: metadata,
      projections: {},
      index: i
    });
  });
  return points;
}

describe('constructor_with_traces', () => {
  it('Simple forward pointing traces', () => {
    // The input is: 0->2, 1->None, 2->3, 3->None. This should return
    // one trace 0->2->3.
    const points = makePointsWithTraces([2, -1, 3, -1]);
    let dataset = new DataSet(points);
    expect(dataset.traces.length).toEqual(1);
    expect(dataset.traces[0].pointIndices).toEqual([0, 2, 3]);
  });

  it('No traces', () => {
    let points = makePointsWithTraces([-1, -1, -1, -1]);
    let dataset = new DataSet(points);
    expect(dataset.traces.length).toEqual(0);
  });

  it('A trace that goes backwards and forward in the array', () => {
    // The input is: 0->2, 1->0, 2->nothing, 3->1. This should return
    // one trace 3->1->0->2.
    let points = makePointsWithTraces([2, 0, -1, 1]);
    let dataset = new DataSet(points);
    expect(dataset.traces.length).toEqual(1);
    expect(dataset.traces[0].pointIndices).toEqual([3, 1, 0, 2]);
  });
});

describe('stateGetAccessorDimensions', () => {
  it('returns [0, 1] for 2d t-SNE', () => {
    const state = new State();
    state.selectedProjection = 'tsne';
    state.tSNEis3d = false;
    expect(stateGetAccessorDimensions(state)).toEqual([0, 1]);
  });

  it('returns [0, 1, 2] for 3d t-SNE', () => {
    const state = new State();
    state.selectedProjection = 'tsne';
    state.tSNEis3d = true;
    expect(stateGetAccessorDimensions(state)).toEqual([0, 1, 2]);
  });

  it('returns pca component dimensions array for pca', () => {
    const state = new State();
    state.selectedProjection = 'pca';
    state.pcaComponentDimensions = [13, 12, 11, 10];
    expect(stateGetAccessorDimensions(state))
        .toEqual(state.pcaComponentDimensions);
  });

  it('returns ["x", "y"] for custom projections', () => {
    const state = new State();
    state.selectedProjection = 'custom';
    expect(stateGetAccessorDimensions(state)).toEqual(['x', 'y']);
  });
});
