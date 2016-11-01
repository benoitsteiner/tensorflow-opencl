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

import {TSNE} from './bh_tsne';
import * as knn from './knn';
import * as scatterPlot from './scatterPlot';
import {shuffle, getSearchPredicate, runAsyncTask} from './util';
import * as logging from './logging';
import * as vector from './vector';
import {SpriteMetadata} from './data-provider';

export type DistanceFunction = (a: number[], b: number[]) => number;
export type PointAccessor = (index: number) => number;

export interface PointMetadata {
  [key: string]: number | string;
}

export interface DataProto {
  shape: [number, number];
  tensor: number[];
  metadata: {
    columns: Array<{
      name: string;
      stringValues: string[];
      numericValues: number[];
    }>;
  };
}

/** Statistics for a metadata column. */
export interface ColumnStats {
  name: string;
  isNumeric: boolean;
  tooManyUniqueValues: boolean;
  uniqueEntries?: Array<{label: string, count: number}>;
  min: number;
  max: number;
}

export interface MetadataInfo {
  stats: ColumnStats[];
  pointsInfo: PointMetadata[];
  spriteImage?: HTMLImageElement;
  spriteMetadata?: SpriteMetadata;
}

export interface DataPoint extends scatterPlot.DataPoint {
  /** The point in the original space. */
  vector: number[];

  /*
   * Metadata for each point. Each metadata is a set of key/value pairs
   * where the value can be a string or a number.
   */
  metadata: PointMetadata;

  /** This is where the calculated projections space are cached */
  projections: {[key: string]: number};
}

/** Checks to see if the browser supports webgl. */
function hasWebGLSupport(): boolean {
  try {
    let c = document.createElement('canvas');
    let gl = c.getContext('webgl') || c.getContext('experimental-webgl');
    return gl != null && typeof weblas !== 'undefined';
  } catch (e) {
    return false;
  }
}

const WEBGL_SUPPORT = hasWebGLSupport();
const IS_FIREFOX = navigator.userAgent.toLowerCase().indexOf('firefox') >= 0;
/** Controls whether nearest neighbors computation is done on the GPU or CPU. */
const KNN_GPU_ENABLED = WEBGL_SUPPORT && !IS_FIREFOX;

/** Sampling is used when computing expensive operations such as T-SNE. */
export const SAMPLE_SIZE = 10000;
/** Number of dimensions to sample when doing approximate PCA. */
export const PCA_SAMPLE_DIM = 200;
/** Number of pca components to compute. */
const NUM_PCA_COMPONENTS = 10;
/** Reserved metadata attribute used for trace information. */
const TRACE_METADATA_ATTR = '__next__';

/**
 * Dataset contains a DataPoints array that should be treated as immutable. This
 * acts as a working subset of the original data, with cached properties
 * from computationally expensive operations. Because creating a subset
 * requires normalizing and shifting the vector space, we make a copy of the
 * data so we can still always create new subsets based on the original data.
 */
export class DataSet {
  points: DataPoint[];
  traces: scatterPlot.DataTrace[];

  sampledDataIndices: number[] = [];

  /**
   * This keeps a list of all current projections so you can easily test to see
   * if it's been calculated already.
   */
  projections = d3.set();
  nearest: knn.NearestEntry[][];
  nearestK: number;
  tSNEIteration: number = 0;
  tSNEShouldStop = true;
  dim = [0, 0];
  hasTSNERun: boolean = false;
  metadataInfo: MetadataInfo;

  private tsne: TSNE;

  /** Creates a new Dataset */
  constructor(points: DataPoint[], metadataInfo?: MetadataInfo) {
    this.points = points;
    this.sampledDataIndices =
        shuffle(d3.range(this.points.length)).slice(0, SAMPLE_SIZE);
    this.traces = this.computeTraces(points);
    this.dim = [this.points.length, this.points[0].vector.length];
    this.metadataInfo = metadataInfo;
  }

  private computeTraces(points: DataPoint[]) {
    // Keep a list of indices seen so we don't compute traces for a given
    // point twice.
    let indicesSeen = new Int8Array(points.length);
    // Compute traces.
    let indexToTrace: {[index: number]: scatterPlot.DataTrace} = {};
    let traces: scatterPlot.DataTrace[] = [];
    for (let i = 0; i < points.length; i++) {
      if (indicesSeen[i]) {
        continue;
      }
      indicesSeen[i] = 1;

      // Ignore points without a trace attribute.
      let next = points[i].metadata[TRACE_METADATA_ATTR];
      if (next == null || next === '') {
        continue;
      }
      if (next in indexToTrace) {
        let existingTrace = indexToTrace[+next];
        // Pushing at the beginning of the array.
        existingTrace.pointIndices.unshift(i);
        indexToTrace[i] = existingTrace;
        continue;
      }
      // The current point is pointing to a new/unseen trace.
      let newTrace: scatterPlot.DataTrace = {pointIndices: []};
      indexToTrace[i] = newTrace;
      traces.push(newTrace);
      let currentIndex = i;
      while (points[currentIndex]) {
        newTrace.pointIndices.push(currentIndex);
        let next = points[currentIndex].metadata[TRACE_METADATA_ATTR];
        if (next != null && next !== '') {
          indicesSeen[+next] = 1;
          currentIndex = +next;
        } else {
          currentIndex = -1;
        }
      }
    }
    return traces;
  }

  getPointAccessors(projection: Projection, components: (number|string)[]):
      [PointAccessor, PointAccessor, PointAccessor] {
    if (components.length > 3) {
      throw new RangeError('components length must be <= 3');
    }
    const accessors: [PointAccessor, PointAccessor, PointAccessor] =
        [null, null, null];
    const prefix = (projection === 'custom') ? 'linear' : projection;
    for (let i = 0; i < components.length; ++i) {
      if (components[i] == null) {
        continue;
      }
      accessors[i] =
          (index =>
               this.points[index].projections[prefix + '-' + components[i]]);
    }
    return accessors;
  }

  hasMeaningfulVisualization(projection: Projection): boolean {
    if (projection !== 'tsne') {
      return true;
    }
    return this.tSNEIteration > 0;
  }

  /**
   * Returns a new subset dataset by copying out data. We make a copy because
   * we have to modify the vectors by normalizing them.
   *
   * @param subset Array of indices of points that we want in the subset.
   *
   * @return A subset of the original dataset.
   */
  getSubset(subset?: number[]): DataSet {
    let pointsSubset = subset && subset.length ?
        subset.map(i => this.points[i]) : this.points;
    let points = pointsSubset.map(dp => {
      return {
        metadata: dp.metadata,
        index: dp.index,
        vector: dp.vector.slice(),
        projectedPoint: [0, 0, 0] as [number, number, number],
        projections: {} as {[key: string]: number}
      };
    });
    return new DataSet(points, this.metadataInfo);
  }

  /**
   * Computes the centroid, shifts all points to that centroid,
   * then makes them all unit norm.
   */
  normalize() {
    // Compute the centroid of all data points.
    let centroid = vector.centroid(this.points, a => a.vector);
    if (centroid == null) {
      throw Error('centroid should not be null');
    }
    // Shift all points by the centroid and make them unit norm.
    for (let id = 0; id < this.points.length; ++id) {
      let dataPoint = this.points[id];
      dataPoint.vector = vector.sub(dataPoint.vector, centroid);
      vector.unit(dataPoint.vector);
    }
  }

  /** Projects the dataset onto a given vector and caches the result. */
  projectLinear(dir: vector.Vector, label: string) {
    this.projections.add(label);
    this.points.forEach(dataPoint => {
      dataPoint.projections[label] = vector.dot(dataPoint.vector, dir);
    });
  }

  /** Projects the dataset along the top 10 principal components. */
  projectPCA(): Promise<void> {
    if (this.projections.has('pca-0')) {
      return Promise.resolve<void>(null);
    }
    return runAsyncTask('Computing PCA...', () => {
      // Approximate pca vectors by sampling the dimensions.
      let dim = this.points[0].vector.length;
      let vectors = this.points.map(d => d.vector);
      if (dim > PCA_SAMPLE_DIM) {
        vectors = vector.projectRandom(vectors, PCA_SAMPLE_DIM);
      }
      let sigma = numeric.div(
          numeric.dot(numeric.transpose(vectors), vectors), vectors.length);
      let U: any;
      U = numeric.svd(sigma).U;
      let pcaVectors = vectors.map(vector => {
        let newV: number[] = [];
        for (let d = 0; d < NUM_PCA_COMPONENTS; d++) {
          let dot = 0;
          for (let i = 0; i < vector.length; i++) {
            dot += vector[i] * U[i][d];
          }
          newV.push(dot);
        }
        return newV;
      });
      for (let j = 0; j < NUM_PCA_COMPONENTS; j++) {
        let label = 'pca-' + j;
        this.projections.add(label);
        this.points.forEach((d, i) => {
          d.projections[label] = pcaVectors[i][j];
        });
      }
    });
  }

  /** Runs tsne on the data. */
  projectTSNE(
      perplexity: number, learningRate: number, tsneDim: number,
      stepCallback: (iter: number) => void) {
    this.hasTSNERun = true;
    let k = Math.floor(3 * perplexity);
    let opt = {epsilon: learningRate, perplexity: perplexity, dim: tsneDim};
    this.tsne = new TSNE(opt);
    this.tSNEShouldStop = false;
    this.tSNEIteration = 0;

    let step = () => {
      if (this.tSNEShouldStop) {
        stepCallback(null);
        this.tsne = null;
        return;
      }
      this.tsne.step();
      let result = this.tsne.getSolution();
      this.sampledDataIndices.forEach((index, i) => {
        let dataPoint = this.points[index];

        dataPoint.projections['tsne-0'] = result[i * tsneDim + 0];
        dataPoint.projections['tsne-1'] = result[i * tsneDim + 1];
        if (tsneDim === 3) {
          dataPoint.projections['tsne-2'] = result[i * tsneDim + 2];
        }
      });
      this.tSNEIteration++;
      stepCallback(this.tSNEIteration);
      requestAnimationFrame(step);
    };

    // Nearest neighbors calculations.
    let knnComputation: Promise<knn.NearestEntry[][]>;

    if (this.nearest != null && k === this.nearestK) {
      // We found the nearest neighbors before and will reuse them.
      knnComputation = Promise.resolve(this.nearest);
    } else {
      let sampledData = this.sampledDataIndices.map(i => this.points[i]);
      this.nearestK = k;
      knnComputation = KNN_GPU_ENABLED ?
          knn.findKNNGPUCosine(sampledData, k, (d => d.vector)) :
          knn.findKNN(
              sampledData, k, (d => d.vector),
              (a, b, limit) => vector.cosDistNorm(a, b));
    }
    knnComputation.then(nearest => {
      this.nearest = nearest;
      runAsyncTask('Initializing T-SNE...', () => {
        this.tsne.initDataDist(this.nearest);
      }).then(step);
    });
  }

  mergeMetadata(metadata: MetadataInfo) {
    if (metadata.pointsInfo.length !== this.points.length) {
      logging.setWarningMessage(
          `Number of tensors (${this.points.length}) do not match` +
          ` the number of lines in metadata (${metadata.pointsInfo.length}).`);
    }
    this.metadataInfo = metadata;
    metadata.pointsInfo.slice(0, this.points.length)
        .forEach((m, i) => this.points[i].metadata = m);
  }

  stopTSNE() { this.tSNEShouldStop = true; }

  /**
   * Finds the nearest neighbors of the query point using a
   * user-specified distance metric.
   */
  findNeighbors(pointIndex: number, distFunc: DistanceFunction, numNN: number):
      knn.NearestEntry[] {
    // Find the nearest neighbors of a particular point.
    let neighbors = knn.findKNNofPoint(this.points, pointIndex, numNN,
        (d => d.vector), distFunc);
    // TODO(smilkov): Figure out why we slice.
    let result = neighbors.slice(0, numNN);
    return result;
  }

  /**
   * Search the dataset based on a metadata field.
   */
  query(query: string, inRegexMode: boolean, fieldName: string): number[] {
    let predicate = getSearchPredicate(query, inRegexMode, fieldName);
    let matches: number[] = [];
    this.points.forEach((point, id) => {
      if (predicate(point)) {
        matches.push(id);
      }
    });
    return matches;
  }
}

export type Projection = 'tsne' | 'pca' | 'custom';

export interface ColorOption {
  name: string;
  desc?: string;
  map?: (value: string|number) => string;
  /** List of items for the color map. Defined only for categorical map. */
  items?: {label: string, count: number}[];
  /** Threshold values and their colors. Defined for gradient color map. */
  thresholds?: {value: number, color: string}[];
  isSeparator?: boolean;
}

/**
 * An interface that holds all the data for serializing the current state of
 * the world.
 */
export class State {
  /** A label identifying this state. */
  label: string = '';

  /** Whether this State is selected in the bookmarks pane. */
  isSelected: boolean = false;

  /** The selected projection tab. */
  selectedProjection: Projection;

  /** t-SNE parameters */
  tSNEIteration: number = 0;
  tSNEPerplexity: number = 0;
  tSNELearningRate: number = 0;
  tSNEis3d: boolean = true;

  /** PCA projection component dimensions */
  pcaComponentDimensions: number[] = [];

  /** Custom projection parameters */
  customSelectedSearchByMetadataOption: string;
  customXLeftText: string;
  customXLeftRegex: boolean;
  customXRightText: string;
  customXRightRegex: boolean;
  customYUpText: string;
  customYUpRegex: boolean;
  customYDownText: string;
  customYDownRegex: boolean;

  /** The computed projections of the tensors. */
  projections: Array<{[key: string]: number}> = [];

  /** The indices of selected points. */
  selectedPoints: number[] = [];

  /** Camera state (2d/3d, position, target, zoom, etc). */
  cameraDef: scatterPlot.CameraDef;

  /** Color by option. */
  selectedColorOptionName: string;

  /** Label by option. */
  selectedLabelOption: string;
}

export function stateGetAccessorDimensions(state: State): Array<number|string> {
  let dimensions: Array<number|string>;
  switch (state.selectedProjection) {
    case 'pca':
      dimensions = state.pcaComponentDimensions.slice();
      break;
    case 'tsne':
      dimensions = [0, 1];
      if (state.tSNEis3d) {
        dimensions.push(2);
      }
      break;
    case 'custom':
      dimensions = ['x', 'y'];
      break;
    default:
      throw new Error('Unexpected fallthrough');
  }
  return dimensions;
}
