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

import {DataSet, SpriteAndMetadataInfo, PCA_SAMPLE_DIM, Projection, SAMPLE_SIZE, State} from './data';
import * as vector from './vector';
import {Projector} from './vz-projector';
import {ProjectorInput} from './vz-projector-input';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

// tslint:disable-next-line
export let ProjectionsPanelPolymer = PolymerElement({
  is: 'vz-projector-projections-panel',
  properties: {
    pcaIs3d:
        {type: Boolean, value: true, observer: '_pcaDimensionToggleObserver'},
    tSNEis3d:
        {type: Boolean, value: true, observer: '_tsneDimensionToggleObserver'},
    // PCA projection.
    pcaComponents: {type: Array, value: d3.range(0, 10)},
    pcaX: {type: Number, value: 0, observer: 'showPCAIfEnabled'},
    pcaY: {type: Number, value: 1, observer: 'showPCAIfEnabled'},
    pcaZ: {type: Number, value: 2, observer: 'showPCAIfEnabled'},
    // Custom projection.
    customSelectedSearchByMetadataOption: {
      type: String,
      observer: '_customSelectedSearchByMetadataOptionChanged'
    },
  }
});

type InputControlName = 'xLeft' | 'xRight' | 'yUp' | 'yDown';

type CentroidResult = {
  centroid?: number[]; numMatches?: number;
};

type Centroids = {
  [key: string]: number[]; xLeft: number[]; xRight: number[]; yUp: number[];
  yDown: number[];
};

/**
 * A polymer component which handles the projection tabs in the projector.
 */
export class ProjectionsPanel extends ProjectionsPanelPolymer {
  private projector: Projector;
  private currentProjection: Projection;
  private polymerChangesTriggerReprojection: boolean;
  private dataSet: DataSet;
  private originalDataSet: DataSet;
  private dim: number;

  /** T-SNE perplexity. Roughly how many neighbors each point influences. */
  private perplexity: number;
  /** T-SNE learning rate. */
  private learningRate: number;

  private searchByMetadataOptions: string[];

  /** Centroids for custom projections. */
  private centroidValues: any;
  private centroids: Centroids;
  /** The centroid across all points. */
  private allCentroid: number[];

  /** Polymer properties. */
  // TODO(nsthorat): Move these to a separate view controller.
  public tSNEis3d: boolean;
  public pcaIs3d: boolean;
  public pcaX: number;
  public pcaY: number;
  public pcaZ: number;
  public customSelectedSearchByMetadataOption: string;

  /** Polymer elements. */
  private dom: d3.Selection<any>;
  private runTsneButton: d3.Selection<HTMLButtonElement>;
  private stopTsneButton: d3.Selection<HTMLButtonElement>;
  private perplexitySlider: HTMLInputElement;
  private learningRateInput: HTMLInputElement;
  private zDropdown: d3.Selection<HTMLElement>;
  private iterationLabel: d3.Selection<HTMLElement>;

  private customProjectionXLeftInput: ProjectorInput;
  private customProjectionXRightInput: ProjectorInput;
  private customProjectionYUpInput: ProjectorInput;
  private customProjectionYDownInput: ProjectorInput;

  initialize(projector: Projector) {
    this.polymerChangesTriggerReprojection = true;
    this.projector = projector;

    // Set up TSNE projections.
    this.perplexity = 30;
    this.learningRate = 10;

    // Setup Custom projections.
    this.centroidValues = {xLeft: null, xRight: null, yUp: null, yDown: null};
    this.clearCentroids();

    this.setupUIControls();
  }

  ready() {
    this.dom = d3.select(this);
    this.zDropdown = this.dom.select('#z-dropdown');
    this.runTsneButton = this.dom.select('.run-tsne');
    this.stopTsneButton = this.dom.select('.stop-tsne');
    this.perplexitySlider = this.$$('#perplexity-slider') as HTMLInputElement;
    this.learningRateInput =
        this.$$('#learning-rate-slider') as HTMLInputElement;
    this.iterationLabel = this.dom.select('.run-tsne-iter');
  }

  disablePolymerChangesTriggerReprojection() {
    this.polymerChangesTriggerReprojection = false;
  }

  enablePolymerChangesTriggerReprojection() {
    this.polymerChangesTriggerReprojection = true;
  }

  private updateTSNEPerplexityFromUIChange() {
    if (this.perplexitySlider) {
      this.perplexity = +this.perplexitySlider.value;
    }
    this.dom.select('.tsne-perplexity span').text(this.perplexity);
  }

  private updateTSNELearningRateFromUIChange() {
    if (this.learningRateInput) {
      this.learningRate = Math.pow(10, +this.learningRateInput.value);
    }
    this.dom.select('.tsne-learning-rate span').text(this.learningRate);
  }

  private setupUIControls() {
    {
      const self = this;
      this.dom.selectAll('.ink-tab').on('click', function() {
        let id = this.getAttribute('data-tab');
        self.showTab(id);
      });
    }

    this.runTsneButton.on('click', () => this.runTSNE());
    this.stopTsneButton.on('click', () => this.dataSet.stopTSNE());

    this.perplexitySlider.value = this.perplexity.toString();
    this.perplexitySlider.addEventListener(
        'change', () => this.updateTSNEPerplexityFromUIChange());
    this.updateTSNEPerplexityFromUIChange();

    this.learningRateInput.addEventListener(
        'change', () => this.updateTSNELearningRateFromUIChange());
    this.updateTSNELearningRateFromUIChange();

    this.setupCustomProjectionInputFields();
    // TODO: figure out why `--paper-input-container-input` css mixin didn't
    // work.
    this.dom.selectAll('paper-dropdown-menu paper-input input')
      .style('font-size', '14px');
  }

  restoreUIFromBookmark(bookmark: State) {
    this.disablePolymerChangesTriggerReprojection();

    // PCA
    this.pcaX = bookmark.pcaComponentDimensions[0];
    this.pcaY = bookmark.pcaComponentDimensions[1];
    if (bookmark.pcaComponentDimensions.length === 3) {
      this.pcaZ = bookmark.pcaComponentDimensions[2];
    }
    this.pcaIs3d = (bookmark.pcaComponentDimensions.length === 3);

    // t-SNE
    if (this.perplexitySlider) {
      this.perplexitySlider.value = bookmark.tSNEPerplexity.toString();
    }
    if (this.learningRateInput) {
      this.learningRateInput.value = bookmark.tSNELearningRate.toString();
    }
    this.tSNEis3d = bookmark.tSNEis3d;

    // custom
    this.customSelectedSearchByMetadataOption =
        bookmark.customSelectedSearchByMetadataOption;
    if (this.customProjectionXLeftInput) {
      this.customProjectionXLeftInput.set(
          bookmark.customXLeftText, bookmark.customXLeftRegex);
    }
    if (this.customProjectionXRightInput) {
      this.customProjectionXRightInput.set(
          bookmark.customXRightText, bookmark.customXRightRegex);
    }
    if (this.customProjectionYUpInput) {
      this.customProjectionYUpInput.set(
          bookmark.customYUpText, bookmark.customYUpRegex);
    }
    if (this.customProjectionYDownInput) {
      this.customProjectionYDownInput.set(
          bookmark.customYDownText, bookmark.customYDownRegex);
    }
    this.computeAllCentroids();

    this.setZDropdownEnabled(this.pcaIs3d);
    this.updateTSNEPerplexityFromUIChange();
    this.updateTSNELearningRateFromUIChange();
    if (this.iterationLabel) {
      this.iterationLabel.text(bookmark.tSNEIteration.toString());
    }
    this.showTab(bookmark.selectedProjection);
    this.enablePolymerChangesTriggerReprojection();
  }

  populateBookmarkFromUI(bookmark: State) {
    this.disablePolymerChangesTriggerReprojection();

    // PCA
    bookmark.pcaComponentDimensions = [this.pcaX, this.pcaY];
    if (this.pcaIs3d) {
      bookmark.pcaComponentDimensions.push(this.pcaZ);
    }

    // t-SNE
    if (this.perplexitySlider != null) {
      bookmark.tSNEPerplexity = +this.perplexitySlider.value;
    }
    if (this.learningRateInput != null) {
      bookmark.tSNELearningRate = +this.learningRateInput.value;
    }
    bookmark.tSNEis3d = this.tSNEis3d;

    // custom
    bookmark.customSelectedSearchByMetadataOption =
        this.customSelectedSearchByMetadataOption;
    if (this.customProjectionXLeftInput != null) {
      bookmark.customXLeftText = this.customProjectionXLeftInput.getValue();
      bookmark.customXLeftRegex =
          this.customProjectionXLeftInput.getInRegexMode();
    }
    if (this.customProjectionXRightInput != null) {
      bookmark.customXRightText = this.customProjectionXRightInput.getValue();
      bookmark.customXRightRegex =
          this.customProjectionXRightInput.getInRegexMode();
    }
    if (this.customProjectionYUpInput != null) {
      bookmark.customYUpText = this.customProjectionYUpInput.getValue();
      bookmark.customYUpRegex = this.customProjectionYUpInput.getInRegexMode();
    }
    if (this.customProjectionYDownInput != null) {
      bookmark.customYDownText = this.customProjectionYDownInput.getValue();
      bookmark.customYDownRegex =
          this.customProjectionYDownInput.getInRegexMode();
    }

    this.enablePolymerChangesTriggerReprojection();
  }

  // This method is marked as public as it is used as the view method that
  // abstracts DOM manipulation so we can stub it in a test.
  // TODO(nsthorat): Move this to its own class as the glue between this class
  // and the DOM.
  setZDropdownEnabled(enabled: boolean) {
    if (this.zDropdown) {
      this.zDropdown.attr('disabled', enabled ? null : true);
    }
  }

  dataSetUpdated(dataSet: DataSet, originalDataSet: DataSet, dim: number) {
    this.dataSet = dataSet;
    this.originalDataSet = originalDataSet;
    this.dim = dim;
    this.clearCentroids();

    this.dom.select('#tsne-sampling')
        .style('display', dataSet.points.length > SAMPLE_SIZE ? null : 'none');
    this.dom.select('#pca-sampling')
        .style('display', dataSet.dim[1] > PCA_SAMPLE_DIM ? null : 'none');
    this.showTab('pca');
  }

  _pcaDimensionToggleObserver() {
    this.setZDropdownEnabled(this.pcaIs3d);
    this.beginProjection(this.currentProjection);
  }

  _tsneDimensionToggleObserver() {
    this.beginProjection(this.currentProjection);
  }

  metadataChanged(spriteAndMetadata: SpriteAndMetadataInfo) {
    // Project by options for custom projections.
    let searchByMetadataIndex = -1;
    this.searchByMetadataOptions = spriteAndMetadata.stats.map((stats, i) => {
      // Make the default label by the first non-numeric column.
      if (!stats.isNumeric && searchByMetadataIndex === -1) {
        searchByMetadataIndex = i;
      }
      return stats.name;
    });
    this.customSelectedSearchByMetadataOption =
        this.searchByMetadataOptions[Math.max(0, searchByMetadataIndex)];
  }

  public showTab(id: Projection) {
    this.currentProjection = id;

    let tab = this.dom.select('.ink-tab[data-tab="' + id + '"]');
    this.dom.selectAll('.ink-tab').classed('active', false);
    tab.classed('active', true);
    this.dom.selectAll('.ink-panel-content').classed('active', false);
    this.dom.select('.ink-panel-content[data-panel="' + id + '"]')
        .classed('active', true);

    // guard for unit tests, where polymer isn't attached and $ doesn't exist.
    if (this.$ != null) {
      const main = this.$['main'];
      // In order for the projections panel to animate its height, we need to
      // set it explicitly.
      requestAnimationFrame(() => {
        this.style.height = main.clientHeight + 'px';
      });
    }

    this.beginProjection(id);
  }

  private beginProjection(projection: string) {
    if (this.polymerChangesTriggerReprojection === false) {
      return;
    }
    if (projection === 'pca') {
      this.dataSet.stopTSNE();
      this.showPCA();
    } else if (projection === 'tsne') {
      this.showTSNE();
    } else if (projection === 'custom') {
      this.dataSet.stopTSNE();
      this.computeAllCentroids();
      this.reprojectCustom();
    }
  }

  private showTSNE() {
    const dataSet = this.dataSet;
    if (dataSet == null) {
      return;
    }
    const accessors =
        dataSet.getPointAccessors('tsne', [0, 1, this.tSNEis3d ? 2 : null]);
    this.projector.setProjection('tsne', this.tSNEis3d ? 3 : 2, accessors);

    if (!this.dataSet.hasTSNERun) {
      this.runTSNE();
    } else {
      this.projector.notifyProjectionsUpdated();
    }
  }

  private runTSNE() {
    this.runTsneButton.attr('disabled', true);
    this.stopTsneButton.attr('disabled', null);
    this.dataSet.projectTSNE(
        this.perplexity, this.learningRate, this.tSNEis3d ? 3 : 2,
        (iteration: number) => {
          if (iteration != null) {
            this.iterationLabel.text(iteration);
            this.projector.notifyProjectionsUpdated();
          } else {
            this.runTsneButton.attr('disabled', null);
            this.stopTsneButton.attr('disabled', true);
          }
        });
  }

  // tslint:disable-next-line:no-unused-variable
  private showPCAIfEnabled() {
    if (this.polymerChangesTriggerReprojection) {
      this.showPCA();
    }
  }

  private showPCA() {
    if (this.dataSet == null) {
      return;
    }
    this.dataSet.projectPCA().then(() => {
      // Polymer properties are 1-based.
      const accessors = this.dataSet.getPointAccessors(
          'pca', [this.pcaX, this.pcaY, this.pcaZ]);

      this.projector.setProjection('pca', this.pcaIs3d ? 3 : 2, accessors);
    });
  }

  private reprojectCustom() {
    if (this.centroids == null || this.centroids.xLeft == null ||
        this.centroids.xRight == null || this.centroids.yUp == null ||
        this.centroids.yDown == null) {
      return;
    }
    const xDir = vector.sub(this.centroids.xRight, this.centroids.xLeft);
    this.dataSet.projectLinear(xDir, 'linear-x');

    const yDir = vector.sub(this.centroids.yUp, this.centroids.yDown);
    this.dataSet.projectLinear(yDir, 'linear-y');

    const accessors = this.dataSet.getPointAccessors('custom', ['x', 'y']);
    this.projector.setProjection('custom', 2, accessors);
  }

  clearCentroids(): void {
    this.centroids = {xLeft: null, xRight: null, yUp: null, yDown: null};
    this.allCentroid = null;
  }

  _customSelectedSearchByMetadataOptionChanged(newVal: string, oldVal: string) {
    if (this.polymerChangesTriggerReprojection === false) {
      return;
    }
    if (this.currentProjection === 'custom') {
      this.computeAllCentroids();
      this.reprojectCustom();
    }
  }

  private setupCustomProjectionInputFields() {
    this.customProjectionXLeftInput =
        this.setupCustomProjectionInputField('xLeft');
    this.customProjectionXRightInput =
        this.setupCustomProjectionInputField('xRight');
    this.customProjectionYUpInput = this.setupCustomProjectionInputField('yUp');
    this.customProjectionYDownInput =
        this.setupCustomProjectionInputField('yDown');
  }

  private computeAllCentroids() {
    this.computeCentroid('xLeft');
    this.computeCentroid('xRight');
    this.computeCentroid('yUp');
    this.computeCentroid('yDown');
  }

  private computeCentroid(name: InputControlName) {
    const input = this.querySelector('#' + name) as ProjectorInput;
    if (input == null) {
      return;
    }
    const value = input.getValue();
    if (value == null) {
      return;
    }
    let inRegexMode = input.getInRegexMode();
    let result = this.getCentroid(value, inRegexMode);
    if (result.numMatches === 0) {
      input.message = '0 matches. Using a random vector.';
      result.centroid = vector.rn(this.dim);
    } else {
      input.message = `${result.numMatches} matches.`;
    }
    this.centroids[name] = result.centroid;
    this.centroidValues[name] = value;
  }

  private setupCustomProjectionInputField(name: InputControlName):
      ProjectorInput {
    let input = this.querySelector('#' + name) as ProjectorInput;
    input.registerInputChangedListener((input, inRegexMode) => {
      if (this.polymerChangesTriggerReprojection) {
        this.computeCentroid(name);
        this.reprojectCustom();
      }
    });
    return input;
  }

  private getCentroid(pattern: string, inRegexMode: boolean): CentroidResult {
    if (pattern == null || pattern === '') {
      return {numMatches: 0};
    }
    // Search by the original dataset since we often want to filter and project
    // only the nearest neighbors of A onto B-C where B and C are not nearest
    // neighbors of A.
    let accessor = (i: number) => this.originalDataSet.points[i].vector;
    let r = this.originalDataSet.query(
        pattern, inRegexMode, this.customSelectedSearchByMetadataOption);
    return {centroid: vector.centroid(r, accessor), numMatches: r.length};
  }

  getPcaSampledDim() {
    return PCA_SAMPLE_DIM.toLocaleString();
  }

  getTsneSampleSize() {
    return SAMPLE_SIZE.toLocaleString();
  }

  _addOne(value: number) {
    return value + 1;
  }
}

document.registerElement(ProjectionsPanel.prototype.is, ProjectionsPanel);
