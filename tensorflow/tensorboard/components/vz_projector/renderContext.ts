/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http:www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * LabelRenderParams describes the set of points that should have labels
 * rendered next to them.
 */
export class LabelRenderParams {
  constructor(
      public pointIndices: Float32Array, public scaleFactors: Float32Array,
      public useSceneOpacityFlags: Int8Array, public defaultFontSize: number,
      public fillColors: Uint8Array, public strokeColors: Uint8Array) {}
}

/** Details about the camera projection being used to render the scene. */
export enum CameraType {
  Perspective,
  Orthographic
}

/**
 * RenderContext contains all of the state required to color and render the data
 * set. ScatterPlot passes this to every attached visualizer as part of the
 * render callback.
 * TODO(nicholsonc): This should only contain the data that's changed between
 * each frame. Data like colors / scale factors / labels should be reapplied
 * only when they change.
 */
export class RenderContext {
  constructor(
      public camera: THREE.Camera, public cameraType: CameraType,
      public cameraTarget: THREE.Vector3, public screenWidth: number,
      public screenHeight: number, public nearestCameraSpacePointZ: number,
      public farthestCameraSpacePointZ: number, public backgroundColor: number,
      public pointColors: Float32Array, public pointScaleFactors: Float32Array,
      public labelAccessor: (index: number) => string,
      public labels: LabelRenderParams,
      public traceColors: {[trace: number]: Float32Array},
      public traceOpacities: Float32Array, public traceWidths: Float32Array) {}
}
