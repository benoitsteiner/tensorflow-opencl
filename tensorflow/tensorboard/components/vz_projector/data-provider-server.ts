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

import {DataSet, MetadataInfo, State} from './data';
import {ProjectorConfig, DataProvider, TENSORS_MSG_ID, EmbeddingInfo} from './data-provider';
import * as dataProvider from './data-provider';
import * as logging from './logging';


/**
 * Data provider that loads data provided by a python server (usually backed
 * by a checkpoint file).
 */
export class ServerDataProvider implements DataProvider {
  private routePrefix: string;
  private runProjectorConfigCache: {[run: string]: ProjectorConfig} = {};

  constructor(routePrefix: string) {
    this.routePrefix = routePrefix;
  }

  private getEmbeddingInfo(run: string, tensorName: string,
      callback: (e: EmbeddingInfo) => void): void {
    this.retrieveProjectorConfig(run, config => {
      let embeddings = config.embeddings;
      for (let i = 0; i < embeddings.length; i++) {
        let embedding = embeddings[i];
        if (embedding.tensorName === tensorName) {
          callback(embedding);
          return;
        }
      }
      callback(null);
    });
  }

  retrieveRuns(callback: (runs: string[]) => void): void {
    let msgId = logging.setModalMessage('Fetching runs...');
    d3.json(`${this.routePrefix}/runs`, (err, runs: string[]) => {
      if (err) {
        logging.setModalMessage('Error: ' + err.responseText);
        return;
      }
      logging.setModalMessage(null, msgId);
      callback(runs);
    });
  }

  retrieveProjectorConfig(run: string, callback: (d: ProjectorConfig) => void)
      : void {
    if (run in this.runProjectorConfigCache) {
      callback(this.runProjectorConfigCache[run]);
      return;
    }

    let msgId = logging.setModalMessage('Fetching projector config...');
    d3.json(`${this.routePrefix}/info?run=${run}`, (err,
        config: ProjectorConfig) => {
      if (err) {
        logging.setModalMessage('Error: ' + err.responseText);
        return;
      }
      logging.setModalMessage(null, msgId);
      this.runProjectorConfigCache[run] = config;
      callback(config);
    });
  }

  retrieveTensor(run: string, tensorName: string, callback: (ds: DataSet) => void) {
    // Get the tensor.
    logging.setModalMessage('Fetching tensor values...', TENSORS_MSG_ID);
    d3.text(
        `${this.routePrefix}/tensor?run=${run}&name=${tensorName}`,
        (err: any, tsv: string) => {
          if (err) {
            logging.setModalMessage('Error: ' + err.responseText);
            return;
          }
          dataProvider.parseTensors(tsv).then(dataPoints => {
            callback(new DataSet(dataPoints));
          });
        });
  }

  retrieveMetadata(run: string, tensorName: string,
      callback: (r: MetadataInfo) => void) {
    this.getEmbeddingInfo(run, tensorName, embedding => {
      let metadataPath = null;
      if (embedding.metadataPath) {
        metadataPath =
            `${this.routePrefix}/metadata?run=${run}&name=${tensorName}`;
      }
      let spriteImagePath = null;
      if (embedding.sprite && embedding.sprite.imagePath) {
        spriteImagePath =
            `${this.routePrefix}/sprite_image?run=${run}&name=${tensorName}`;
      }
      dataProvider.retrieveMetadataInfo(metadataPath, spriteImagePath,
          embedding.sprite, callback);
    });
  }

  getDefaultTensor(run: string, callback: (tensorName: string) => void) {
    this.retrieveProjectorConfig(run, config => {
      let tensorNames = config.embeddings.map(e => e.tensorName);
      // Return the first tensor that has metadata.
      for (let i = 0; i < tensorNames.length; i++) {
        let e = config.embeddings[i];
        if (e.metadataPath) {
          callback(e.tensorName);
          return;
        }
      }
      callback(tensorNames.length >= 1 ? tensorNames[0] : null);
    });
  }

  getBookmarks(
      run: string, tensorName: string, callback: (r: State[]) => void) {
    let msgId = logging.setModalMessage('Fetching bookmarks...');
    d3.json(
        `${this.routePrefix}/bookmarks?run=${run}&name=${tensorName}`,
        (err, bookmarks: State[]) => {
          logging.setModalMessage(null, msgId);
          if (!err) {
            callback(bookmarks);
          }
        });
  }
}
