import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import {bundleResourceIO, decodeJpeg} from '@tensorflow/tfjs-react-native';
import {Base64Binary} from '../utils/utils';
const BITMAP_DIMENSION = 224;

// const modelJson = require('../model/model_original_mango_1_224/model.json');
// const modelJson = require('../model/model_original_mango_1_224/model.json');
const modelJson = require('../model/transfer_model_original_total_1_224/model.json');
// const modelWeights_1 = require('../model/shine/group1-shard1of1.bin');
// const modelWeights_1 = require('../model/model_original_mango_1_224/group1-shard1of1.bin');
// const modelWeights_2 = require('../model/model_rembg_mango_1_94/group1-shard2of2.bin');
// const modelWeights = require('../model/model_original_mango_1_224/group1-shard1of1.bin');
let modelWeights = [];

let a = require('../model/transfer_model_original_total_1_224/group1-shard1of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard2of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard3of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard4of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard5of17.bin');

modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard6of17.bin');

modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard7of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard8of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard9of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard10of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard11of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard12of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard13of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard14of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard15of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard16of17.bin');
modelWeights.push(a);
a = require('../model/transfer_model_original_total_1_224/group1-shard17of17.bin');
modelWeights.push(a);

// 0: channel from JPEG-encoded image
// 1: gray scale
// 3: RGB image
const TENSORFLOW_CHANNEL = 3;

export const getModel = async () => {
  try {
    // wait until tensorflow is ready
    console.log(modelJson, '\n weight:', modelWeights);

    await tf.ready();

    // load the trained model
    return await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
  } catch (error) {
    console.log('Could not load model', error);
  }
};

export const convertBase64ToTensor = async (base64) => {
  try {
    const uIntArray = Base64Binary.decode(base64);
    // decode a JPEG-encoded image to a 3D Tensor of dtype
    const decodedImage = decodeJpeg(uIntArray, 3);
    // reshape Tensor into a 4D array
    return decodedImage.reshape([
      1,
      BITMAP_DIMENSION,
      BITMAP_DIMENSION,
      TENSORFLOW_CHANNEL,
    ]);
  } catch (error) {
    console.log('Could not convert base64 string to tesor', error);
  }
};

export const startPrediction = async (model, tensor) => {
  try {
    // predict against the model
    // console.log(model);
    const output = await model.predict(tensor);
    // return typed array
    return output.dataSync();
  } catch (error) {
    console.log('Error predicting from tesor image', error);
  }
};
