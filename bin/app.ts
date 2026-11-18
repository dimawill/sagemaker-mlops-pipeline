#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { MlPipelineStack, type Stage } from '../lib/ml-pipeline-stack';

const app = new cdk.App();
const stage = app.node.tryGetContext('stage') as string;
if (stage !== 'dev' && stage !== 'prod') throw new Error('Set -c stage=dev or -c stage=prod');
const flag = app.node.tryGetContext('deployEndpoint');
if (flag !== true && flag !== false && flag !== 'true' && flag !== 'false') {
  throw new Error('Set -c deployEndpoint=true or false');
}
new MlPipelineStack(app, `DemandForecastMlops-${stage}`, {
  stage: stage as Stage,
  deployEndpoint: flag === true || flag === 'true',
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});
