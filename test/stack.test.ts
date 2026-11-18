import * as cdk from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { describe, expect, it } from 'vitest';
import { MlPipelineStack } from '../lib/ml-pipeline-stack';

function synth(deployEndpoint = false, stage: 'dev' | 'prod' = 'dev') {
  const app = new cdk.App();
  return Template.fromStack(new MlPipelineStack(app, `DemandForecastMlops-${stage}`, { stage, deployEndpoint }));
}
function definition(template: Template): string {
  return JSON.stringify(Object.values(template.findResources('AWS::StepFunctions::StateMachine'))[0].Properties.DefinitionString);
}

describe('demand forecasting MLOps', () => {
  it('protects both buckets, keeps runtime scoped, retains production data', () => {
    const dev = synth();
    dev.resourceCountIs('AWS::S3::Bucket', 2);
    dev.resourceCountIs('AWS::S3::BucketPolicy', 2);
    for (const resource of Object.values(dev.findResources('AWS::S3::Bucket'))) {
      expect(resource.Properties.BucketEncryption.ServerSideEncryptionConfiguration[0]
        .ServerSideEncryptionByDefault.SSEAlgorithm).toBe('AES256');
      expect(resource.Properties.PublicAccessBlockConfiguration).toEqual({
        BlockPublicAcls: true, BlockPublicPolicy: true, IgnorePublicAcls: true, RestrictPublicBuckets: true,
      });
      expect(resource.Properties.LifecycleConfiguration.Rules[0].ExpirationInDays).toBe(30);
    }
    dev.hasResourceProperties('AWS::S3::BucketPolicy', {
      PolicyDocument: { Statement: Match.arrayWith([Match.objectLike({
        Effect: 'Deny', Condition: { Bool: { 'aws:SecureTransport': 'false' } },
      })]) },
    });
    for (const bucket of Object.values(synth(false, 'prod').findResources('AWS::S3::Bucket'))) {
      expect(bucket.DeletionPolicy).toBe('Retain');
    }
    const policies = JSON.stringify(dev.findResources('AWS::IAM::Policy'));
    expect(policies).not.toContain('s3:*');
    expect(policies).not.toContain('sagemaker:*');
  });

  it('enforces training, heldout transform, gate, registry, optional batch and no default endpoint', () => {
    const template = synth();
    const body = definition(template);
    expect(body).toContain('sagemaker:createTrainingJob.sync');
    expect(body).toContain('sagemaker:createTransformJob.sync');
    expect(body).toContain('reg:squarederror');
    expect(body).toContain('MetricGate');
    expect(body).toContain('BelowThreshold');
    expect(body).toContain('RegisterApprovedVersion');
    expect(body).toContain('OptionalBatchInference');
    expect(body).not.toContain('sagemaker:createEndpoint');
    template.resourceCountIs('AWS::SageMaker::ModelPackageGroup', 1);
    template.resourceCountIs('AWS::CloudWatch::Alarm', 2);
    template.resourceCountIs('AWS::Lambda::Function', 3);
    template.hasResourceProperties('AWS::Lambda::Function', { Runtime: 'python3.12' });
    template.hasParameter('XgboostImageUri', Match.objectLike({ Type: 'String' }));
  });

  it('only permits expensive captured endpoint flow with explicit compile-time flag', () => {
    const body = definition(synth(true));
    expect(body).toContain('sagemaker:createEndpointConfig');
    expect(body).toContain('sagemaker:createEndpoint');
    expect(body).toContain('DataCaptureConfig');
    expect(body).toContain('CheckEndpoint');
    expect(body).toContain('InService');
  });
});
