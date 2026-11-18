import * as path from 'node:path';
import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as sm from 'aws-cdk-lib/aws-sagemaker';
import { Construct } from 'constructs';

export type Stage = 'dev' | 'prod';
export interface MlPipelineStackProps extends cdk.StackProps {
  stage: Stage;
  deployEndpoint: boolean;
}

export class MlPipelineStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: MlPipelineStackProps) {
    super(scope, id, props);
    const removalPolicy = props.stage === 'prod' ? cdk.RemovalPolicy.RETAIN : cdk.RemovalPolicy.DESTROY;
    const bucket = (name: string) => new s3.Bucket(this, name, {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy,
      lifecycleRules: [{ expiration: cdk.Duration.days(30), abortIncompleteMultipartUploadAfter: cdk.Duration.days(7) }],
    });
    const data = bucket('InputBucket');
    const artifacts = bucket('ArtifactBucket');
    const imageUri = new cdk.CfnParameter(this, 'XgboostImageUri', {
      type: 'String', description: 'Region-specific SageMaker XGBoost 3.0-5 algorithm container URI (resolve with SageMaker image_uris.retrieve)',
      allowedPattern: '[0-9]+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(\.cn)?/[^ ]+',
    });
    const group = new sm.CfnModelPackageGroup(this, 'ModelGroup', {
      modelPackageGroupName: `demand-forecast-${props.stage}`,
      modelPackageGroupDescription: 'Gated synthetic-demand XGBoost model versions',
    });
    const smRole = new iam.Role(this, 'SageMakerRole', { assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com') });
    smRole.addToPolicy(new iam.PolicyStatement({ actions: ['s3:GetObject'], resources: [artifacts.arnForObjects('runs/*')] }));
    smRole.addToPolicy(new iam.PolicyStatement({ actions: ['s3:PutObject'], resources: [artifacts.arnForObjects('runs/*'), artifacts.arnForObjects('models/*')] }));
    smRole.addToPolicy(new iam.PolicyStatement({ actions: ['s3:GetObject'], resources: [artifacts.arnForObjects('models/*')] }));
    smRole.addToPolicy(new iam.PolicyStatement({ actions: ['s3:ListBucket'], resources: [artifacts.bucketArn],
      conditions: { StringLike: { 's3:prefix': ['runs/*', 'models/*'] } } }));
    smRole.addToPolicy(new iam.PolicyStatement({ actions: ['s3:GetBucketLocation'], resources: [artifacts.bucketArn] }));
    smRole.addToPolicy(new iam.PolicyStatement({
      actions: ['logs:CreateLogStream', 'logs:PutLogEvents', 'logs:CreateLogGroup'],
      resources: [this.formatArn({ service: 'logs', resource: 'log-group',
        resourceName: '/aws/sagemaker/*', arnFormat: cdk.ArnFormat.COLON_RESOURCE_NAME })],
    }));
    const mlCode = lambda.Code.fromAsset(path.join(__dirname, '../ml'));
    const fn = (name: string, file: string, environment: Record<string, string>) => new lambda.Function(this, name, {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: `${file}.handler`, code: mlCode, environment,
      memorySize: 256, timeout: cdk.Duration.seconds(60),
      logGroup: new logs.LogGroup(this, `${name}Logs`, { retention: logs.RetentionDays.ONE_WEEK,
        removalPolicy: cdk.RemovalPolicy.DESTROY }),
    });
    const prep = fn('Preprocess', 'preprocess', { DATA_BUCKET: data.bucketName, ARTIFACT_BUCKET: artifacts.bucketName });
    const evaluate = fn('Evaluate', 'evaluate', { ARTIFACT_BUCKET: artifacts.bucketName, RMSE_THRESHOLD: '15' });
    const register = fn('Register', 'registry', { GROUP_NAME: group.ref, IMAGE_URI: imageUri.valueAsString });
    data.grantRead(prep, 'input/demand.csv');
    artifacts.grantPut(prep, 'runs/*');
    artifacts.grantRead(evaluate, 'runs/*');
    artifacts.grantPut(evaluate, 'runs/*');
    artifacts.grantRead(register, 'runs/*');
    register.addToRolePolicy(new iam.PolicyStatement({ actions: ['sagemaker:CreateModelPackage'],
      resources: [this.formatArn({ service: 'sagemaker', resource: 'model-package', resourceName: `${group.ref}/*`,
        arnFormat: cdk.ArnFormat.SLASH_RESOURCE_NAME }),
      this.formatArn({ service: 'sagemaker', resource: 'model-package-group', resourceName: group.ref })] }));

    const machineRole = new iam.Role(this, 'OrchestratorRole', { assumedBy: new iam.ServicePrincipal('states.amazonaws.com') });
    for (const function_ of [prep, evaluate, register]) function_.grantInvoke(machineRole);
    machineRole.addToPolicy(new iam.PolicyStatement({ actions: ['iam:PassRole'], resources: [smRole.roleArn],
      conditions: { StringEquals: { 'iam:PassedToService': 'sagemaker.amazonaws.com' } } }));
    const smArn = (type: string, name: string) => this.formatArn({ service: 'sagemaker', resource: type,
      resourceName: name, arnFormat: cdk.ArnFormat.SLASH_RESOURCE_NAME });
    machineRole.addToPolicy(new iam.PolicyStatement({
      actions: ['sagemaker:CreateTrainingJob', 'sagemaker:DescribeTrainingJob', 'sagemaker:StopTrainingJob'],
      resources: [smArn('training-job', 'ml-*')],
    }));
    machineRole.addToPolicy(new iam.PolicyStatement({
      actions: ['sagemaker:CreateModel', 'sagemaker:DescribeModel'], resources: [smArn('model', 'ml-*')],
    }));
    machineRole.addToPolicy(new iam.PolicyStatement({
      actions: ['sagemaker:CreateTransformJob', 'sagemaker:DescribeTransformJob', 'sagemaker:StopTransformJob'],
      resources: [smArn('transform-job', 'eval-*'), smArn('transform-job', 'batch-*')],
    }));
    machineRole.addToPolicy(new iam.PolicyStatement({
      actions: ['events:PutRule', 'events:PutTargets', 'events:DescribeRule'],
      resources: [this.formatArn({ service: 'events', resource: 'rule',
        resourceName: 'StepFunctionsGetEventsForSageMaker*', arnFormat: cdk.ArnFormat.SLASH_RESOURCE_NAME })],
    }));
    if (props.deployEndpoint) machineRole.addToPolicy(new iam.PolicyStatement({
      actions: ['sagemaker:CreateEndpointConfig', 'sagemaker:DescribeEndpointConfig',
        'sagemaker:CreateEndpoint', 'sagemaker:DescribeEndpoint'],
      resources: [smArn('endpoint-config', 'ml-*'), smArn('endpoint', 'ml-*')],
    }));
    const state = (name: string, resource: string, parameters: Record<string, unknown>, resultPath: string) =>
      new sfn.CustomState(this, name, { stateJson: { Type: 'Task', Resource: resource, Parameters: parameters,
        ResultPath: resultPath } });
    const invoke = (name: string, func: lambda.Function, payload: Record<string, unknown>, resultPath: string) =>
      state(name, 'arn:aws:states:::lambda:invoke', { FunctionName: func.functionArn, Payload: payload }, resultPath);
    const start = invoke('PrepareData', prep, { 'run.$': '$$.Execution.Name' }, '$.prep');
    const training = state('TrainXGBoost', 'arn:aws:states:::sagemaker:createTrainingJob.sync', {
      'TrainingJobName.$': "States.Format('ml-{}', $$.Execution.Name)",
      RoleArn: smRole.roleArn,
      AlgorithmSpecification: { TrainingImage: imageUri.valueAsString, TrainingInputMode: 'File' },
      InputDataConfig: ['train', 'validation'].map(channel => ({ ChannelName: channel,
        DataSource: { S3DataSource: { S3DataType: 'S3Prefix', 'S3Uri.$': `$.prep.Payload.${channel}`,
          S3DataDistributionType: 'FullyReplicated' } }, ContentType: 'text/csv' })),
      OutputDataConfig: { S3OutputPath: `s3://${artifacts.bucketName}/models/` },
      ResourceConfig: { InstanceCount: 1, InstanceType: 'ml.m5.large', VolumeSizeInGB: 10 },
      StoppingCondition: { MaxRuntimeInSeconds: 1800 },
      HyperParameters: { objective: 'reg:squarederror', num_round: '60', max_depth: '3', eta: '0.2',
        eval_metric: 'rmse' },
    }, '$.training');
    const model = state('CreateEvaluationModel', 'arn:aws:states:::aws-sdk:sagemaker:createModel', {
      'ModelName.$': "States.Format('ml-{}', $$.Execution.Name)", ExecutionRoleArn: smRole.roleArn,
      PrimaryContainer: { Image: imageUri.valueAsString,
        'ModelDataUrl.$': '$.training.ModelArtifacts.S3ModelArtifacts' },
    }, '$.model');
    const transform = (name: string, prefix: string, input: string, output: string, resultPath: string) =>
      state(name, 'arn:aws:states:::sagemaker:createTransformJob.sync', {
        'TransformJobName.$': `States.Format('${prefix}-{}', $$.Execution.Name)`,
        'ModelName.$': "States.Format('ml-{}', $$.Execution.Name)",
        TransformInput: { DataSource: { S3DataSource: { S3DataType: 'S3Prefix',
          'S3Uri.$': `$.prep.Payload.${input}` } }, ContentType: 'text/csv', SplitType: 'Line' },
        TransformOutput: { 'S3OutputPath.$': `$.prep.Payload.${output}`, AssembleWith: 'Line',
          Accept: 'text/csv' },
        TransformResources: { InstanceCount: 1, InstanceType: 'ml.m5.large' },
        BatchStrategy: 'SingleRecord',
      }, resultPath);
    const evalTransform = transform('EvaluateWithBatchTransform', 'eval', 'test', 'eval_output', '$.prediction');
    const gate = invoke('CalculateHeldoutRMSE', evaluate, { 'run.$': '$.prep.Payload.run' }, '$.evaluated');
    const registry = invoke('RegisterApprovedVersion', register, {
      'passed.$': '$.evaluated.Payload.passed',
      'report_uri.$': '$.evaluated.Payload.report_uri',
      'model_uri.$': '$.training.ModelArtifacts.S3ModelArtifacts',
    }, '$.registered');
    const success = new sfn.Succeed(this, 'Done');
    let afterBatch: sfn.IChainable = success;
    if (props.deployEndpoint) {
      const endpointConfig = state('CreateEndpointConfig', 'arn:aws:states:::aws-sdk:sagemaker:createEndpointConfig', {
        'EndpointConfigName.$': "States.Format('ml-{}', $$.Execution.Name)",
        ProductionVariants: [{ VariantName: 'AllTraffic',
          'ModelName.$': "States.Format('ml-{}', $$.Execution.Name)",
          InitialInstanceCount: 1, InstanceType: 'ml.m5.large', InitialVariantWeight: 1 }],
        DataCaptureConfig: { EnableCapture: true, InitialSamplingPercentage: 100,
          DestinationS3Uri: `s3://${artifacts.bucketName}/runs/monitor/`,
          CaptureOptions: [{ CaptureMode: 'Input' }, { CaptureMode: 'Output' }] },
      }, '$.endpointConfig');
      const endpoint = state('CreateEndpoint', 'arn:aws:states:::aws-sdk:sagemaker:createEndpoint', {
        'EndpointName.$': "States.Format('ml-{}', $$.Execution.Name)",
        'EndpointConfigName.$': "States.Format('ml-{}', $$.Execution.Name)",
      }, '$.endpoint');
      const wait = new sfn.Wait(this, 'WaitForEndpoint', { time: sfn.WaitTime.duration(cdk.Duration.seconds(60)) });
      const describe = state('CheckEndpoint', 'arn:aws:states:::aws-sdk:sagemaker:describeEndpoint', {
        'EndpointName.$': "States.Format('ml-{}', $$.Execution.Name)",
      }, '$.endpointStatus');
      const status = new sfn.Choice(this, 'EndpointReady');
      status.when(sfn.Condition.stringEquals('$.endpointStatus.EndpointStatus', 'InService'), success);
      status.when(sfn.Condition.stringEquals('$.endpointStatus.EndpointStatus', 'Failed'),
        new sfn.Fail(this, 'EndpointFailed', { cause: 'Inspect SageMaker endpoint failure reason and delete resources' }));
      status.otherwise(wait);
      wait.next(describe).next(status);
      endpointConfig.next(endpoint).next(wait);
      afterBatch = endpointConfig;
    }
    const batch = transform('OptionalBatchInference', 'batch', 'batch', 'batch_output', '$.batchResult');
    batch.next(afterBatch);
    const chooseBatch = new sfn.Choice(this, 'RunBatchInference');
    chooseBatch.when(sfn.Condition.booleanEquals('$.runBatch', true), batch);
    chooseBatch.otherwise(afterBatch);
    registry.next(chooseBatch);
    const metric = new sfn.Choice(this, 'MetricGate');
    metric.when(sfn.Condition.booleanEquals('$.evaluated.Payload.passed', true), registry);
    metric.otherwise(new sfn.Fail(this, 'BelowThreshold', { cause: 'Held-out RMSE exceeded 15; no model registration' }));
    start.next(training).next(model).next(evalTransform).next(gate).next(metric);
    const logGroup = new logs.LogGroup(this, 'WorkflowLogs', {
      retention: logs.RetentionDays.ONE_WEEK, removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const machine = new sfn.StateMachine(this, 'Pipeline', {
      definitionBody: sfn.DefinitionBody.fromChainable(start), role: machineRole,
      timeout: cdk.Duration.hours(2),
      logs: { destination: logGroup, level: sfn.LogLevel.ERROR, includeExecutionData: false },
      tracingEnabled: true,
    });
    new cloudwatch.Alarm(this, 'FailedExecutionsAlarm', {
      metric: machine.metricFailed({ period: cdk.Duration.minutes(5) }),
      threshold: 1, evaluationPeriods: 1, treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    new cloudwatch.Alarm(this, 'TimedOutExecutionsAlarm', {
      metric: machine.metricTimedOut({ period: cdk.Duration.minutes(5) }),
      threshold: 1, evaluationPeriods: 1, treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    new cdk.CfnOutput(this, 'InputBucketName', { value: data.bucketName });
    new cdk.CfnOutput(this, 'ArtifactBucketName', { value: artifacts.bucketName });
    new cdk.CfnOutput(this, 'StateMachineArn', { value: machine.stateMachineArn });
    new cdk.CfnOutput(this, 'ModelPackageGroupName', { value: group.ref });
  }
}
