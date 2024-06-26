# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os

from kfp import compiler
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, OutputPath

from kfp import kubernetes

# This component downloads the evaluation data, scaler and model from an S3 bucket and saves it to the correspoding output paths.
# The connection to the S3 bucket is created using this environment variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_S3_BUCKET
# - AWS_S3_ENDPOINT
# - SCALER_S3_KEY
# - EVALUATION_DATA_S3_KEY
# - MODEL_S3_KEY
# The data is in pickel format and the file name is passed as an environment variable S3_KEY.
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3", "botocore"]
)
def get_evaluation_kit(
    evaluation_data_output_dataset: Output[Dataset],
    scaler_output_model: Output[Model],
    output_model: Output[Model]
):
    import boto3
    import botocore
    import os
    import zipfile
    import shutil

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')
    evaluation_kit_s3_key = os.environ.get('EVALUATION_KIT_S3_KEY')

    evaluation_data_zip_path = os.environ.get('EVALUATION_DATA_ZIP_PATH')
    scaler_zip_path = os.environ.get('SCALER_ZIP_PATH')
    model_zip_path = os.environ.get('MODEL_ZIP_PATH')

    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name
    )

    bucket = s3_resource.Bucket(bucket_name)

    # Create a temporary directory to store the evaluation kit
    
    local_tmp_dir = '/tmp/get_evaluation_kit'
    print(f"local_tmp_dir: {local_tmp_dir}")
    
    # Ensure local_tmp_dir exists
    if not os.path.exists(local_tmp_dir):
        os.makedirs(local_tmp_dir)

    # Get the file name from the S3 key
    file_name = os.path.basename(evaluation_kit_s3_key)    
    # Download the evaluation kit
    local_file_path = f'{local_tmp_dir}/{file_name}'
    print(f"Downloading {evaluation_kit_s3_key} to {local_file_path}")
    bucket.download_file(evaluation_kit_s3_key, local_file_path)
    print(f"Downloaded {evaluation_kit_s3_key}")

    # Unzip the evaluation kit using zipfile module
    extraction_dir = f'{local_tmp_dir}/evaluation_kit'
    print(f"Extracting {local_file_path} in {extraction_dir}")
    with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)
    print(f"Extracted {local_file_path} in {extraction_dir}")

     # Copy the evaluation evaluation_kit/model.onnx to the model output path
    print(f"Copying {extraction_dir}/{model_zip_path} to {output_model.path}")
    shutil.copy(f'{extraction_dir}/{model_zip_path}', output_model.path)
    
    # Copy the evaluation evaluation_kit/scaler.pkl to the scaler output path
    print(f"Copying {extraction_dir}/{scaler_zip_path} to {scaler_output_model.path}")
    shutil.copy(f'{extraction_dir}/{scaler_zip_path}', scaler_output_model.path)

    # Copy the evaluation evaluation_kit/evaluation_data.pkl to the evaluation data output path
    print(f"Copying {extraction_dir}/{evaluation_data_zip_path} to {evaluation_data_output_dataset.path}")
    shutil.copy(f'{extraction_dir}/{evaluation_data_zip_path}', evaluation_data_output_dataset.path)

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["onnx==1.16.1", "onnxruntime==1.18.0", "scikit-learn==1.5.0", "numpy==1.24.3", "pandas==2.2.2"]
)
def test_model(
    evaluation_data_input_dataset: Input[Dataset],
    scaler_input_model: Input[Model],
    model_input_model: Input[Model],
    results_output_metrics: Output[Metrics]
):
    import numpy as np
    import pickle
    import onnxruntime as rt

    # Load the evaluation data and scaler
    with open(evaluation_data_input_dataset.path, 'rb') as handle:
        (X_test, y_test) = pickle.load(handle)
    with open(scaler_input_model.path, 'rb') as handle:
        scaler = pickle.load(handle)

    sess = rt.InferenceSession(model_input_model.path, providers=rt.get_available_providers())
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    y_pred_temp = sess.run([output_name], {input_name: scaler.transform(X_test.values).astype(np.float32)}) 
    y_pred_temp = np.asarray(np.squeeze(y_pred_temp[0]))
    threshold = 0.995
    y_pred = np.where(y_pred_temp > threshold, 1, 0)

    accuracy = np.sum(np.asarray(y_test) == y_pred) / len(y_pred)
    # print("Accuracy: " + str(accuracy))

    results_output_metrics.log_metric("accuracy", accuracy)

# This component parses metrics and returns the accuracy
# @dsl.component(
#     base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301"
# )
# def parse_metrics(metrics_input: Input[Metrics], accuracy_output: Output[float]):
#     accuracy = metrics_input.get_metric("accuracy")
#     accuracy_output.write(accuracy)
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301"
)
def parse_metrics(metrics_input: Input[Metrics], accuracy_output: OutputPath(float)):
    print(f"metrics_input: {dir(metrics_input)}")
    accuracy = metrics_input.metadata["accuracy"]
    with open(accuracy_output, 'w') as f:
        f.write(str(accuracy))

# This component uploads the model to an S3 bucket. The connection to the S3 bucket is created using this environment variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_S3_BUCKET
# - AWS_S3_ENDPOINT
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3", "botocore"]
)
def upload_model(input_model: Input[Model]):
    import os
    import boto3
    import botocore

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    s3_key = os.environ.get("MODEL_S3_KEY")

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)

    print(f"Uploading {s3_key}")
    bucket.upload_file(input_model.path, s3_key)


# This pipeline will download evaluation data, download the model, test the model and if it performs well, 
# upload the model to the runtime S3 bucket and refresh the runtime deployment.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(accuracy_threshold: float = 0.95, no_metrics: bool = True, local: bool = False):
    # Get the evaluation data, scaler and model
    get_evaluation_kit_task = get_evaluation_kit()
    # evaluation_data_set = get_evaluation_kit_task.outputs["evaluation_data_output_dataset"]
    # scaler_model = get_evaluation_kit_task.outputs["scaler_output_model"]
    # model = get_evaluation_kit_task.outputs["output_model"]

    # Test the model
    test_model_task = test_model(
        evaluation_data_input_dataset=get_evaluation_kit_task.outputs["evaluation_data_output_dataset"],
        scaler_input_model=get_evaluation_kit_task.outputs["scaler_output_model"], 
        model_input_model=get_evaluation_kit_task.outputs["output_model"]
    )

    # Parse the metrics and extract the accuracy
    parse_metrics_task = parse_metrics(metrics_input=test_model_task.outputs["results_output_metrics"])
    accuracy = parse_metrics_task.outputs["accuracy_output"]

    # Use the parsed accuracy to decide if we should upload the model
    with dsl.If(accuracy >= accuracy_threshold):
        upload_model_task = upload_model(input_model=get_evaluation_kit_task.outputs["scaler_output_model"])

        # Setting environment variables for upload_model_task
        upload_model_task.set_env_variable(name="S3_KEY", value="models/fraud/1/model.onnx")
        kubernetes.use_secret_as_env(
            task=upload_model_task,
            secret_name='aws-connection-model-runtime',
            secret_key_to_env={
                'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
                'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
                'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
                'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
                'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            }
        )

    # Print the accuracy
    print(f"accuracy = {accuracy}")

    # Set the S3 keys for get_evaluation_kit_task and kubernetes secret to be used in the task
    get_evaluation_kit_task.set_env_variable(name="EVALUATION_KIT_S3_KEY", value="evaluation_kit.zip")
    get_evaluation_kit_task.set_env_variable(name="EVALUATION_DATA_ZIP_PATH", value="artifact/test_data.pkl")
    get_evaluation_kit_task.set_env_variable(name="SCALER_ZIP_PATH", value="artifact/scaler.pkl")
    get_evaluation_kit_task.set_env_variable(name="MODEL_ZIP_PATH", value="models/fraud/1/model.onnx")

    kubernetes.use_secret_as_env(
        task=get_evaluation_kit_task,
        secret_name='aws-connection-model-runtime',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=__file__.replace('.py', '.yaml')
    )