# Sagemaker to RHOAI Fraud Detection

Based on: <https://rh-aiservices-bu.github.io/fraud-detection/>

# Preparation

Once this repo is cloned in Sagemaker Jupyter Notebook create a `.env` file with credentials to access and manipute objects in a bucket, something like:

```sh
export AWS_ACCESS_KEY_ID="AK..."
export AWS_SECRET_ACCESS_KEY="7S.."
export AWS_REGION="eu-central-1"
export AWS_S3_BUCKET="sagemaker-models-XYZ"
```