#!/bin/bash

cd /app

export LOCALSTACK_ENDPOINT=http://localhost:4566
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test

aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set default.region $AWS_REGION
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY


echo "🚀 Setting up LocalStack services..."

# Create buckets 
create_s3_bucket() {
  BUCKET_NAME=$1
  if aws s3api head-bucket --bucket "$BUCKET_NAME" --endpoint-url=$LOCALSTACK_ENDPOINT 2>/dev/null; then
    echo "✅ S3 bucket '$BUCKET_NAME' already exists. Skipping creation."
  else
    aws s3 mb s3://$BUCKET_NAME --endpoint-url=$LOCALSTACK_ENDPOINT
    echo "✅ S3 bucket '$BUCKET_NAME' created."
  fi
}

create_s3_bucket "clustered-sentences"

create_s3_bucket "vocab"

create_s3_bucket "tf-idf"

create_s3_bucket "vectors"

create_s3_bucket "cluster-models"

create_s3_bucket "cluster-plots"

echo "✅ Buckets built (or not)."

# 2. Create an IAM Role for Lambda
ROLE_NAME="lambda-local-role"
ROLE_EXISTS=$(aws iam get-role --role-name $ROLE_NAME --endpoint-url=$LOCALSTACK_ENDPOINT 2>/dev/null)

if [ -n "$ROLE_EXISTS" ]; then
    echo "✅ IAM Role '$ROLE_NAME' already exists. Skipping creation."
else
    aws iam create-role --role-name $ROLE_NAME \
      --assume-role-policy-document file://role-policy.json \
      --endpoint-url=$LOCALSTACK_ENDPOINT > /dev/null
    echo "✅ IAM Role created."
fi

# 3. Deploy Lambdas

## test lambdas
function test_lambda {
    LAMBDA_NAME=$1
    LAMBDA_DIR=$2
    
    cd $LAMBDA_DIR

    ## run script
    python test_lambda.py

    local exit_code=$?

    # Capture the exit code of the test
    if [ $exit_code -ne 0 ]; then
        echo "Lambda test failed for $LAMBDA_NAME. Aborting build."
        return 1
    else
        echo "Lambda test passed for $LAMBDA_NAME."
        return 0
    fi
}

# Function to zip lambda code 
function zip_lambdas {
    LAMBDA_NAME=$1
    LAMBDA_DIR=$2

    # Create the zip file
    (cd "$LAMBDA_DIR" && zip -r "$LAMBDA_NAME" .)

    echo "✅ Lambda $LAMBDA_NAME zipped successfully!"
}

# Function to check if a Lambda function exists
function check_lambda_exists {
  LAMBDA_NAME=$1
  aws lambda get-function --function-name $LAMBDA_NAME --endpoint-url=$LOCALSTACK_ENDPOINT 2>/dev/null
}

# Function to deploy Lambda function if code has changed
function deploy_lambda {
  LAMBDA_NAME=$1
  LAMBDA_PATH=$2
  ZIP_FILE="${LAMBDA_NAME}.zip"

  # Create a hash of the current code
  HASH_NEW=$(zip -r -q "$ZIP_FILE" "$LAMBDA_PATH" && md5sum "$ZIP_FILE" | awk '{print $1}')
  
  # Check if Lambda function exists
  LAMBDA_EXISTS=$(check_lambda_exists "$LAMBDA_NAME")
  
  if [ -n "$LAMBDA_EXISTS" ]; then
    # Compare hash of new and existing Lambda code
    HASH_EXISTING=$(aws lambda get-function --function-name $LAMBDA_NAME --endpoint-url=$LOCALSTACK_ENDPOINT --query "Configuration.CodeSha256" --output text)
    
    if [ "$HASH_NEW" != "$HASH_EXISTING" ]; then
      echo "✅ Lambda function '$LAMBDA_NAME' exists, but code has changed. Updating deployment..."
      aws lambda update-function-code \
        --function-name $LAMBDA_NAME \
        --zip-file fileb://"$ZIP_FILE" \
        --endpoint-url=$LOCALSTACK_ENDPOINT > /dev/null
      echo "✅ Lambda function '$LAMBDA_NAME' updated."
    else
      echo "✅ Lambda function '$LAMBDA_NAME' already has the latest code. Skipping update."
    fi
  else
    # Deploy Lambda if it doesn't exist
    echo "✅ Lambda function '$LAMBDA_NAME' does not exist. Deploying..."
    aws lambda create-function \
      --function-name $LAMBDA_NAME \
      --runtime python3.9 \
      --role arn:aws:iam::000000000000:role/$ROLE_NAME \
      --handler cluster_handler.lambda_handler \
      --zip-file fileb://"$ZIP_FILE" \
      --endpoint-url=$LOCALSTACK_ENDPOINT > /dev/null
    echo "✅ Lambda function '$LAMBDA_NAME' deployed."
  fi
}

## test
test_lambda "find-guten-lambda" "guten_lambda"
if [ $? -ne 0 ]; then
    echo "❌ Build aborted due to lambda test failure."
    exit 1
fi
test_lambda "embed-lambda"      "embed_lambda"
if [ $? -ne 0 ]; then
    echo "❌ Build aborted due to lambda test failure."
    exit 1
fi
test_lambda "cluster-lambda"    "cluster_lambda"
if [ $? -ne 0 ]; then
    echo "❌ Build aborted due to lambda test failure."
    exit 1
fi
test_lambda "predict-lambda"    "predict_lambda"
if [ $? -ne 0 ]; then
    echo "❌ Build aborted due to lambda test failure."
    exit 1
fi

## zip
zip_lambdas "find-guten-lambda" "guten_lambda"
zip_lambdas "embed-lambda"      "embed_lambda"
zip_lambdas "cluster-lambda"    "cluster_lambda"
zip_lambdas "predict-lambda"    "predict_lambda"

## deploy
deploy_lambda "find-guten-lambda" "guten_lambda/handler.py"
deploy_lambda "insert-lambda"     "embed_lambda/handler.py"
deploy_lambda "cluster-lambda"    "cluster_lambda/cluster_handler.py"
deploy_lambda "predict-lambda"    "predict_lambda/predict_handler.py"

# 4. Create an API Gateway
API_ID=$(aws apigateway create-rest-api --name "local-api" --endpoint-url=$LOCALSTACK_ENDPOINT --query "id" --output text)
echo "✅ API Gateway created with ID: $API_ID"

# 5. Create Endpoints for each lambda
create_resource() {
  RESOURCE_NAME=$1
  RESOURCE_PATH=$2
  PARENT_ID=$(aws apigateway get-resources --rest-api-id $API_ID --endpoint-url=$LOCALSTACK_ENDPOINT --query "items[0].id" --output text)
  
  RESOURCE_ID=$(aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $PARENT_ID \
    --path-part $RESOURCE_PATH \
    --endpoint-url=$LOCALSTACK_ENDPOINT \
    --query "id" --output text)
  
  echo "✅ Resource '$RESOURCE_PATH' created with ID: $RESOURCE_ID"
  
  # Create a Method for each Resource (POST)
  aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $RESOURCE_ID \
    --http-method POST \
    --authorization-type "NONE" \
    --endpoint-url=$LOCALSTACK_ENDPOINT

  # Integrate Lambda with the Method
  aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $RESOURCE_ID \
    --http-method POST \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:000000000000:function:$2/invocations \
    --endpoint-url=$LOCALSTACK_ENDPOINT
  
  echo "✅ API Gateway method for '$RESOURCE_PATH' integrated with Lambda function."
}

# Create Resources and Methods for each Lambda
create_resource "find-guten-lambda" "find-guten-lambda"
create_resource "insert-lambda" "insert-lambda"
create_resource "cluster-lambda" "cluster-lambda"
create_resource "predict-lambda" "predict-lambda"

# 6. Deploy API Gateway
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name dev \
  --endpoint-url=$LOCALSTACK_ENDPOINT > /dev/null
echo "✅ API Gateway deployed."

# 5. Deploy API Gateway
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name dev \
  --endpoint-url=$LOCALSTACK_ENDPOINT > /dev/null
echo "✅ API Gateway deployed."

# 6. List All Services
aws s3 ls --endpoint-url=$LOCALSTACK_ENDPOINT
aws lambda list-functions --endpoint-url=$LOCALSTACK_ENDPOINT
aws apigateway get-rest-apis --endpoint-url=$LOCALSTACK_ENDPOINT

echo "🎉 LocalStack setup complete!"
