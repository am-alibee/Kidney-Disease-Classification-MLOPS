# Kidney-Disease-Classification-MLOPS

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. Update app.py


## Deployment

1. Build docker image of the source code
2. Push your docker image to ECR
3. Launch your EC2
4. Push your image from ECR to EC2
5. Launch your docker image to EC2

### Policy

1. AmazonEC2ContainerRegistryFullAccess
2. AmazonEC2FullAccess