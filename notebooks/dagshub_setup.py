import mlflow
import dagshub
mlflow.set_tracking_uri("https://dagshub.com/jaymavani16/emotion_detection_model_serving.mlflow")
dagshub.init(repo_owner='jaymavani16', repo_name='emotion_detection_model_serving', mlflow=True)

with mlflow.start_run():
  # Your training code here...
  mlflow.log_metric('accuracy', 42)
  mlflow.log_param('Param name', 'Value')