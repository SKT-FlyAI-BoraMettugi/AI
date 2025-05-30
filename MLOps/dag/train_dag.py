import sys

sys.path.append("/opt/airflow/train")

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import update_learn

with DAG(
    dag_id='model_training',
    description='Nolli MLOps 구현',
    schedule_interval='@monthly',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    concurrency=1,
) as dag:
    task = PythonOperator(
        task_id='train_model_task',
        python_callable=update_learn.update_learning,
        provide_context=True, 
    )
