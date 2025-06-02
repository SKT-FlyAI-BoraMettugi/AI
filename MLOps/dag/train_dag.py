import sys

sys.path.append("/opt/airflow/train")   # Airflow 작업용 모듈이 위치한 경로를 sys.path에 추가

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import update_learn

with DAG(
    dag_id='model_training',     # DAG의 ID
    description='Nolli MLOps 구현', # DAG에 대한 설명
    schedule_interval='@monthly',   # DAG 실행 주기를 매월 1회로 설정
    start_date=datetime(2025, 1, 1),  # DAG가 처음 실행될 날짜 지정
    catchup=False,  # 이전 실행을 모두 수행하지 않고 최신 스케줄만 실행하도록 설정
    max_active_runs=1,  # 동시에 실행 가능한 DAG 인스턴스 수를 1개로 제한
    concurrency=1,  # 동시에 실행 가능한 task 수를 1개로 제한
) as dag:
    task = PythonOperator(  # PythonOperator를 사용해 Python 함수를 실행하는 태스크 정의
        task_id='train_model_task', # 태스크의 고유 ID 설정
        python_callable=update_learn.update_learning,   # 실행할 함수 지정
        provide_context=True,   # Airflow 실행 컨텍스트를 함수에 전달할지 여부
    )
