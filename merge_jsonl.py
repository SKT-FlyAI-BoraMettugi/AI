import os
import json

def merge_jsonl_files(input_folder, output_file):
    """
    여러 JSONL 파일을 하나의 JSONL 파일로 병합하는 함수.
    
    :param input_folder: JSONL 파일들이 저장된 폴더 경로
    :param output_file: 병합된 JSONL 파일 경로
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in sorted(os.listdir(input_folder)):  # 파일 정렬하여 병합
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(input_folder, file_name)
                
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        json_obj = json.loads(line.strip())  # JSONL 한 줄씩 파싱
                        json.dump(json_obj, outfile, ensure_ascii=False)  # JSONL 형식 유지
                        outfile.write('\n')  # 줄바꿈 추가

                print(f'✅ {file_name} 병합 완료')

    print(f'\n🎉 모든 JSONL 파일이 "{output_file}"로 병합되었습니다.')

# 사용 예시
input_folder = "colab_json_outputs"  # JSONL 파일들이 있는 폴더 경로
output_file = "merged_output.jsonl"  # 병합된 JSONL 파일명
merge_jsonl_files(input_folder, output_file)