#!/bin/bash

# 다운로드할 URL 및 파일명
URL="https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z"
FILENAME="ComfyUI_windows_portable_nvidia.7z"

# 다운로드 여부 선택
echo "ComfyUI를 다운로드하시겠습니까?"
echo "1. 네 (다운로드)"
echo "2. 아니요 (다운로드 안 함)"
read -p "선택 (1 또는 2): " download_choice

if [ "$download_choice" -eq 1 ]; then
    # 기존 파일 삭제 여부 확인 후 삭제
    if [ -f "$FILENAME" ]; then
        echo "기존의 $FILENAME 파일이 감지되었습니다. 삭제 후 다시 다운로드합니다."
        rm "$FILENAME"
    fi

    echo "ComfyUI 다운로드를 시작합니다..."

    # PowerShell에서 curl을 실행하면서 진행률을 실시간으로 출력
    powershell -NoProfile -ExecutionPolicy Bypass -Command "& {
        \$ProgressPreference = 'Continue';
        curl.exe -L --progress-bar '$URL' --output '$FILENAME'
    }"

    # 다운로드 완료 확인
    if [ ! -f "$FILENAME" ]; then
        echo "다운로드 실패! 스크립트를 종료합니다."
        exit 1
    fi

    echo "다운로드 완료: $FILENAME"
elif [ "$download_choice" -eq 2 ]; then
    echo "다운로드를 건너뜁니다."
else
    echo "잘못된 입력입니다. 스크립트를 종료합니다."
    exit 1
fi

echo "-----------------------"

# 압축 해제 여부 선택
echo "압축을 푸시겠습니까? (반디집 사용)"
echo "1. 네 (압축 풀기)"
echo "2. 아니요 (그대로 두기)"
read -p "선택 (1 또는 2): " extract_choice

if [ "$extract_choice" -eq 1 ]; then
    echo "반디집을 사용하여 압축을 풉니다..."

    # 반디집 실행 파일 경로 (고정 경로)
    BANDIZIP_CLI_PATH="C:\\Program Files\\Bandizip\\Bandizip.exe"

    # PowerShell을 사용하여 반디집으로 압축 해제 실행
    powershell -NoProfile -ExecutionPolicy Bypass -Command "& {
        & '$BANDIZIP_CLI_PATH' x -o:. '$FILENAME'
    }"

    if [ $? -eq 0 ]; then
        echo "압축 해제 완료!"
    else
        echo "압축 해제 중 오류 발생!"
    fi
elif [ "$extract_choice" -eq 2 ]; then
    echo "압축 풀기 건너뜀."
else
    echo "잘못된 입력입니다. 압축을 풀지 않고 종료합니다."
fi

# 다운로드할 URL 및 파일 경로 설정
CLIP_G_URL="https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/text_encoders/clip_g.safetensors?download=true"
CLIP_G_FILENAME="clip_g.safetensors"
CLIP_G_DEST_DIR=".\\ComfyUI_windows_portable\\ComfyUI\\models\\clip\\"

# # 다운로드 여부 선택
# echo "CLIP 모델 파일을 다운로드하시겠습니까?"
# echo "1. 네 (다운로드)"
# echo "2. 아니요 (다운로드 안 함)"
# read -p "선택 (1 또는 2): " clip_g_download_choice

# if [ "$clip_g_download_choice" -eq 1 ]; then
#     # 대상 디렉터리 존재 여부 확인
#     if [ ! -d "$CLIP_G_DEST_DIR" ]; then
#         echo "폴더가 존재하지 않습니다. 종료합니다."
#         exit 1
#     fi

#     # 기존 파일 삭제 여부 확인 후 삭제
#     if [ -f "$CLIP_G_DEST_DIR$CLIP_G_FILENAME" ]; then
#         echo "기존의 $CLIP_G_FILENAME 파일이 감지되었습니다. 삭제 후 다시 다운로드합니다."
#         rm "$CLIP_G_DEST_DIR$CLIP_G_FILENAME"
#     fi

#     echo "CLIP 모델 파일 다운로드를 시작합니다..."

#     # PowerShell에서 curl을 실행하면서 진행률을 실시간으로 출력
#     powershell -NoProfile -ExecutionPolicy Bypass -Command "& {
#         \$ProgressPreference = 'Continue';
#         curl.exe -L --progress-bar '$CLIP_G_URL' --output '$CLIP_G_DEST_DIR$CLIP_G_FILENAME'
#     }"

#     # 다운로드 성공 여부 확인
#     if [ $? -eq 0 ]; then
#         echo "다운로드가 완료되었습니다: ${CLIP_G_DEST_DIR}${CLIP_G_FILENAME}"
#     else
#         echo "다운로드에 실패했습니다."
#     fi
# else
#     echo "다운로드를 취소하였습니다."
# fi