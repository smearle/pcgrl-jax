import os

def create_message_box(text):
    # 텍스트를 줄 단위로 나눕니다.
    lines = text.split('\n')

    # 각 줄의 길이를 구하고, 가장 긴 줄의 길이를 찾습니다.
    max_length = max(len(line) for line in lines)

    # 메시지 박스의 길이를 가장 긴 줄의 길이에 맞춰서 설정합니다.
    box_length = max_length + 6  # ### 양쪽 3자리씩 차지함

    # 메시지 박스를 구성합니다.
    top_bottom_border = "#" * box_length
    middle_lines = [f"### {line.ljust(max_length)} ###" for line in lines]

    # 메시지 박스를 문자열로 반환
    return f"{top_bottom_border}\n" + "\n".join(middle_lines) + f"\n{top_bottom_border}"

def get_textfile_tail(log_path, tail: int = 60) -> str:
    with open(log_path, 'rb') as f:
        # Move to the end of the file
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        block_size = 1024
        data = []
        while file_size > 0 and len(data) < tail:
            if file_size - block_size > 0:
                f.seek(-block_size, os.SEEK_CUR)
            else:
                f.seek(0)
                block_size = file_size
            chunk = f.read(block_size).splitlines()
            data = chunk + data
            file_size -= block_size
            f.seek(file_size, os.SEEK_SET)

        # Trim the list to the last 'tail' lines
        if len(data) > tail:
            data = data[-tail:]
        logs = [line.decode('utf-8') for line in data]

    return '\n'.join(logs)

