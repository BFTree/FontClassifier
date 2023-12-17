def string_to_unicode(input_str, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for char in input_str:
            unicode_code = ord(char)
            file.write(f"{unicode_code}\n")


# 例子
input_string = input("请输入字符串: ")
output_filename = input("请输入输出文件名: ")

string_to_unicode(input_string, output_filename)
print(f"Unicode码已保存到文件 {output_filename}")
