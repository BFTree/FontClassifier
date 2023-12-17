#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <locale>
#include <codecvt>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <algorithm>
#include <random>
#include <sstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" /* http://nothings.org/stb/stb_image_write.h */

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h" /* http://nothings.org/stb/stb_truetype.h */

/* 语句的 unicode 编码 */
std::vector<unsigned short> unicode;

int getSentenceUnicode(const char *unicodepath)
{
    // 打开文件
    std::ifstream file(unicodepath);

    // 检查文件是否成功打开
    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << unicodepath << std::endl;
        return 1; // 返回错误码
    }

    // 逐行读取文件内容，并将每行的数字保存到vector中
    std::string line;
    while (std::getline(file, line))
    {
        try
        {
            // 将字符串转换为整数，并将其添加到vector中
            int number = std::stoi(line);
            unicode.push_back(number);
        }
        catch (const std::invalid_argument &e)
        {
            std::cerr << "无效的数字格式: " << line << std::endl;
        }
        catch (const std::out_of_range &e)
        {
            std::cerr << "数字超出范围: " << line << std::endl;
        }
    }

    // 关闭文件
    file.close();

    // 输出vector中的数字
    std::cout << "从文件中读取的数字和对应字符: " << std::endl;
    for (int number : unicode)
    {
        wchar_t unicodeChar = static_cast<wchar_t>(number);
        std::cout << number << " " << unicodeChar << std::endl;
    }
    unicode.push_back(0);
    std::cout << std::endl;

    return 0; // 返回成功码
}

bool isDirectoryExists(const std::string &path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

void generateSentence(const char *fontfilepath, const char *targetpath)
{
    /* 加载字体（.ttf）文件 */
    long int size = 0;
    unsigned char *fontBuffer = NULL;

    FILE *fontFile = fopen(fontfilepath, "rb");
    if (fontFile == NULL)
    {
        printf("Can not open font file!\n");
        return;
    }
    fseek(fontFile, 0, SEEK_END); /* 设置文件指针到文件尾，基于文件尾偏移0字节 */
    size = ftell(fontFile);       /* 获取文件大小（文件尾 - 文件头  单位：字节） */
    fseek(fontFile, 0, SEEK_SET); /* 重新设置文件指针到文件头 */

    fontBuffer = (unsigned char *)calloc(size, sizeof(unsigned char));
    fread(fontBuffer, size, 1, fontFile);
    fclose(fontFile);

    /* 初始化字体 */
    stbtt_fontinfo info;
    if (!stbtt_InitFont(&info, fontBuffer, 0))
    {
        printf("stb init font failed\n");
    }

    /* 创建位图 */
    int bitmap_w = 128 * unicode.size(); /* 位图的宽 */
    int bitmap_h = 128;                  /* 位图的高 */
    unsigned char *bitmap = (unsigned char *)calloc(bitmap_w * bitmap_h, sizeof(unsigned char));

    /* 计算字体缩放 */
    float pixels = 120.0;                                   /* 字体大小（字号） */
    float scale = stbtt_ScaleForPixelHeight(&info, pixels); /* scale = pixels / (ascent - descent) */

    /**
     * 获取垂直方向上的度量
     * ascent：字体从基线到顶部的高度；
     * descent：基线到底部的高度，通常为负值；
     * lineGap：两个字体之间的间距；
     * 行间距为：ascent - descent + lineGap。
     */
    int ascent = 0;
    int descent = 0;
    int lineGap = 0;
    stbtt_GetFontVMetrics(&info, &ascent, &descent, &lineGap);

    /* 根据缩放调整字高 */
    ascent = roundf(ascent * scale);
    descent = roundf(descent * scale);

    int x = 0; /*位图的x*/

    /* 循环加载word中每个字符 */
    for (int i = 0; i < unicode.size() - 1; ++i)
    {
        int r = stbtt_FindGlyphIndex(&info, unicode[i]);
        if (r == 0)
        {
            std::cout << "No corresponding word found! word unicode: " << unicode[i] << " fontpath: " << fontfilepath << std::endl;
            continue;
        }

        /**
         * 获取水平方向上的度量
         * advanceWidth：字宽；
         * leftSideBearing：左侧位置；
         */
        int advanceWidth = 0;
        int leftSideBearing = 0;
        stbtt_GetCodepointHMetrics(&info, unicode[i], &advanceWidth, &leftSideBearing);

        /* 获取字符的边框（边界） */
        int c_x1, c_y1, c_x2, c_y2;
        stbtt_GetCodepointBitmapBox(&info, unicode[i], scale, scale, &c_x1, &c_y1, &c_x2, &c_y2);

        /* 计算位图的y (不同字符的高度不同） */
        int y = ascent + c_y1;

        /* 渲染字符 */
        int byteOffset = x + roundf(leftSideBearing * scale) + (y * bitmap_w);
        stbtt_MakeCodepointBitmap(&info, bitmap + byteOffset, c_x2 - c_x1, c_y2 - c_y1, bitmap_w, scale, scale, unicode[i]);

        /* 调整x */
        x += roundf(advanceWidth * scale);

        /* 调整字距 */
        int kern;
        kern = stbtt_GetCodepointKernAdvance(&info, unicode[i], unicode[i + 1]);
        x += roundf(kern * scale);
    }

    std::string s = std::string(fontfilepath);
    std::string fontname = s.substr(s.find_last_of("/") + 1, s.find_last_of(".") - s.find_last_of("/") - 1);

    std::string newdir = std::string(targetpath) + std::string("/") + fontname;
    std::cout << newdir << std::endl;
    if (!isDirectoryExists(newdir))
    {

        int r = mkdir(newdir.c_str(), 0700);
        if (r != 0)
        {
            std::cout << "Can not mkdir! fontname: " << fontfilepath << std::endl;
            free(fontBuffer);
            return;
        }
    }
    std::wstring unicodeString;

    // 将Unicode码点数组转换为wstring
    for (unsigned short codePoint : unicode)
    {
        if (codePoint == 0)
            break;
        unicodeString += static_cast<wchar_t>(codePoint);
    }
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::string str = converter.to_bytes(unicodeString);
    std::string targetname = newdir + std::string("/");
    str = str + std::string(".png");
    targetname += str;
    std::cout << targetname.c_str() << std::endl;
    /* 将位图数据保存到1通道的png图像中 */
    stbi_write_png(targetname.c_str(), bitmap_w, bitmap_h, 1, bitmap, bitmap_w);

    free(fontBuffer);
    free(bitmap);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "usage: ./generateSentence sentenceunicodepath fontdir targetPath" << std::endl;
        return 0;
    }
    if (getSentenceUnicode(argv[1]) != 0)
    {
        return 1;
    }

    /* 遍历指定文件夹获取ttf或者TTF文件 */
    std::vector<std::string> ttf_files;
    std::string folder_path = argv[2];
    DIR *dir = opendir(folder_path.c_str());
    if (dir == NULL)
    {
        std::cout << "Error opening directory" << std::endl;
        return 1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        std::string filename = entry->d_name;
        if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".ttf" || filename.size() >= 4 && filename.substr(filename.size() - 4) == ".TTF")
        {
            ttf_files.push_back(filename);
        }
    }

    closedir(dir);

    if (ttf_files.empty())
    {
        std::cout << "No ttf or TTF files found in directory" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "The following ttf and TTF files were found in directory:" << std::endl;
        for (const auto &file : ttf_files)
        {
            std::cout << file << std::endl;
        }
    }
    std::random_device rd;
    std::mt19937 g(rd());

    // 使用 std::shuffle 对 vector 进行随机重排
    std::shuffle(ttf_files.begin(), ttf_files.end(), g);

    // Print the list of TTF files
    for (const auto &file : ttf_files)
    {
        try
        {
            std::cout << file << std::endl;
            std::string tmp = std::string(argv[2]) + std::string("/") + file;
            generateSentence(tmp.c_str(), argv[3]);
        }
        catch (...)
        {
            continue;
        }
    }
    return 0;
}