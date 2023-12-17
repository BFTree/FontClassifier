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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" /* http://nothings.org/stb/stb_image_write.h */

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h" /* http://nothings.org/stb/stb_truetype.h */

char32_t get_char_from_unicode(unsigned int unicode)
{
    return static_cast<char32_t>(unicode);
}

/*  unicode 编码 */
std::vector<int> nums;

void loadGB2312()
{
    std::ifstream file("gb2312.txt");
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            // int temp = stoi(line);
            nums.push_back(stoi(line, nullptr, 16));
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }
    return;
}

bool isDirectoryExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

void printFont(const char *fontfilename, const char *targetPath)
{
    /* 加载字体（.ttf）文件 */
    long int size = 0;
    unsigned char *fontBuffer = NULL;

    FILE *fontFile = fopen(fontfilename, "rb");
    if (fontFile == NULL)
    {
        // printf("Can not open font file!\n");
        std::cout << "Can not open font file! fontname: " << fontfilename << std::endl;
        return;
    }
    fseek(fontFile, 0, SEEK_END); /* 设置文件指针到文件尾，基于文件尾偏移0字节 */
    size = ftell(fontFile) + 10;  /* 获取文件大小（文件尾 - 文件头  单位：字节） */
    fseek(fontFile, 0, SEEK_SET); /* 重新设置文件指针到文件头 */

    fontBuffer = (unsigned char *)calloc(size, sizeof(unsigned char));
    fread(fontBuffer, size, 1, fontFile);
    fclose(fontFile);

    /* 初始化字体 */
    stbtt_fontinfo info;
    if (!stbtt_InitFont(&info, fontBuffer, 0))
    {
        printf("stb init font failed\n");
        free(fontBuffer);
        return;
    }
    std::string s = std::string(fontfilename);
    std::string fontname = s.substr(s.find_last_of("/") + 1, s.find_last_of(".") - s.find_last_of("/") - 1);

    std::string newdir = std::string(targetPath) + std::string("/") + fontname;
    std::cout << newdir << std::endl;
    if(!isDirectoryExists(newdir))
    {
        
        int r = mkdir(newdir.c_str(), 0700);
        if (r != 0)
        {
            std::cout << "Can not mkdir! fontname: " << fontfilename << std::endl;
            free(fontBuffer);
            return;
        }
    }
    /* 计算字体缩放 */
    float pixels = 100.0;                                   /* 字体大小（字号） */
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

    /* 循环加载word中每个字符 */
    for (int i = 0; i < nums.size(); ++i)
    {
        unsigned short word = nums[i];
        int r = stbtt_FindGlyphIndex(&info, word);
        if (r == 0)
        {
            std::cout << "No corresponding word found! word unicode: " << word << " fontname: " << fontfilename << std::endl;
            continue;
        }

        int x = 0; /*位图的x*/

        /**
         * 获取水平方向上的度量
         * advanceWidth：字宽；
         * leftSideBearing：左侧位置；
         */
        int advanceWidth = 0;
        int leftSideBearing = 0;
        stbtt_GetCodepointHMetrics(&info, word, &advanceWidth, &leftSideBearing);

        /* 获取字符的边框（边界） */
        int c_x1, c_y1, c_x2, c_y2;
        stbtt_GetCodepointBitmapBox(&info, word, scale, scale, &c_x1, &c_y1, &c_x2, &c_y2);

        /* 创建位图 */
        int bitmap_w = 128 > c_x2 + 10 ? 128 : c_x2 + 10; /* 位图的宽 */
        int bitmap_h = 128 > c_y2 + 10 ? 128 : c_y2 + 10; /* 位图的高 */
        unsigned char *bitmap = (unsigned char *)calloc(bitmap_w * bitmap_h, sizeof(unsigned char));

        /* 计算位图的y (不同字符的高度不同） */
        int y = ascent + c_y1;
        /* 渲染字符 */
        int byteOffset = x + roundf(leftSideBearing * scale) + (y * bitmap_w);
        stbtt_MakeCodepointBitmap(&info, bitmap, c_x2 - c_x1, c_y2 - c_y1, bitmap_w, scale, scale, word);

        /* 调整x */
        // x += roundf(advanceWidth * scale);

        /* 调整字距 */
        // int kern;
        // kern = stbtt_GetCodepointKernAdvance(&info, word[i], word[i + 1]);
        // x += roundf(kern * scale);

        /* 将位图数据保存到1通道的png图像中 */
        std::string filename = newdir + std::string("/") + std::to_string(word) + std::string(".png");
        stbi_write_png(filename.c_str(), bitmap_w, bitmap_h, 1, bitmap, bitmap_w);
        std::cout << "释放指针： word unicode: " << word << std::endl;
        free(bitmap);
        std::cout << "释放完毕： word unicode: " << word << std::endl;
    }
    std::cout << "释放字体文件拷贝指针： font: " << fontfilename << std::endl;
    free(fontBuffer);
    return;
}

int main(int argc, const char *argv[])
{

    if (argc < 3)
        return 0;
    loadGB2312();

    /* 遍历指定文件夹获取ttf或者TTF文件 */
    std::vector<std::string> ttf_files;
    std::string folder_path = argv[1];
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
            std::string tmp = std::string(argv[1]) + std::string("/") + file;
            printFont(tmp.c_str(), argv[2]);
        }catch (...) {
            continue;
        }   

        
    }
    return 0;
}