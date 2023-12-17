#include <vector>

std::vector<unsigned short> unicode;

void getSentenceUnicode(const char *unicodepath)
{
    // 打开文件
    std::ifstream file(filename);

    // 检查文件是否成功打开
    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << filename << std::endl;
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
            numbers.push_back(number);
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
    std::cout << "从文件中读取的数字: ";
    for (int number : numbers)
    {
        std::cout << number << " ";
    }
    std::cout << std::endl;

    return 0; // 返回成功码
}

void generateSentence(const char *fontfilepath)
{
    /* 加载字体（.ttf）文件 */
    long int size = 0;
    unsigned char *fontBuffer = NULL;

    FILE *fontFile = fopen(fontfilepath, "rb");
    if (fontFile == NULL)
    {
        printf("Can not open font file!\n");
        return 0;
    }
    fseek(fontFile, 0, SEEK_END); /* 设置文件指针到文件尾，基于文件尾偏移0字节 */
    size = ftell(fontFile);       /* 获取文件大小（文件尾 - 文件头  单位：字节） */
    fseek(fontFile, 0, SEEK_SET); /* 重新设置文件指针到文件头 */

    fontBuffer = calloc(size, sizeof(unsigned char));
    fread(fontBuffer, size, 1, fontFile);
    fclose(fontFile);

    /* 初始化字体 */
    stbtt_fontinfo info;
    if (!stbtt_InitFont(&info, fontBuffer, 0))
    {
        printf("stb init font failed\n");
    }

    /* 创建位图 */
    int bitmap_w = 512; /* 位图的宽 */
    int bitmap_h = 128; /* 位图的高 */
    unsigned char *bitmap = calloc(bitmap_w * bitmap_h, sizeof(unsigned char));

    /* "STB"的 unicode 编码 */
    char word[20] = {0x53, 0x54, 0x42};

    /* 计算字体缩放 */
    float pixels = 64.0;                                    /* 字体大小（字号） */
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
    for (int i = 0; i < strlen(word); ++i)
    {
        /**
         * 获取水平方向上的度量
         * advanceWidth：字宽；
         * leftSideBearing：左侧位置；
         */
        int advanceWidth = 0;
        int leftSideBearing = 0;
        stbtt_GetCodepointHMetrics(&info, word[i], &advanceWidth, &leftSideBearing);

        /* 获取字符的边框（边界） */
        int c_x1, c_y1, c_x2, c_y2;
        stbtt_GetCodepointBitmapBox(&info, word[i], scale, scale, &c_x1, &c_y1, &c_x2, &c_y2);

        /* 计算位图的y (不同字符的高度不同） */
        int y = ascent + c_y1;

        /* 渲染字符 */
        int byteOffset = x + roundf(leftSideBearing * scale) + (y * bitmap_w);
        stbtt_MakeCodepointBitmap(&info, bitmap + byteOffset, c_x2 - c_x1, c_y2 - c_y1, bitmap_w, scale, scale, word[i]);

        /* 调整x */
        x += roundf(advanceWidth * scale);

        /* 调整字距 */
        int kern;
        kern = stbtt_GetCodepointKernAdvance(&info, word[i], word[i + 1]);
        x += roundf(kern * scale);
    }

    /* 将位图数据保存到1通道的png图像中 */
    stbi_write_png("STB.png", bitmap_w, bitmap_h, 1, bitmap, bitmap_w);

    free(fontBuffer);
    free(bitmap);
}