#include <windows.h>//计算机中的头文件
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//wchar_t:在windows下是Unicode16编码，也就是俗称宽字节。char:当然就是指一个字节，在windows下面默认是gbk编码的。
char* WcharToChar(const wchar_t* wp)//Wchar转char
{
	char *m_char;
	int len = WideCharToMultiByte(CP_ACP, 0, wp, wcslen(wp), NULL, 0, NULL, NULL);
	m_char = new char[len + 1];
	WideCharToMultiByte(CP_ACP, 0, wp, wcslen(wp), m_char, len, NULL, NULL);
	m_char[len] = '\0';
	return m_char;
}

wchar_t* CharToWchar(const char* c)//char转Wchar
{
	wchar_t *m_wchar;
	int len = MultiByteToWideChar(CP_ACP, 0, c, strlen(c), NULL, 0);
	m_wchar = new wchar_t[len + 1];
	MultiByteToWideChar(CP_ACP, 0, c, strlen(c), m_wchar, len);
	m_wchar[len] = '\0';
	return m_wchar;
}

wchar_t* StringToWchar(const string& s)//String转Wchar
{
	const char* p = s.c_str();
	return CharToWchar(p);
}

#define training
#define Test

int main()
{
	const int image_cols = 16;//宽
	const int image_rows = 8;//高

	const string fileform = "*.jpg";//文件形式
	const string perfileReadPath = "charSamples";//每个文件可读路径

	const int sample_mun_perclass = 2000;//训练字符每类数量
	const int class_mun = 12;//训练字符类数

	string  fileReadName,fileReadPath;//文件可读名字，文件可读路径
	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = { { 0 } };//每一行一个训练样本
	float labels[class_mun*sample_mun_perclass][class_mun] = { { 0 } };//训练样本标签

#ifndef training
	char temp[256];
	for (int i = 0; i <= class_mun - 1; ++i)//不同类
	{
		//读取每个类文件夹下所有图像
		int j = 0;//每一类读取图像个数计数
		sprintf_s(temp, 256,"%d", i);
		fileReadPath = perfileReadPath + "/" + temp + "/" + fileform;
		cout << "文件夹" << i << endl;
		HANDLE hFile;//句柄，是windows用来表示对象的
		LPCTSTR lpFileName = StringToWchar(fileReadPath);//指定搜索目录和文件类型，如搜索d盘的音频文件可以是"D:\\*.mp3"。LPCTSTR用来表示你的字符是否使用UNICODE, 如果你的程序定义了UNICODE或者其他相关的宏，那么这个字符或者字符串将被作为UNICODE字符串，否则就是标准的ANSI字符串。
		WIN32_FIND_DATA pNextInfo;  //搜索得到的文件信息将储存在pNextInfo中;在用findfirst()和findnext()函数去查找磁盘文件时经常使用的一个数据结构WIN32_FIND_DATA的成员变量里包含了以上所有的文件属性，因此可以通过这个结构作为获取和更改文件属性的手段。
		hFile = FindFirstFile(lpFileName, &pNextInfo);//请注意是 &pNextInfo,不是 pNextInfo;FindFirstFile()获得指定目录的第一个文件
		if (hFile == INVALID_HANDLE_VALUE)//INVALID_HANDLE_VALUE类似与指针里的NULL，如果将指针释放后，应该立即将指针赋为NULL，否则出现野指针；同理，句柄执行closehandle后，应该立即将句柄置为INVALID_HANDLE_VALUE，即让句柄失效。
		{
			exit(-1);//搜索失败，调用失败 返回为INVALID_HANDLE_VALUE(即-1)
		}
		//do-while循环读取
		do
		{
			if (pNextInfo.cFileName[0] == '.')//过滤.和..//长文件名不能是.开头
				continue;
			j++;//读取一张图片
			//wcout<<pNextInfo.cFileName<<endl;//输出宽字符
			printf("%s\n", WcharToChar(pNextInfo.cFileName));
			//对读入的图片进行处理
			Mat srcImage = imread(perfileReadPath + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName), CV_LOAD_IMAGE_GRAYSCALE);
			Mat resizeImage;
			Mat trainImage;

			resize(srcImage, resizeImage, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//使用像素关系重采样。当图像缩小时候，该方法可以避免波纹出现
			threshold(resizeImage, trainImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			for (int k = 0; k<image_rows*image_cols; ++k)
			{
				trainingData[i*sample_mun_perclass + (j - 1)][k] = (float)trainImage.data[k];//trainImage.data描述了一个一维数组，包含以RGBA顺序的数据，数据使用0至255（包含）的整数表示
				//trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.at<unsigned char>((int)k/8,(int)k%8);//(float)train_image.data[k];
				//cout<<trainingData[i*sample_mun_perclass+(j-1)][k] <<" "<< (float)trainImage.at<unsigned char>(k/8,k%8)<<endl;
			}
		} while (FindNextFile(hFile, &pNextInfo) && j<sample_mun_perclass);//如果设置读入的图片数量，则以设置的为准，如果图片不够，则读取文件夹下所有图片

	}
	// Set up training data Mat
	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout << "trainingDataMat——OK！" << endl;

	// Set up label data 
	for (int i = 0; i <= class_mun - 1; ++i)
	{
		for (int j = 0; j <= sample_mun_perclass - 1; ++j)
		{
			for (int k = 0; k<class_mun; ++k)
			{
				if (k == i)
					labels[i*sample_mun_perclass + j][k] = 1;
				else labels[i*sample_mun_perclass + j][k] = 0;
			}
		}
	}
	Mat labelsMat(class_mun*sample_mun_perclass, class_mun, CV_32FC1, labels);
	cout << "labelsMat:" << endl;
	cout << labelsMat << endl;
	cout << "labelsMat——OK！" << endl;

	//训练代码
	cout << "training start...." << endl;
	CvANN_MLP bp;//人工神经网络：ANN，多层感知器MLP
	// Set up BPNetwork's parameters
	CvANN_MLP_TrainParams params;//神经网络的参数
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;//神经网络训练方法：BACKPROP表示使用back-propagation的训练方法，RPROP即最简单的propagation训练方法。
	params.bp_dw_scale = 0.001;//学习效率
	params.bp_moment_scale = 0.1;//动量参数：提供了一些惯性，使之平滑权值的随机波动。
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001); //设置结束条件：表示训练迭代的终止条件，默认为迭代次数（大于1000）和权值变化率（小于0.01）
	//params.train_method=CvANN_MLP_TrainParams::RPROP;
	//params.rp_dw0 = 0.1;
	//params.rp_dw_plus = 1.2;
	//params.rp_dw_minus = 0.5;
	//params.rp_dw_min = FLT_EPSILON;
	//params.rp_dw_max = 50.;

	//Setup the BPNetwork
	Mat layerSizes = (Mat_<int>(1, 5) << image_rows*image_cols, 128, 128, 128, class_mun);//层的结构
	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1.0, 1.0);//CvANN_MLP::SIGMOID_SYM，选用sigmoid作为激励函数，即上次所说的S形函数（包括单极性S形函数和双极性S形函数）
	//CvANN_MLP::GAUSSIAN：GAUSS函数
	//CvANN_MLP::IDENTITY：阶跃函数
	cout << "training...." << endl;
	bp.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
	bp.save("../bpcharModel1119.xml"); //save classifier
	cout << "training finish...bpModel1.xml saved " << endl;
#endif

#ifdef Test
	//测试神经网络
	cout << "测试：" << endl;
	CvANN_MLP bp;
	//bp.load("C:\\Users\\Administrator\\Desktop\\BP_ANN\\bpcharModel.xml");
	bp.load("../bpcharModel1119.xml");
	char buf[100];
	for (int i = 0;; i++)
	{
		sprintf_s(buf,100,"%s%d.jpg","C:\\Users\\Administrator\\Desktop\\图像处理算法开源程序设计\\BP_ANN\\测试库\\",i);
		Mat test_image = imread(buf, CV_LOAD_IMAGE_GRAYSCALE);
		Mat test_temp;
		resize(test_image, test_temp, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
		threshold(test_temp, test_temp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		Mat_<float>sampleMat(1, image_rows*image_cols);
		for (int i = 0; i<image_rows*image_cols; ++i)
		{
			sampleMat.at<float>(0, i) = (float)test_temp.at<uchar>(i / image_cols, i % image_cols);
		}
		Mat responseMat;
		bp.predict(sampleMat, responseMat);
		Point maxLoc;
		double maxVal = 0;
		minMaxLoc(responseMat, NULL, &maxVal, NULL, &maxLoc);//minMaxLoc寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置
		maxVal*=0.9;
		cout << "识别结果：" << maxLoc.x << "  相似度:" << maxVal * 100 << "%" << endl;
		imshow("test_image", test_image);
		waitKey(0);
	}
#endif
	return 0;
}