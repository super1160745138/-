#include <windows.h>//������е�ͷ�ļ�
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//wchar_t:��windows����Unicode16���룬Ҳ�����׳ƿ��ֽڡ�char:��Ȼ����ָһ���ֽڣ���windows����Ĭ����gbk����ġ�
char* WcharToChar(const wchar_t* wp)//Wcharתchar
{
	char *m_char;
	int len = WideCharToMultiByte(CP_ACP, 0, wp, wcslen(wp), NULL, 0, NULL, NULL);
	m_char = new char[len + 1];
	WideCharToMultiByte(CP_ACP, 0, wp, wcslen(wp), m_char, len, NULL, NULL);
	m_char[len] = '\0';
	return m_char;
}

wchar_t* CharToWchar(const char* c)//charתWchar
{
	wchar_t *m_wchar;
	int len = MultiByteToWideChar(CP_ACP, 0, c, strlen(c), NULL, 0);
	m_wchar = new wchar_t[len + 1];
	MultiByteToWideChar(CP_ACP, 0, c, strlen(c), m_wchar, len);
	m_wchar[len] = '\0';
	return m_wchar;
}

wchar_t* StringToWchar(const string& s)//StringתWchar
{
	const char* p = s.c_str();
	return CharToWchar(p);
}

#define training
#define Test

int main()
{
	const int image_cols = 16;//��
	const int image_rows = 8;//��

	const string fileform = "*.jpg";//�ļ���ʽ
	const string perfileReadPath = "charSamples";//ÿ���ļ��ɶ�·��

	const int sample_mun_perclass = 2000;//ѵ���ַ�ÿ������
	const int class_mun = 12;//ѵ���ַ�����

	string  fileReadName,fileReadPath;//�ļ��ɶ����֣��ļ��ɶ�·��
	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = { { 0 } };//ÿһ��һ��ѵ������
	float labels[class_mun*sample_mun_perclass][class_mun] = { { 0 } };//ѵ��������ǩ

#ifndef training
	char temp[256];
	for (int i = 0; i <= class_mun - 1; ++i)//��ͬ��
	{
		//��ȡÿ�����ļ���������ͼ��
		int j = 0;//ÿһ���ȡͼ���������
		sprintf_s(temp, 256,"%d", i);
		fileReadPath = perfileReadPath + "/" + temp + "/" + fileform;
		cout << "�ļ���" << i << endl;
		HANDLE hFile;//�������windows������ʾ�����
		LPCTSTR lpFileName = StringToWchar(fileReadPath);//ָ������Ŀ¼���ļ����ͣ�������d�̵���Ƶ�ļ�������"D:\\*.mp3"��LPCTSTR������ʾ����ַ��Ƿ�ʹ��UNICODE, �����ĳ�������UNICODE����������صĺ꣬��ô����ַ������ַ���������ΪUNICODE�ַ�����������Ǳ�׼��ANSI�ַ�����
		WIN32_FIND_DATA pNextInfo;  //�����õ����ļ���Ϣ��������pNextInfo��;����findfirst()��findnext()����ȥ���Ҵ����ļ�ʱ����ʹ�õ�һ�����ݽṹWIN32_FIND_DATA�ĳ�Ա������������������е��ļ����ԣ���˿���ͨ������ṹ��Ϊ��ȡ�͸����ļ����Ե��ֶΡ�
		hFile = FindFirstFile(lpFileName, &pNextInfo);//��ע���� &pNextInfo,���� pNextInfo;FindFirstFile()���ָ��Ŀ¼�ĵ�һ���ļ�
		if (hFile == INVALID_HANDLE_VALUE)//INVALID_HANDLE_VALUE������ָ�����NULL�������ָ���ͷź�Ӧ��������ָ�븳ΪNULL���������Ұָ�룻ͬ�����ִ��closehandle��Ӧ�������������ΪINVALID_HANDLE_VALUE�����þ��ʧЧ��
		{
			exit(-1);//����ʧ�ܣ�����ʧ�� ����ΪINVALID_HANDLE_VALUE(��-1)
		}
		//do-whileѭ����ȡ
		do
		{
			if (pNextInfo.cFileName[0] == '.')//����.��..//���ļ���������.��ͷ
				continue;
			j++;//��ȡһ��ͼƬ
			//wcout<<pNextInfo.cFileName<<endl;//������ַ�
			printf("%s\n", WcharToChar(pNextInfo.cFileName));
			//�Զ����ͼƬ���д���
			Mat srcImage = imread(perfileReadPath + "/" + temp + "/" + WcharToChar(pNextInfo.cFileName), CV_LOAD_IMAGE_GRAYSCALE);
			Mat resizeImage;
			Mat trainImage;

			resize(srcImage, resizeImage, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���
			threshold(resizeImage, trainImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			for (int k = 0; k<image_rows*image_cols; ++k)
			{
				trainingData[i*sample_mun_perclass + (j - 1)][k] = (float)trainImage.data[k];//trainImage.data������һ��һά���飬������RGBA˳������ݣ�����ʹ��0��255����������������ʾ
				//trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.at<unsigned char>((int)k/8,(int)k%8);//(float)train_image.data[k];
				//cout<<trainingData[i*sample_mun_perclass+(j-1)][k] <<" "<< (float)trainImage.at<unsigned char>(k/8,k%8)<<endl;
			}
		} while (FindNextFile(hFile, &pNextInfo) && j<sample_mun_perclass);//������ö����ͼƬ�������������õ�Ϊ׼�����ͼƬ���������ȡ�ļ���������ͼƬ

	}
	// Set up training data Mat
	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout << "trainingDataMat����OK��" << endl;

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
	cout << "labelsMat����OK��" << endl;

	//ѵ������
	cout << "training start...." << endl;
	CvANN_MLP bp;//�˹������磺ANN������֪��MLP
	// Set up BPNetwork's parameters
	CvANN_MLP_TrainParams params;//������Ĳ���
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;//������ѵ��������BACKPROP��ʾʹ��back-propagation��ѵ��������RPROP����򵥵�propagationѵ��������
	params.bp_dw_scale = 0.001;//ѧϰЧ��
	params.bp_moment_scale = 0.1;//�����������ṩ��һЩ���ԣ�ʹ֮ƽ��Ȩֵ�����������
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001); //���ý�����������ʾѵ����������ֹ������Ĭ��Ϊ��������������1000����Ȩֵ�仯�ʣ�С��0.01��
	//params.train_method=CvANN_MLP_TrainParams::RPROP;
	//params.rp_dw0 = 0.1;
	//params.rp_dw_plus = 1.2;
	//params.rp_dw_minus = 0.5;
	//params.rp_dw_min = FLT_EPSILON;
	//params.rp_dw_max = 50.;

	//Setup the BPNetwork
	Mat layerSizes = (Mat_<int>(1, 5) << image_rows*image_cols, 128, 128, 128, class_mun);//��Ľṹ
	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1.0, 1.0);//CvANN_MLP::SIGMOID_SYM��ѡ��sigmoid��Ϊ�������������ϴ���˵��S�κ���������������S�κ�����˫����S�κ�����
	//CvANN_MLP::GAUSSIAN��GAUSS����
	//CvANN_MLP::IDENTITY����Ծ����
	cout << "training...." << endl;
	bp.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
	bp.save("../bpcharModel1119.xml"); //save classifier
	cout << "training finish...bpModel1.xml saved " << endl;
#endif

#ifdef Test
	//����������
	cout << "���ԣ�" << endl;
	CvANN_MLP bp;
	//bp.load("C:\\Users\\Administrator\\Desktop\\BP_ANN\\bpcharModel.xml");
	bp.load("../bpcharModel1119.xml");
	char buf[100];
	for (int i = 0;; i++)
	{
		sprintf_s(buf,100,"%s%d.jpg","C:\\Users\\Administrator\\Desktop\\ͼ�����㷨��Դ�������\\BP_ANN\\���Կ�\\",i);
		Mat test_image = imread(buf, CV_LOAD_IMAGE_GRAYSCALE);
		Mat test_temp;
		resize(test_image, test_temp, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���
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
		minMaxLoc(responseMat, NULL, &maxVal, NULL, &maxLoc);//minMaxLocѰ�Ҿ���(һά���鵱������,��Mat����) ����Сֵ�����ֵ��λ��
		maxVal*=0.9;
		cout << "ʶ������" << maxLoc.x << "  ���ƶ�:" << maxVal * 100 << "%" << endl;
		imshow("test_image", test_image);
		waitKey(0);
	}
#endif
	return 0;
}