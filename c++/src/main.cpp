//
// Example of training the model created by create_graph.py in a C++ program.
//

#include <iostream>
#include <string>
#include <typeinfo>
#include <opencv2/opencv.hpp>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor_slice.h"

#include "tensorflow/cc/saved_model/tag_constants.h"



void printTensor(const tensorflow::Tensor& tensor){
	int num_dimensions = tensor.shape().dims();
	for(int ii_dim=0; ii_dim<num_dimensions; ii_dim++) {
        std::cout << tensor.shape().dim_size(ii_dim) << ":";
    }
	std::cout << std::endl;

    auto p = tensor.flat<float>();
	const int N = p.size();
	std::cout << N << std::endl;
	for (int i=0; i<N; i++){
		std::cout << p(i) <<",";
	}
	std::cout << std::endl;
}

// CVMatのデータをTensorflowのTensorにコピーする
tensorflow::Status convertImagesTensorFromCVMat(const std::vector<cv::Mat>& images, 
												tensorflow::Tensor& imagesTensor,
												const std::vector<unsigned> inputShape){
	unsigned frames = inputShape[0];
	unsigned rows   = inputShape[1];
	unsigned cols   = inputShape[2];
	unsigned chans  = inputShape[3];

    float *p = imagesTensor.flat<float>().data();
    for (auto img : images){
		for (unsigned r=0; r<rows; r++){
			cv::Vec3f *ptr = img.ptr<cv::Vec3f>(r);
			std::memcpy(p+chans*r*cols, ptr, cols*chans*sizeof(float));
		}
        p += rows*cols*chans;
    }

    return tensorflow::Status::OK();
}

// TensorflowのTensorをCVMatのデータにコピーする
tensorflow::Status convertCVMatFromImagesTensor(tensorflow::Tensor& imagesTensor, 
												std::vector<cv::Mat>& images,
												const std::vector<unsigned>& inputShape){
	unsigned frames = inputShape[0];
	unsigned rows   = inputShape[1];
	unsigned cols   = inputShape[2];
	unsigned chans  = inputShape[3];
	//printTensor(tensor);
	float *p = imagesTensor.flat<float>().data();
	for (unsigned i=0; i<static_cast<unsigned>(frames); i++){
		cv::Mat img = cv::Mat::zeros(rows, cols, CV_32FC3);
		for (unsigned r=0; r<rows; r++){
			cv::Vec3f *ptr = img.ptr<cv::Vec3f>(r);
			std::memcpy(ptr, p+chans*r*cols, cols*chans*sizeof(float));
		}
		p += rows*cols*chans;
		images.push_back(img);
	}

	return tensorflow::Status::OK();
}

tensorflow::Status makeImageTensor(std::vector<tensorflow::Tensor>& imagesTensor){
	//tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,6,128,128,3}));
	tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,4,64,64,3}));
	imagesTensor.push_back(tensor);
	return tensorflow::Status::OK();
}

void printModelStatus(const tensorflow::SavedModelBundle& model){
	auto sig_map = model.GetSignatures();
	auto model_def = sig_map.at("serving_default");
	std::clog << "Model Signature";
	for (auto const& p : sig_map) {
	    std::clog << "key:" << p.first.c_str() << std::endl;
	}

	std::clog <<  "Model Input Nodes" << std::endl;
	for (auto const& p : model_def.inputs()) {
	    std::clog << "key: " << p.first.c_str() << " value: " << p.second.name().c_str() << std::endl;
	}

	std::clog << "Model Output Nodes" << std::endl;
	for (auto const& p : model_def.outputs()) {
	    std::clog << "key: " << p.first.c_str() << " value: " << p.second.name().c_str() << std::endl;
	}
}


int process(bool anomalyScore=false, 
			const std::string modelpath = "/content/VAE_movie/tmpdir/vae_movie/1/",
			const std::string inputmovie = "/content/data/movies/test/S_ball_room.mp4",
			const std::string outputmovie = "output.mov"
	){
	tensorflow::SavedModelBundle bundle;
	tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
	tensorflow::RunOptions run_options = tensorflow::RunOptions();


	session_options.config.mutable_gpu_options()->set_allow_growth(true);	
	//auto status = tensorflow::LoadSavedModel(session_options, run_options, path, {"serve"}, &bundle);
	auto status = tensorflow::LoadSavedModel(session_options, run_options, modelpath, {tensorflow::kSavedModelTagServe}, &bundle);
	if (!status.ok()){
		std::cerr << "Error in loading image from file" << std::endl;
		return -1;
	}
	else{
		std::clog << "Loaded correctly!" << std::endl;
	}
	printModelStatus(bundle);

	/*
	 *	動画用意
	*/
    cv::VideoCapture cap;
    cap.open(inputmovie.c_str());
    if (cap.isOpened() == false){
        std::cerr << "can't open file" << std::endl;
        return -1;
    }
    unsigned videoHeight = static_cast<unsigned>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    unsigned videoWidth  = static_cast<unsigned>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    unsigned videoFrames = static_cast<unsigned>(cap.get(cv::CAP_PROP_FRAME_COUNT));
	//unsigned fps = static_cast<unsigned>(cap.get(cv::CAP_PROP_FPS));
	unsigned fps = 30;
	unsigned ex = static_cast<unsigned>(cap.get(cv::CAP_PROP_FOURCC)); 

	unsigned modelFrames = 6;
	unsigned modelHeight = 64;
	unsigned modelWidth = 64;
	unsigned modelChannel = 3; 
	std::cout << "frames:"  << videoFrames << std::endl;

	double gokei = 0.0;
	double gokei2 = 0.0;
	int over = 0;
    int count = 0;
    std::vector<float> floatArray;

	cv::VideoWriter writer(outputmovie.c_str(), ex, fps, cv::Size(videoWidth, videoHeight), true); 

	// input/outpu name
	auto model_def = bundle.GetSignatures().at("serving_default");
	const std::string inputName   = model_def.inputs().at("input_1").name();; // input
	std::vector<std::string> outputNames = {model_def.outputs().at("output_1").name(), // mu,
						 					model_def.outputs().at("output_2").name()}; // sigma

	for (unsigned i=0; i<static_cast<int>(videoFrames/modelFrames); i++){
	//for (unsigned i=0; i<10; i++){
		std::cerr << i << std::endl;
		std::vector<cv::Mat> inputframes;

		// modelFrames分のフレームを取得
		int j=0;
		while (j<modelFrames){
		   	cv::Mat frame;
			cap >> frame;
			if (frame.rows == videoHeight && frame.cols == videoWidth){
				cv::Mat anImage(frame.rows, frame.cols, CV_32FC3);
				frame.convertTo(anImage, CV_32FC3, 1.0/255.0);
				inputframes.push_back(anImage);
				j++;
			}
		}
		// outputsを生成
		std::vector<cv::Mat> outputs;
		for (unsigned i=0; i<modelFrames; i++){
			cv::Mat zero = cv::Mat::zeros(videoHeight, videoWidth, CV_32FC3);
			outputs.push_back(zero);
		}
		size_t stride = 64;

		// modelHeight, modelWidthの矩形に切り出していく
		std::vector<unsigned> inputShape = {modelFrames, modelHeight, modelWidth, modelChannel};
		for (size_t y = 0; y <= videoHeight-modelHeight; y+=stride){
			for (size_t x = 0; x <= videoWidth-modelWidth; x+=stride){
				//　矩形に切り出したものを作成 imagesはinputShapeと同じサイズ
				std::vector<cv::Mat> modelInputImages;
				for (auto f : inputframes){
					cv::Mat subimg = cv::Mat(f, cv::Rect(x,y,modelHeight,modelWidth));
					modelInputImages.push_back(subimg);
				}
				
				// CVMatからtensorに変換
		    	tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,modelFrames,modelHeight,modelWidth,modelChannel}));
				convertImagesTensorFromCVMat(modelInputImages, tensor, inputShape);

				// 推論
				std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_data;
				inputs_data.push_back({inputName, tensor});

				std::vector<tensorflow::Tensor> predictions;
				status = bundle.session->Run(inputs_data, outputNames, {}, &predictions);
				
				if (!status.ok()){
					std::cerr << "Error in running session" << std::endl;
					return -1;
				}
				//printTensor(predictions[0]);

				// inputShapeと同じサイズのcv::Matに変換
				std::vector<cv::Mat> img_mu;
				convertCVMatFromImagesTensor(predictions[0], img_mu, inputShape);
				
				// 出力用のcv::Matに結果をコピー
				if (anomalyScore){
					// 異常度の計算
					std::vector<cv::Mat> img_sigma;
					convertCVMatFromImagesTensor(predictions[1], img_sigma, inputShape);
					
					std::vector<cv::Mat>::iterator it_inputs = modelInputImages.begin();
					std::vector<cv::Mat>::iterator it_img_mu = img_mu.begin();
					std::vector<cv::Mat>::iterator it_img_sigma = img_sigma.begin();
					std::vector<cv::Mat>::iterator it_outputs = outputs.begin();
					for (unsigned k=0; k<modelFrames; k++){
						cv::Rect roi(x,y,modelHeight,modelWidth);
						cv::Mat dest_roi = cv::Mat(*it_outputs, roi);
						cv::Mat score(modelHeight, modelWidth, CV_32FC3, cv::Scalar(0));
						cv::Mat div = 
						score = (((*it_inputs)-(*it_img_mu)).mul((*it_inputs)-(*it_img_mu)))/(*it_img_sigma);
						cv::Mat sum_of_score(1,1,CV_32FC3, cv::sum(score).dot(cv::Scalar::ones()));///modelHeight/modelWidth/modelChannel
						double value =sum_of_score.at<float>(0,0);
						
						if(value>1.0){
							over = over + 1;
						}
						floatArray.push_back(value);
						// std::cout <<"異常スコア"<< sum_of_score <<"座標"<< x<<" "<<y << std::endl;
						gokei = gokei + value;
						//std::cout <<std::setprecision(10)<<"は"<< gokei << std::endl;
						count = count + 1;
						score.copyTo(dest_roi);
                        
						// bool hasValueGreaterThanOne = false;
						// for (int o = 0; o < score.rows; o++) {
                        //     for (int p = 0; p < score.cols; p++) {
                        //         for (int s = 0; s < score.channels(); s++) {
                        //             if (score.at<cv::Vec3i>(o, p)[s] > 1) {
                        //                 hasValueGreaterThanOne = true;
						// 				std::cout<<score.at<cv::Vec3i>(o, p)[s]<<std::endl;
                        //                 break;
                        //             }
                        //         }
                                
                        //     }
                            
                        // }

                        // if (hasValueGreaterThanOne) {
                        //     std::cout << "三次元行列内に1以上の値が含まれています" << std::endl;
                        // } else {
                        //       std::cout << "三次元行列内に1以上の値は含まれていません" << std::endl;
                        // }
						//std::cout << score << std::endl;

						it_inputs++;
						it_outputs++;
						it_img_mu++;
						it_img_sigma++;
					}

				}else{
					// 単純なVAEの出力(mu)
					std::vector<cv::Mat>::iterator it_outputs = outputs.begin();
					std::vector<cv::Mat>::iterator it_img_mu = img_mu.begin();
					for (unsigned k=0; k<modelFrames; k++){
						cv::Rect roi(x,y,modelHeight,modelWidth);
						cv::Mat dest_roi = cv::Mat(*it_outputs, roi);
						it_img_mu->copyTo(dest_roi);
						it_outputs++;
						it_img_mu++;
					}
				}
			}
		}

		for (cv::Mat frame : outputs){
			cv::Mat m;
			frame.convertTo(m, CV_8UC3, 255.0);//255
			writer << m;
			cv::Mat tmp_score(1, 1, CV_32FC3, cv::sum(m).dot(cv::Scalar::ones()));
			double  value2 =tmp_score.at<float>(0,0);
			gokei2 = gokei2 +value2;

		}

	}
	std::cout <<"データ量は"<< count << std::endl;
	std::cout <<"超えた値は"<< over << std::endl;
    std::cout <<std::setprecision(10)<<"異常スコアは"<< gokei << std::endl;
	std::cout <<std::setprecision(10)<<"異常スコアは"<< gokei2 << std::endl;
	std::sort(floatArray.begin(), floatArray.end(), std::greater<float>());
	std::cout <<"floatArray"<<floatArray[0]<<std::endl;
	std::cout <<"floatArray"<<floatArray[1]<<std::endl;
	std::cout <<"floatArray"<<floatArray[100]<<std::endl;
	writer.release();

	return 0;
}




int heatmap(
			const std::string modelpath = "/content/VAE_movie/tmpdir/vae_movie/1/",
			const std::string inputmovie = "/content/data/movies/test/S_ball_room.mp4",
			const std::string outputmovie = "output.mov"
	){
	// tensorflow::SavedModelBundle bundle;
	// tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
	// tensorflow::RunOptions run_options = tensorflow::RunOptions();


	// session_options.config.mutable_gpu_options()->set_allow_growth(true);	
	//auto status = tensorflow::LoadSavedModel(session_options, run_options, path, {"serve"}, &bundle);
	// auto status = tensorflow::LoadSavedModel(session_options, run_options, modelpath, {tensorflow::kSavedModelTagServe}, &bundle);
	// if (!status.ok()){
	// 	std::cerr << "Error in loading image from file" << std::endl;
	// 	return -1;
	// }
	// else{
	// 	std::clog << "Loaded correctly!" << std::endl;
	// }
	// printModelStatus(bundle);

	/*
	 *	動画用意
	*/
    cv::VideoCapture cap;
    cap.open(inputmovie.c_str());
    if (cap.isOpened() == false){
        std::cerr << "can't open file" << std::endl;
        return -1;
    }
    unsigned videoHeight = static_cast<unsigned>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    unsigned videoWidth  = static_cast<unsigned>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    unsigned videoFrames = static_cast<unsigned>(cap.get(cv::CAP_PROP_FRAME_COUNT));
	//unsigned videoFrames = 30;
	// unsigned fps = static_cast<unsigned>(cap.get(cv::CAP_PROP_FPS));
	unsigned fps = 30;
	unsigned ex = static_cast<unsigned>(cap.get(cv::CAP_PROP_FOURCC)); 

	unsigned modelFrames = 6;
	unsigned modelHeight = 64;
	unsigned modelWidth = 64;
	unsigned modelChannel = 3; 
	//std::cout << "frames:"  << videoFrames << std::endl;
    
	//cv::VideoWriter writer(outputmovie.c_str(), ex, fps, cv::Size((videoWidth-modelWidth)/stride+((videoWidth-modelWidth)%stride!=0), (videoHeight-modelHeight)/stride+((videoHeight-modelHeight)%stride!=0)), true); 
    cv::VideoWriter writer(outputmovie.c_str(), ex, fps, cv::Size(videoWidth, videoHeight), true);
	int over = 0;
    int count = 0;
	// input/outpu name
	// auto model_def = bundle.GetSignatures().at("serving_default");
	// const std::string inputName   = model_def.inputs().at("input_1").name();; // input
	// std::vector<std::string> outputNames = {model_def.outputs().at("output_1").name(), // mu,
	// 					 					model_def.outputs().at("output_2").name()}; // sigma

	for (unsigned i=0; i<static_cast<int>(videoFrames/modelFrames); i++){
	//for (unsigned i=0; i<10; i++){
		std::cerr << i << std::endl;
		std::vector<cv::Mat> inputframes;

		// modelFrames分のフレームを取得
		int j=0;
		while (j<modelFrames){
		   	cv::Mat frame;
			cap >> frame;
			if (frame.rows == videoHeight && frame.cols == videoWidth){
				cv::Mat anImage(frame.rows, frame.cols, CV_32FC3);
				frame.convertTo(anImage, CV_32FC3, 1.0/255.0);
				inputframes.push_back(anImage);
				j++;
			}
		}
		// outputsを生成

		
		// std::vector<cv::Mat> outputs;
		// for (unsigned i=0; i<modelFrames; i++){
		// 	cv::Mat zero = cv::Mat::zeros((videoHeight-modelHeight)/stride+((videoHeight-modelHeight)%stride!=0), (videoWidth-modelWidth)/stride+((videoWidth-modelWidth)%stride!=0), CV_32FC3);
		// 	//cv::Mat zero = cv::Mat::zeros(1000, 1000, CV_32FC3);
		// 	outputs.push_back(zero);
		// }
		size_t stride = 64;
		

		// modelHeight, modelWidthの矩形に切り出していく
		std::vector<unsigned> inputShape = {modelFrames, modelHeight, modelWidth, modelChannel};
		for (size_t y = 0; y <= videoHeight-modelHeight; y+=stride){
			for (size_t x = 0; x <= videoWidth-modelWidth; x+=stride){
				//　矩形に切り出したものを作成 imagesはinputShapeと同じサイズ
				std::vector<cv::Mat> modelInputImages;
				for (auto f : inputframes){
					cv::Mat subimg = cv::Mat(f, cv::Rect(x,y,modelHeight,modelWidth));
					modelInputImages.push_back(subimg);
				}
				cv::Mat freq(1,1,CV_32FC3, cv::sum(modelInputImages).dot(cv::Scalar::ones()));
				double value =freq.at<float>(0,0);
				std::cout <<"ああああ"<<freq<<std::endl;
				count = count + 1;
				if(value > 1.0){
					over = over + 1;
				}
				// // CVMatからtensorに変換
		    	// tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,modelFrames,modelHeight,modelWidth,modelChannel}));
				// convertImagesTensorFromCVMat(modelInputImages, tensor, inputShape);

				// // 推論
				// std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_data;
				// inputs_data.push_back({inputName, tensor});

				// std::vector<tensorflow::Tensor> predictions;
				// status = bundle.session->Run(inputs_data, outputNames, {}, &predictions);
				
				// if (!status.ok()){
				// 	std::cerr << "Error in running session" << std::endl;
				// 	return -1;
				// }
				// //printTensor(predictions[0]);

				// // inputShapeと同じサイズのcv::Matに変換
				// std::vector<cv::Mat> img_mu;
				// convertCVMatFromImagesTensor(predictions[0], img_mu, inputShape);
				
				// // 出力用のcv::Matに結果をコピー
				
				// 	// 異常度の計算
				// 	std::vector<cv::Mat> img_sigma;
				// 	convertCVMatFromImagesTensor(predictions[1], img_sigma, inputShape);
					
				// 	std::vector<cv::Mat>::iterator it_inputs = modelInputImages.begin();
				// 	std::vector<cv::Mat>::iterator it_img_mu = img_mu.begin();
				// 	std::vector<cv::Mat>::iterator it_img_sigma = img_sigma.begin();
				// 	std::vector<cv::Mat>::iterator it_outputs = outputs.begin();

				// 	//std::cout << "before for frame" <<x <<" "<< y << std::endl;
				// 	for (unsigned k=0; k<modelFrames; k++){
				// 		//if( y == 960) std::cout << "before roi" <<x <<" "<< y << std::endl;
				// 		cv::Rect roi(x/stride,y/stride,1,1);
				// 		//if( y == 960) std::cout << "before destroi" <<x <<" "<< y << std::endl;
				// 		cv::Mat dest_roi = cv::Mat(*it_outputs, roi);
				// 		//if( y == 960) std::cout << "before score" <<x <<" "<< y << std::endl;
				// 		cv::Mat score(modelHeight, modelWidth, CV_32FC3, cv::Scalar(0));
				// 		//if( y == 960) std::cout << "before div" <<x <<" "<< y << std::endl;
				// 		cv::Mat div = 
				// 		score = (((*it_inputs)-(*it_img_mu)).mul((*it_inputs)-(*it_img_mu)))/(*it_img_sigma);
				// 		//if( y == 960) std::cout << "before sum" <<x <<" "<< y << std::endl;
				// 		cv::Mat sum_of_score(1,1,CV_32FC3, cv::sum(score).dot(cv::Scalar::ones())/modelHeight/modelWidth);///64
				// 		//if( y == 960) std::cout << "before copy" <<x <<" "<< y << std::endl;
				// 		std::cout <<"異常スコア"<< sum_of_score <<"座標"<< x<<" "<<y << std::endl;
				// 		sum_of_score.copyTo(dest_roi);

                        
				// 		// bool hasValueGreaterThanOne = false;
				// 		// for (int o = 0; o < score.rows; o++) {
                //         //     for (int p = 0; p < score.cols; p++) {
                //         //         for (int s = 0; s < score.channels(); s++) {
                //         //             if (score.at<cv::Vec3i>(o, p)[s] > 1) {
                //         //                 hasValueGreaterThanOne = true;
				// 		// 				std::cout<<score.at<cv::Vec3i>(o, p)[s]<<std::endl;
                //         //                 break;
                //         //             }
                //         //         }
                                
                //         //     }
                            
                //         // }

                //         // if (hasValueGreaterThanOne) {
                //         //     std::cout << "三次元行列内に1以上の値が含まれています" << std::endl;
                //         // } else {
                //         //       std::cout << "三次元行列内に1以上の値は含まれていません" << std::endl;
                //         // }
				// 		//std::cout << score << std::endl;

				// 		it_inputs++;
				// 		it_outputs++;
				// 		it_img_mu++;
				// 		it_img_sigma++;
				// 	}

				}
			}
		
		// std::cout << "before write" << std::endl;
		// for (cv::Mat frame : outputs){
		// 	cv::Mat m;
		// 	frame.convertTo(m, CV_8UC3, 255.0);//255
		// 	writer << m;
		}

	

	// writer.release();
    std::cout<<count<<std::endl;
	std::cout<<over<<std::endl;
	return 0;
}















int main(int argc, char* argv[]) {
	process(true, "/content/VAE_movie/tmpdir/vae_movie/1/", "/content/data/movies/overtest/detectoutput5.mp4", "h_overHD_detectouput5_10000e.mp4");
//(true,false)
    //heatmap("/content/VAE_movie/tmpdir/vae_movie/1/", "/content/data/movies/overtest/Circ001_t.mp4", "h_overfullHD_Circ00111_10000e.mp4");
	return 0;
}