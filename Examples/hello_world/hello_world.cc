/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW2
#include "ORBVocabulary.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Python.h>
#include <numpy/arrayobject.h>

#include <cstdlib>

using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features, string image_dir, string image_list_file);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &features);
// void testDatabase(const vector<vector<cv::Mat > > &features);


int call_py(const char *fileName, const char *funcName, cv::Mat img1, cv::Mat img2, cv::Mat &descriptors, bool &fail)
{
    // descriptors.clear();

    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue, *output_array;
    import_array();

    npy_intp dims1[] = {img1.rows, img1.cols};
    npy_intp dims2[] = {img2.rows, img2.cols};
    uint8_t *ptr1 = img1.ptr<uint8_t>(0);
    uint8_t *ptr2 = img2.ptr<uint8_t>(0);

    // in_array_1 = PyArray_SimpleNewFromData(2, dims1, NPY_UINT8, ptr1);
    // in_array_2 = PyArray_SimpleNewFromData(2, dims2, NPY_UINT8, ptr2);

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
    pName = PyUnicode_DecodeFSDefault(fileName);
    /* Error checking of pName left out */

    // std::cout << "Finished initialization...\n";

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, funcName);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(2);
            // pValue = PyUnicode_FromString("arg1");
            pValue = PyArray_SimpleNewFromData(2, dims1, NPY_UINT8, ptr1);
            // std::cout << "&&&&&&&&&&&&&&&&\n";

            if (!pValue) {
                Py_DECREF(pArgs);
                Py_DECREF(pModule);
                fprintf(stderr, "Cannot convert argument\n");
                return 1;
            }
            /* pValue reference stolen here: */
            PyTuple_SetItem(pArgs, 0, pValue);
            // pValue = PyUnicode_FromString("arg2");
            pValue = PyArray_SimpleNewFromData(2, dims2, NPY_UINT8, ptr2);

            if (!pValue) {
                Py_DECREF(pArgs);
                Py_DECREF(pModule);
                fprintf(stderr, "Cannot convert argument\n");
                return 1;
            }
            /* pValue reference stolen here: */
            PyTuple_SetItem(pArgs, 1, pValue);
            // std::cout << "Ready to call function\n";
            pValue = PyObject_CallObject(pFunc, pArgs);

            // double val1 = PyFloat_AsDouble(PyList_GetItem(output_array, 0));
            // double val2 = PyFloat_AsDouble(PyList_GetItem(output_array, 1));

            // std::cout << "value1 : " << val1 << "\n";
            // std::cout << "value2 : " << val2 << "\n";

            // output_array = PyList_GetItem(pValue, 1);
            // val1 = PyFloat_AsDouble(PyList_GetItem(output_array, 0));
            // val2 = PyFloat_AsDouble(PyList_GetItem(output_array, 1));

            // std::cout << "value1 : " << val1 << "\n";
            // std::cout << "value2 : " << val2 << "\n";

            output_array = PyList_GetItem(pValue, 0);
            double N = PyFloat_AsDouble(output_array);

            std::cout << "int(N / 2): " << int(N / 2) << "\n";

            if(int(N / 2) < 384)
            {
                fail = true;
                return -1;
            }
            else if(int(N / 2) > 500)
            {
              N = 1000;
            }

            descriptors.create(int(N / 2), 384, CV_64FC1);

            std::cout << "descriptors.size(): " << descriptors.size() << "\n";
            // abort();

            // output_array = PyList_GetItem(pValue, 1);
            // std::vector<std::vector<double>> mkpts0;
            // for(int i = 0; i < N; i++)
            // {
            //     std::vector<double> row;

            //     row.push_back(PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(output_array, i), 0)));
            //     row.push_back(PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(output_array, i), 1)));

            //     mkpts0.push_back(row);
            // }

            // output_array = PyList_GetItem(pValue, 2);
            // std::vector<std::vector<double>> mkpts1;
            // for(int i = 0; i < N; i++)
            // {
            //     std::vector<double> row;

            //     row.push_back(PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(output_array, i), 0)));
            //     row.push_back(PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(output_array, i), 1)));

            //     mkpts1.push_back(row);
            // }

            output_array = PyList_GetItem(pValue, 3);
            // std::vector<std::vector<double>> fm0;
            for(int i = 0; i < int(N / 2); i++)
            {
                std::vector<double> row;

                for(int j = 0; j < 384; j++)
                    // row.push_back(PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(output_array, i), j)));
                    descriptors.at<double>(j, i) = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(output_array, i), j));

                // fm0.push_back(row);
            }

            // output_array = PyList_GetItem(pValue, 4);
            // std::vector<std::vector<double>> fm1;
            // for(int i = 0; i < N; i++)
            // {
            //     std::vector<double> row;

            //     for(int j = 0; j < 348; j++)
            //         row.push_back(PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(output_array, i), j)));

            //     fm1.push_back(row);
            // }

            // Visualize vector
            // for(int i = 0; i < N; i++)
            // {
            //     std::cout << mkpts0.at(i).at(0) << "\t" << mkpts0.at(i).at(1) << "\n";
            //     std::cout << mkpts1.at(i).at(0) << "\t" << mkpts1.at(i).at(1) << "\n";
            //     for(int j = 0; j < 348; j++)
            //         std::cout << fm0.at(i).at(j) << " ";
            //     std::cout << "\n";
            //     for(int j = 0; j < 348; j++)
            //         std::cout << fm1.at(i).at(j) << " ";
            //     std::cout << "\n";
            //     std::cout << "==============\n";
            // }

            Py_DECREF(pArgs);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", funcName);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", fileName);
        return 1;
    }
    return 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main(int argc, char** argv)
{
  string image_dir = argv[1], image_list_file = argv[2];

  // cout << (unsigned long int)time(NULL) << "\n";

  Py_Initialize();

  vector<vector<cv::Mat>> features;
  loadFeatures(features, image_dir, image_list_file);
  cout << "*************************** features size: " << features.size() << "\n";
//   abort();
  testVocCreation(features);

//   testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features, string image_dir, string image_list_file)
{
  cv::theRNG().state = (unsigned long int)time(NULL);

  features.clear();

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;

  string image_file_name;

  ifstream f(image_list_file);

  while(f >> image_file_name)
  {
    cv::Mat image_left = cv::imread(image_dir + "/images_left/" + image_file_name, 0);
    cv::Mat image_right = cv::imread(image_dir + "/images_right/" + image_file_name, 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;

    // Use random feature vector as descriptor
    // cv::Mat descriptors(1200, 32, CV_16U);         // ORB size random feature vector
    // cv::Mat descriptors(1200, 48, CV_16U);            // LoFTR size random feature vector

    // cv::randu(descriptors, cv::Scalar(0), cv::Scalar(300));

    bool fail = false;
    cv::Mat descriptors;
    call_py("SLAM_Matcher", "match_images", image_left, image_right, descriptors, fail);

    // std::cout << "Size of descriptors: " << descriptors.size() << "\n";

    // abort();

    // Use ORB descriptor
    // cv::Mat descriptors;
    // orb->detectAndCompute(image_left, mask, keypoints, descriptors);

    if(!fail)
    {
      features.push_back(vector<cv::Mat >());
      changeStructure(descriptors, features.back());
    }

    std::cout << "\tCompleted processing image: " << image_file_name << "\n";
  }

  // for(int i = 0; i < NIMAGES; ++i)
  // {
  //   stringstream ss;
  //   ss << "demo/images/image" << i << ".png";

  //   cv::Mat image = cv::imread(ss.str(), 0);
  //   cv::Mat mask;
  //   vector<cv::KeyPoint> keypoints;
  //   cv::Mat descriptors;

  //   orb->detectAndCompute(image, mask, keypoints, descriptors);

  //   features.push_back(vector<cv::Mat >());
  //   changeStructure(descriptors, features.back());
  // }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  int NIMAGES = features.size();

  // branching factor and depth levels 
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  ORB_SLAM2::ORBVocabulary voc(k, L, weight, scoring);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  // cout << "Matching images against themselves (0 low, 1 high): " << endl;
  // BowVector v1, v2;
  // for(int i = 0; i < NIMAGES; i++)
  // {
  //   voc.transform(features[i], v1);
  //   for(int j = 0; j < NIMAGES; j++)
  //   {
  //     voc.transform(features[j], v2);
      
  //     double score = voc.score(v1, v2);
  //     cout << "Image " << i << " vs Image " << j << ": " << score << endl;
  //   }
  // }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  // voc.save("small_voc.yml.gz");
//   voc.save("loftr_voc.txt");
  voc.saveToTextFile("loft_voc.txt");

  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

// void testDatabase(const vector<vector<cv::Mat > > &features)
// {
//   int NIMAGES = features.size();

//   cout << "Creating a small database..." << endl;

//   // load the vocabulary from disk
//   OrbVocabulary voc("small_voc.yml.gz");
  
//   OrbDatabase db(voc, false, 0); // false = do not use direct index
//   // (so ignore the last param)
//   // The direct index is useful if we want to retrieve the features that 
//   // belong to some vocabulary node.
//   // db creates a copy of the vocabulary, we may get rid of "voc" now

//   // add images to the database
//   for(int i = 0; i < NIMAGES; i++)
//   {
//     db.add(features[i]);
//   }

//   cout << "... done!" << endl;

//   cout << "Database information: " << endl << db << endl;

//   // and query the database
//   cout << "Querying the database: " << endl;

//   QueryResults ret;
//   for(int i = 0; i < NIMAGES; i++)
//   {
//     db.query(features[i], ret, 4);

//     // ret[0] is always the same image in this case, because we added it to the 
//     // database. ret[1] is the second best match.

//     cout << "Searching for Image " << i << ". " << ret << endl;
//   }

//   cout << endl;

//   // we can save the database. The created file includes the vocabulary
//   // and the entries added
//   cout << "Saving database..." << endl;
//   db.save("small_db.yml.gz");
//   cout << "... done!" << endl;
  
//   // once saved, we can load it again  
//   cout << "Retrieving database once again..." << endl;
//   OrbDatabase db2("small_db.yml.gz");
//   cout << "... done! This is: " << endl << db2 << endl;
// }

// ----------------------------------------------------------------------------


