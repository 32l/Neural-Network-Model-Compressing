
#include<iostream>
#include<vector>
#include<fstream>
#include<memory>
#include<map>
#include<cmath>
#include<stdio.h>

using namespace std;

enum Layer_type {SQUEEZE3x3, OTHER};

int get_wb_stored_num(const int count, Layer_type lt) {
  if (lt == SQUEEZE3x3){
    return (count-1)/3 +1 ;
  } else{
    return (count-1)/4 +1 ;
  }
}

std::map<unsigned char, float> get_code_book(const char min, const int bits){
  std::map<unsigned char, float> code_book;

  for (unsigned char i = 0; i< (1<< (bits-1))-1; ++i){
    code_book.insert(pair<unsigned char, float>(i, -pow(2.0, min+i)));
  }
  code_book.insert(pair<unsigned char, float>((1<< (bits-1))-1, 0.0));

  unsigned char mid = 1<< (bits-1); // 8
  for (unsigned char i = mid; i< (1<< bits)-1; ++i){
    code_book.insert(pair<unsigned char, float>(i, pow(2.0, min+i-mid)));
  }
  map<unsigned char, float>::iterator iter;
  for (iter = code_book.begin(); iter != code_book.end(); ++iter){
    printf("---- %x, %f \n",iter->first, iter->second);
  }
  return code_book;
}

void read_half_layer(ifstream &in, int layer_id, vector<char> &min_exp_w, vector<int> &count_w, 
  vector<vector<unsigned short> > &weights_s, vector<vector<float> > &weights){

  int init_exp = -100;
  char tmp_c;
  int tmp_i;
  //shared_ptr<unsigned char> tmp_short_ptr;
  shared_ptr<unsigned short> tmp_short_ptr;
  // read the minimum exp for this block of data
  // min_exp_w.resize(layer_id, init_exp);
  cout << "-- reading min_exp ..." << endl;
  in.read(&tmp_c, sizeof(char));
  printf("-- min_exp = %d ...\n", tmp_c);
  min_exp_w.push_back(tmp_c);
  cout << "-- forming code_book ..." << endl;
  map<unsigned char, float> code_book;
  code_book = get_code_book(tmp_c, 4);
  // total number of data
  cout << "-- reading param count ..." << endl;
  in.read(reinterpret_cast<char *>(&tmp_i), sizeof(int));
  cout << "-- param count: " << tmp_i << endl;
  count_w.push_back(tmp_i);
  weights_s.resize(layer_id);
  weights.resize(layer_id);
  cout << "-- reading params ..." << endl;

  // squeeze 3x3 layer
  if ((layer_id-1)%3 == 0 && layer_id != 1){
    // read the short 
    int short_len = get_wb_stored_num(count_w[layer_id-1], SQUEEZE3x3);
    cout << "-- 3x3 short_len = " << short_len << endl;
    // weights_s[layer_id-1].resize(short_len, 0);
    tmp_short_ptr.reset(new unsigned short(short_len));
    cout << "-- reading 3x3 params ... " << endl;
    in.read(reinterpret_cast<char *>(tmp_short_ptr.get()), short_len * sizeof(unsigned short)/sizeof(char));
    for (int i=0; i< short_len; ++i){
      weights_s[layer_id-1].push_back(tmp_short_ptr.get()[i]);
    }
    // store the real values to weights
    for (int i= 0; i< short_len; ++i){
      unsigned short tmp_short = weights_s[layer_id-1][i];
      // first
      unsigned char code = 0xF & tmp_short;
      weights[layer_id-1].push_back(code_book.find(code)->second);
      // second
      code = 0xF & (tmp_short >> 4);
      weights[layer_id-1].push_back(code_book.find(code)->second);
      // third
      code = 0xF & (tmp_short >> 8);
      weights[layer_id-1].push_back(code_book.find(code)->second);
    }
    // delete the last few redundant values
    if(short_len % 3 == 1){
      weights.pop_back();
      weights.pop_back();
    } else if (short_len % 3 == 2) {
      weights.pop_back();
    }
  } else { // normal layer or squeeze1x1 layer
    // read the short 
    int short_len = get_wb_stored_num(count_w[layer_id-1], OTHER);
    cout << "-- normal short_len = " << short_len << endl;
    // weights_s[layer_id-1].resize(short_len, 0);
    tmp_short_ptr.reset(new unsigned short(short_len));
    cout << "-- reading normal params ... " << endl;
    in.read(reinterpret_cast<char *>(tmp_short_ptr.get()), short_len );

    cout << "-- moving char to weights_s... " << endl;
    for (int i = 0; i< 4; i++){
      printf("-- ushort[%d] = %x \n", i, tmp_short_ptr.get()[i] );
    }
    
    for (int i=0; i< short_len; ++i){
      weights_s[layer_id-1].push_back(tmp_short_ptr.get()[i]);
    }
    // store the real values to weights
    cout << "-- storing the params... " << endl;
    for (int i= 0; i< short_len; ++i){
      unsigned short tmp_short = weights_s[layer_id-1][i];
      // 1st
      unsigned char code = 0xF & tmp_short;
      weights[layer_id-1].push_back(code_book.find(code)->second);
      // 2nd
      code = 0xF & (tmp_short >> 4);
      weights[layer_id-1].push_back(code_book.find(code)->second);

      // 3rd
      code = 0xF & (tmp_short >> 8);
      weights[layer_id-1].push_back(code_book.find(code)->second);
      // 4th
      code = 0xF & (tmp_short >> 12);
      weights[layer_id-1].push_back(code_book.find(code)->second);

    }
    // delete the last few redundant values
    if(short_len % 2 == 1){
      weights.pop_back();
    }
  }
}


int main(int argc, char** argv){
  // check for the name of input binary files;
  if (argc < 2) {
    cout << "Error: no input file name provided!";
  }
  char *filename = argv[1];
  vector<char> min_exp_w;
  vector<char> min_exp_b;
  vector<int> count_w;
  vector<int> count_b;
  vector<vector<unsigned short> > weights_s;
  vector<vector<unsigned short> > bias_s;
  vector<vector<float> > weights;
  vector<vector<float> > bias;

  ifstream in (filename, ios::in | ios::binary );
  if (! in.is_open()){
    cout<< "Error while opening file!" << endl;
    exit(-1);
  }
  // read data
  int layer_id = 0;
  std::cout<< "reading data..." << endl;  

  while (! in.eof()){
    layer_id++;
    // read the weights
    cout << "reading weights..." << endl;
    weights_s.resize(layer_id);
    weights.resize(layer_id);
    read_half_layer(in, layer_id, min_exp_w, count_w, weights_s, weights);
    // read the bias
    cout << "reading bias ... " << endl;
    bias_s.resize(layer_id);
    bias.resize(layer_id);
    read_half_layer(in, layer_id, min_exp_b, count_b, bias_s, bias);
  }
  // close the file 
  in.close();
  // output some data for validation
  for (int i = 0; i< count_w.size(); ++i){
    std::cout << "layer " << i+1 << " weight count: " << count_w[i] << endl;
  }

  return 0;
}
